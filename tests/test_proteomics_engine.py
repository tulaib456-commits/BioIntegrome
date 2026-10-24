"""Tests for the Proteomics calculation engine (missing values, normalization, DE on log-ratio data)."""
import numpy as np
import pandas as pd
import pytest

from tests.sim_data import make_proteomics
from modules.proteomics.missing_value_handler import MissingValueAnalyzer, ProteomicsFilter, Imputer
from modules.proteomics.normalization import ProteomicsNormalizer
from modules.transcriptomics.differential_expression import DifferentialExpression
from modules.transcriptomics.multi_group_de import MultiGroupDE


@pytest.fixture(scope="module")
def prot():
    return make_proteomics()


def sample_cols(df):
    return [c for c in df.columns if c != "Gene"]


# ------------------------------------------------------------ missing values
def test_missingness_summary_counts_are_exact():
    df = pd.DataFrame({"Gene": ["a", "b"], "S1": [1.0, np.nan], "S2": [2.0, np.nan], "S3": [np.nan, 3.0]})
    out = MissingValueAnalyzer(df, "Gene", ["S1", "S2", "S3"]).missingness_summary()
    assert out["Valid_Count"].tolist() == [2, 1]
    assert out["Missing_Fraction"].round(3).tolist() == [0.333, 0.667]


def test_group_detection_filter_keeps_all_or_nothing_proteins():
    """Protein seen in 100% of A and 0% of B must be KEPT (that is real biology)."""
    df = pd.DataFrame({"Gene": ["onlyA", "sparse", "full"],
                       "A_1": [5.0, 5.0, 5.0], "A_2": [5.1, np.nan, 5.0], "A_3": [4.9, np.nan, 5.2],
                       "B_1": [np.nan, np.nan, 5.0], "B_2": [np.nan, 4.0, 5.1], "B_3": [np.nan, np.nan, 4.9]})
    groups = {"A_1": "A", "A_2": "A", "A_3": "A", "B_1": "B", "B_2": "B", "B_3": "B"}
    out = ProteomicsFilter(df, "Gene", list(groups)).filter_by_group_detection(groups, 0.7)
    assert out["Gene"].tolist() == ["onlyA", "full"]


# ------------------------------------------------------------ imputation
def test_imputation_leaves_no_missing_values(prot):
    df, groups, truth = prot
    out = Imputer().impute(df, sample_cols(df))
    assert not out[sample_cols(df)].isna().any().any()


def test_imputation_never_changes_observed_values(prot):
    df, groups, truth = prot
    s = sample_cols(df)
    out = Imputer().impute(df, s)
    obs = df[s].notna().values
    assert np.allclose(out[s].values[obs], df[s].values[obs])


def test_imputed_values_sit_below_the_bulk_of_observed_values_like_perseus(prot):
    """MNAR logic: missing = below detection. Imputed values must come from the LOW END
    of each sample's distribution (mean - ~1.8 SD), not from a protein's own average."""
    df, groups, truth = prot
    s = sample_cols(df)
    out = Imputer().impute(df, s)
    miss = df[s].isna().values
    imputed = out[s].values[miss]
    observed = df[s].values[~miss]
    assert imputed.mean() < observed.mean() - 1.0 * observed.std()


def test_imputation_is_reproducible(prot):
    df, groups, truth = prot
    s = sample_cols(df)
    a = Imputer(random_state=7).impute(df, s)
    b = Imputer(random_state=7).impute(df, s)
    assert a[s].equals(b[s])


def test_all_or_nothing_protein_is_detected_after_imputation():
    """A protein measured in all of group A and entirely missing in group B (below detection)
    is the classic proteomics discovery. It must come out as a large, significant change."""
    rng = np.random.default_rng(4)
    n_prot, n = 400, 12
    vals = rng.normal(0, 1, (n_prot, 1)) + rng.normal(0, 0.3, (n_prot, n))
    vals[0, :6] = rng.normal(2.0, 0.3, 6)   # abundant in A
    vals[0, 6:] = np.nan                     # absent in B
    cols = [f"A_{i}" for i in range(6)] + [f"B_{i}" for i in range(6)]
    df = pd.DataFrame(vals, columns=cols); df.insert(0, "Gene", [f"P{i}" for i in range(n_prot)])
    imp = Imputer().impute(df, cols)
    res = DifferentialExpression(imp, cols[:6], cols[6:], already_normalized=True).run(log2fc_threshold=0.0)
    row = res[res.Gene == "P0"].iloc[0]
    assert row["Adj_P_Value"] < 0.05
    assert row["Log2FC"] < -1.5, f"effect size understated: {row['Log2FC']:.2f}"


def test_protein_with_every_value_missing_does_not_stay_nan_after_imputation():
    cols = ["A_1", "A_2", "B_1", "B_2"]
    rng = np.random.default_rng(0)
    vals = rng.normal(0, 1, (30, 4)); vals[3, :] = np.nan
    df = pd.DataFrame(vals, columns=cols); df.insert(0, "Gene", [f"P{i}" for i in range(30)])
    out = Imputer().impute(df, cols)
    assert not out[cols].isna().any().any()


# ------------------------------------------------------------ normalization
def test_median_normalization_equalises_sample_medians(prot):
    df, *_ = prot
    s = sample_cols(df)
    out = ProteomicsNormalizer(df, s).median_normalization()
    meds = out[s].median()
    assert meds.max() - meds.min() < 1e-9


def test_median_normalization_preserves_within_sample_differences(prot):
    df, *_ = prot
    s = sample_cols(df)
    out = ProteomicsNormalizer(df, s).median_normalization()
    d_before = (df[s[0]] - df[s[0]].median()).dropna()
    d_after = (out[s[0]] - out[s[0]].median()).dropna()
    assert np.allclose(d_before.values, d_after.values)


def test_quantile_normalization_gives_identical_distributions():
    rng = np.random.default_rng(0)
    vals = rng.normal(0, 1, (300, 5)) + np.array([0, 1, -1, 2, 0.5])
    cols = [f"S{i}" for i in range(5)]
    df = pd.DataFrame(vals, columns=cols); df.insert(0, "Gene", [f"P{i}" for i in range(300)])
    out = ProteomicsNormalizer(df, cols).quantile_normalization()
    sorted_cols = np.sort(out[cols].values, axis=0)
    assert np.allclose(sorted_cols, sorted_cols[:, [0]], atol=1e-9)


def test_quantile_normalization_with_missing_values_does_not_crash_or_invent_numbers(prot):
    df, *_ = prot
    s = sample_cols(df)
    out = ProteomicsNormalizer(df, s).quantile_normalization()
    # originally-missing cells must remain missing (not be turned into numbers)
    assert out[s].isna().values[df[s].isna().values].all()


# ------------------------------------------------------------ DE on log-ratio (negative) data
def test_de_on_negative_log_ratio_data_finds_planted_proteins_using_available_values(prot):
    """Randomly missing values: using each protein's available values (no imputation)
    must find most of the truly changed proteins with few false discoveries."""
    df, groups, truth = prot
    s = sample_cols(df)
    filtered = ProteomicsFilter(df, "Gene", s).filter_by_group_detection(groups, 0.7)
    a = [c for c in s if groups[c] == "A"]; b = [c for c in s if groups[c] == "B"]
    res = DifferentialExpression(filtered, a, b, already_normalized=True).run(log2fc_threshold=0.0)
    sig = set(res.loc[res.Significant, "Gene"])
    assert len(sig & set(truth)) / len(truth) > 0.7
    assert len(sig - set(truth)) / max(len(sig), 1) < 0.10


def test_missing_values_are_handled_gene_by_gene_not_by_dropping_or_zeroing():
    rng = np.random.default_rng(0)
    cols = ["A_1", "A_2", "A_3", "B_1", "B_2", "B_3"]
    vals = rng.normal(0, 1, (50, 6)); vals[0, 3:] += 5; vals[0, 0] = np.nan
    df = pd.DataFrame(vals, columns=cols); df.insert(0, "Gene", [f"P{i}" for i in range(50)])
    res = DifferentialExpression(df, cols[:3], cols[3:], already_normalized=True).run(log2fc_threshold=0)
    row = res[res.Gene == "P0"].iloc[0]
    from scipy import stats as _st
    expected = _st.ttest_ind(vals[0, 3:], vals[0, 1:3], equal_var=False).pvalue   # only the values that exist
    assert row["N_GroupA"] == 2 and row["N_GroupB"] == 3
    assert row["P_Value"] == pytest.approx(expected, rel=1e-9)


def test_untestable_proteins_are_left_out_of_fdr_not_set_to_one():
    cols = ["A_1", "A_2", "B_1", "B_2"]
    df = pd.DataFrame({"Gene": ["ok", "onlyA", "flat"],
                       "A_1": [1.0, 2.0, 3.0], "A_2": [1.2, 2.1, 3.0],
                       "B_1": [3.0, np.nan, 3.0], "B_2": [3.3, np.nan, 3.0]})
    res = DifferentialExpression(df, cols[:2], cols[2:], already_normalized=True).run(log2fc_threshold=0)
    assert res[res.Gene == "onlyA"]["P_Value"].isna().all()
    assert res[res.Gene == "flat"]["P_Value"].isna().all()
    assert res[res.Gene == "ok"]["P_Value"].notna().all()


def test_de_on_negative_data_without_already_normalized_is_refused_not_silently_wrong(prot):
    """Documented trap: CPM math on negative values -> NaN -> p=1 -> 'zero significant results'."""
    df, groups, truth = prot
    s = sample_cols(df)
    imp = Imputer().impute(df, s)
    a = [c for c in s if groups[c] == "A"]; b = [c for c in s if groups[c] == "B"]
    with pytest.raises(ValueError, match="negative"):
        DifferentialExpression(imp, a, b, already_normalized=False).run()


def test_anova_on_negative_data_without_flag_is_refused(prot):
    df, groups, truth = prot
    s = sample_cols(df)
    imp = Imputer().impute(df, s)
    with pytest.raises(ValueError, match="negative"):
        MultiGroupDE(imp, groups, "Gene", already_normalized=False).run()
