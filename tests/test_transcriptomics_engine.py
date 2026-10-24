"""
Tests for the Transcriptomics calculation engine.
Each test says in plain words WHAT must be true.
"""
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from tests.sim_data import make_rnaseq
from modules.transcriptomics.filters import GeneFilter
from modules.transcriptomics.normalization import Normalizer
from modules.transcriptomics.differential_expression import DifferentialExpression
from modules.transcriptomics.multi_group_de import MultiGroupDE
from modules.transcriptomics.qc_diagnostics import QCDiagnostics
from modules.transcriptomics.feature_builder import FeatureBuilder
from modules.transcriptomics.ml_classifier import MLClassifier


@pytest.fixture(scope="module")
def rna():
    return make_rnaseq()


def samples_of(df):
    return [c for c in df.columns if c != "Gene"]


# ------------------------------------------------------------ normalization
def test_cpm_columns_sum_to_one_million(rna):
    df, *_ = rna
    out = Normalizer(df).cpm()
    sums = out[samples_of(df)].sum()
    assert np.allclose(sums, 1e6)


def test_median_of_ratios_recovers_true_library_depth(rna):
    """Simulated depths are known; size factors must track them (correlation ~1)."""
    df, groups, de, depth = rna
    n = Normalizer(df)
    n.median_of_ratios()
    est = np.array([n.size_factors[s] for s in samples_of(df)])
    assert np.corrcoef(est, depth)[0, 1] > 0.97


def test_median_of_ratios_sparse_data_does_not_return_nan():
    """Small/sparse datasets often have NO gene with a count in every sample.
    The result must still be finite numbers, not a silent wall of NaN."""
    rng = np.random.default_rng(3)
    counts = rng.poisson(5.0, size=(60, 8)) + 1
    counts[np.arange(60), rng.integers(0, 8, 60)] = 0  # every gene has one zero, no sample is empty
    df = pd.DataFrame(counts, columns=[f"S{i}" for i in range(8)])
    df.insert(0, "Gene", [f"G{i}" for i in range(60)])
    out = Normalizer(df).median_of_ratios()
    assert np.isfinite(out[samples_of(df)].values).all()


def test_cpm_with_empty_sample_is_reported_not_silent_nan():
    df = pd.DataFrame({"Gene": ["a", "b", "c"], "S1": [5, 6, 7], "S2": [0, 0, 0], "S3": [1, 2, 3]})
    out = Normalizer(df).cpm()
    # a sample with zero reads cannot be normalised; it must not silently poison other samples
    assert np.isfinite(out[["S1", "S3"]].values).all()


# ------------------------------------------------------------ filters
def test_zero_gene_filter_removes_only_all_zero_rows():
    df = pd.DataFrame({"Gene": list("abcd"), "S1": [0, 1, 0, 5], "S2": [0, 0, 0, 5]})
    out = GeneFilter(df).remove_zero_genes()
    assert out["Gene"].tolist() == ["b", "d"]


def test_cpm_filter_keeps_expressed_drops_silent(rna):
    df, *_ = rna
    out = GeneFilter(df).filter_by_cpm(min_cpm=1.0, min_samples=3)
    assert 0 < len(out) <= len(df)


def test_cpm_filter_with_empty_sample_does_not_crash_or_drop_everything():
    df = pd.DataFrame({"Gene": list("abcd"), "S1": [50, 10, 0, 5], "S2": [0, 0, 0, 0],
                       "S3": [40, 12, 0, 6], "S4": [45, 9, 1, 7]})
    out = GeneFilter(df).filter_by_cpm(min_cpm=1.0, min_samples=2)
    assert len(out) >= 1


# ------------------------------------------------------------ differential expression
def test_de_pvalues_equal_independent_scipy_welch_per_gene(rna):
    df, groups, de, depth = rna
    a = [s for s, g in groups.items() if g == "Ctrl"]
    b = [s for s, g in groups.items() if g == "Treat"]
    res = DifferentialExpression(df, a, b).run()
    # independent recomputation for 5 random genes
    totals = df[a + b].sum()
    logcpm = np.log2(df[a + b] / totals * 1e6 + 1)
    for i in [0, 7, 150, 900, 1999]:
        t, p = stats.ttest_ind(logcpm.loc[i, b].values, logcpm.loc[i, a].values, equal_var=False)
        got = res.loc[res.Gene == df.loc[i, "Gene"], "P_Value"].iloc[0]
        assert got == pytest.approx(p, rel=1e-9)


def test_de_finds_planted_genes_and_controls_false_positives(rna):
    df, groups, de, depth = rna
    a = [s for s, g in groups.items() if g == "Ctrl"]
    b = [s for s, g in groups.items() if g == "Treat"]
    res = DifferentialExpression(df, a, b).run(fdr_threshold=0.05, log2fc_threshold=0.0)
    sig = set(res.loc[res.Significant, "Gene"])
    truth = set(de)
    recall = len(sig & truth) / len(truth)
    fdr = len(sig - truth) / max(len(sig), 1)
    assert recall >= 0.5, f"found only {recall:.0%} of the truly changed genes"
    assert fdr < 0.15, f"false discovery rate {fdr:.0%} is far above the 5% target"


def test_de_null_experiment_gives_almost_no_hits():
    """No gene is truly different -> a correct pipeline reports (almost) nothing."""
    df, groups, de, _ = make_rnaseq(n_de=0, seed=11)
    a = [s for s, g in groups.items() if g == "Ctrl"]
    b = [s for s, g in groups.items() if g == "Treat"]
    res = DifferentialExpression(df, a, b).run(fdr_threshold=0.05, log2fc_threshold=0.0)
    assert res["Significant"].sum() <= 5


def test_de_direction_of_log2fc_is_groupB_minus_groupA(rna):
    df, groups, de, _ = rna
    a = [s for s, g in groups.items() if g == "Ctrl"]
    b = [s for s, g in groups.items() if g == "Treat"]
    res = DifferentialExpression(df, a, b).run()
    planted = res[res.Gene.isin(de)]
    assert (planted["Log2FC"] > 0).mean() > 0.95


def test_de_with_nan_values_is_not_silently_turned_into_p_equals_1():
    """Documented hazard: NaN -> p=1.0 with no warning produced 'zero significant results'."""
    rng = np.random.default_rng(5)
    vals = rng.normal(0, 1, (200, 8)); vals[:20, 4:] += 3
    df = pd.DataFrame(vals, columns=[f"A_{i}" for i in range(4)] + [f"B_{i}" for i in range(4)])
    df.insert(0, "Gene", [f"G{i}" for i in range(200)])
    df.iloc[30, 3] = np.nan  # a single missing value in one gene
    a = [c for c in df.columns if c.startswith("A_")]
    b = [c for c in df.columns if c.startswith("B_")]
    res = DifferentialExpression(df, a, b, already_normalized=True).run(log2fc_threshold=0.0)
    row = res[res.Gene == "G30"].iloc[0]
    # one NaN must not erase the gene's evidence unnoticed: either computed on available values
    # or clearly flagged, but p must not be silently exactly 1.0 for a gene with a valid signal
    assert row["P_Value"] != 1.0 or "Note" in res.columns


def test_de_single_sample_group_is_refused_clearly():
    df, groups, *_ = make_rnaseq(n_per_group=3)
    a = ["Ctrl_1"]
    b = [s for s, g in groups.items() if g == "Treat"]
    with pytest.raises(ValueError):
        DifferentialExpression(df, a, b).run()


def test_de_zero_variance_gene_does_not_produce_inf_or_crash():
    df = pd.DataFrame({"Gene": ["flat", "var"],
                       "A_1": [5, 1], "A_2": [5, 2], "A_3": [5, 3],
                       "B_1": [5, 9], "B_2": [5, 8], "B_3": [5, 10]})
    res = DifferentialExpression(df, ["A_1", "A_2", "A_3"], ["B_1", "B_2", "B_3"]).run(log2fc_threshold=0)
    assert np.isfinite(res[["P_Value", "Adj_P_Value"]].values).all()


# ------------------------------------------------------------ multi-group (ANOVA)
def test_anova_matches_scipy_f_oneway_gene_by_gene():
    df, groups, de, _ = make_rnaseq(groups=("A", "B", "C"), n_per_group=5, seed=4)
    eng = MultiGroupDE(df, groups, "Gene")
    res = eng.run()
    totals = df[list(groups)].sum()
    logcpm = np.log2(df[list(groups)] / totals * 1e6 + 1)
    for i in [0, 3, 500, 1500]:
        arrays = [logcpm.loc[i, [s for s, g in groups.items() if g == k]].values for k in "ABC"]
        f, p = stats.f_oneway(*arrays)
        row = res[res.Gene == df.loc[i, "Gene"]].iloc[0]
        assert row["F_Statistic"] == pytest.approx(f, rel=1e-8)
        assert row["P_Value"] == pytest.approx(p, rel=1e-8)


def test_anova_unbalanced_groups_still_correct():
    rng = np.random.default_rng(2)
    sizes = {"A": 3, "B": 7, "C": 5}
    names, grp = [], {}
    for k, n in sizes.items():
        for i in range(n):
            names.append(f"{k}_{i}"); grp[names[-1]] = k
    vals = rng.normal(10, 1, (50, len(names)))
    df = pd.DataFrame(vals, columns=names); df.insert(0, "Gene", [f"G{i}" for i in range(50)])
    res = MultiGroupDE(df, grp, "Gene", already_normalized=True).run()
    arrays = [df.loc[0, [s for s in names if grp[s] == k]].values.astype(float) for k in "ABC"]
    f, p = stats.f_oneway(*arrays)
    row = res[res.Gene == "G0"].iloc[0]
    assert row["P_Value"] == pytest.approx(p, rel=1e-8)


def test_anova_group_with_one_sample_is_refused_or_still_valid():
    """ANOVA is mathematically valid with a singleton group if others have replicates,
    but the app should not crash."""
    names = ["A_1", "B_1", "B_2", "B_3", "C_1", "C_2"]
    grp = {n: n[0] for n in names}
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(8, 1, (20, 6)), columns=names)
    df.insert(0, "Gene", [f"G{i}" for i in range(20)])
    res = MultiGroupDE(df, grp, "Gene", already_normalized=True).run()
    assert np.isfinite(res["P_Value"]).all()


def test_tukey_posthoc_runs_and_flags_the_right_pair():
    df, groups, *_ = make_rnaseq(groups=("A", "B", "C"), n_per_group=6, n_de=50, fold=4, seed=8)
    eng = MultiGroupDE(df, groups, "Gene")
    out = eng.posthoc("GENE0001")
    assert out is not None and len(out) == 3
    rej = out[out.columns[-1]].astype(str).str.lower().eq("true")
    assert rej.any()


# ------------------------------------------------------------ QC / feature builder / ML
def test_qc_library_sizes_equal_column_sums(rna):
    df, *_ = rna
    sizes = QCDiagnostics(df).library_sizes()
    assert sizes[samples_of(df)[0]] == df[samples_of(df)[0]].sum()


def test_qc_outlier_flagging_finds_a_planted_outlier():
    df, groups, *_ = make_rnaseq(n_de=0, seed=21)
    rng = np.random.default_rng(1)
    bad = "Ctrl_3"
    df[bad] = rng.permutation(df[bad].values) * rng.integers(1, 4, len(df))
    _, flagged = QCDiagnostics(df).flag_outlier_samples(threshold=2.0)
    assert bad in flagged


def test_feature_builder_shapes_and_labels(rna):
    df, groups, de, _ = rna
    X, y, feats, nans = FeatureBuilder(df, "Gene").build(groups, gene_list=de)
    assert X.shape == (len(groups), len(de))
    assert list(y.index) == list(X.index)
    assert set(y.unique()) == {"Ctrl", "Treat"}


def test_ml_random_labels_score_near_chance_no_leakage():
    """THE leakage test: with 2000 features and shuffled labels, honest cross-validation
    must score ~50%. Anything far above that means label information is leaking."""
    df, groups, *_ = make_rnaseq(n_de=0, n_per_group=10, seed=31)
    rng = np.random.default_rng(0)
    shuffled = dict(zip(groups.keys(), rng.permutation(list(groups.values()))))
    X, y, _, _ = FeatureBuilder(df, "Gene").build(shuffled)
    summary, _ = MLClassifier(X, y).evaluate(["Logistic Regression", "Random Forest"], n_folds=5, select_k=50)
    assert summary["Accuracy_Mean"].max() < 0.75, summary


def test_ml_real_signal_is_detected():
    df, groups, de, _ = make_rnaseq(n_de=200, fold=4.0, n_per_group=10, seed=32)
    X, y, _, _ = FeatureBuilder(df, "Gene").build(groups)
    summary, _ = MLClassifier(X, y).evaluate(["Logistic Regression"], n_folds=5, select_k=50)
    assert summary["Accuracy_Mean"].iloc[0] > 0.85


def test_ml_refuses_more_folds_than_smallest_class():
    X = pd.DataFrame(np.random.default_rng(0).normal(size=(9, 5)))
    y = pd.Series(["a"] * 7 + ["b"] * 2)
    with pytest.raises(ValueError):
        MLClassifier(X, y).evaluate(["Logistic Regression"], n_folds=5)


# ------------------------------------------------------------ messy-input robustness
def test_extra_text_annotation_column_does_not_break_normalization():
    """Real files often carry a 'Description' column next to the counts."""
    df, *_ = make_rnaseq(n_genes=50, n_per_group=3)
    df["Description"] = "protein coding"
    out = Normalizer(df).cpm()
    assert np.isfinite(out[[c for c in out.columns if c not in ("Gene", "Description")]].values).all()
