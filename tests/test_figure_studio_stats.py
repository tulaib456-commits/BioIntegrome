"""
Known-answer tests. The expected numbers come from published R output
(R's built-in 'sleep' and 'PlantGrowth' datasets) or from hand calculation,
NOT from this program, so they can catch real errors.
"""
import numpy as np
import pandas as pd
import pytest

from modules.figure_studio.statistics_toolkit import (
    compare_two_groups, compare_multiple_groups, pairwise_comparisons,
    test_correlation as correlation_test, test_categorical_association as chi_test,
    adjust_pvalues, test_normality as normality_test,
)

# R: sleep (extra sleep, 2 drugs, 10 patients)
SLEEP_1 = [0.7, -1.6, -0.2, -1.2, -0.1, 3.4, 3.7, 0.8, 0.0, 2.0]
SLEEP_2 = [1.9, 0.8, 1.1, 0.1, -0.1, 4.4, 5.5, 1.6, 4.6, 3.4]
# R: PlantGrowth
CTRL = [4.17, 5.58, 5.18, 6.11, 4.50, 4.61, 5.17, 4.53, 5.33, 5.14]
TRT1 = [4.81, 4.17, 4.41, 3.59, 5.87, 3.83, 6.03, 4.89, 4.32, 4.69]
TRT2 = [6.31, 5.12, 5.54, 5.50, 5.37, 5.29, 4.92, 6.15, 5.80, 5.26]


# ------------------------------------------------------------ known answers from R
def test_paired_t_test_matches_R_sleep():
    # R: t = -4.0621, df = 9, p-value = 0.002833, mean difference = -1.58
    r = compare_two_groups(SLEEP_1, SLEEP_2, paired=True, force_method="Paired t-test")
    assert r["statistic"] == pytest.approx(-4.0621, abs=1e-3)
    assert r["p_value"] == pytest.approx(0.002833, abs=1e-5)
    assert r["mean_a"] - r["mean_b"] == pytest.approx(-1.58, abs=1e-9)


def test_welch_t_test_matches_R_sleep():
    # R: t = -1.8608, df = 17.776, p-value = 0.07939
    r = compare_two_groups(SLEEP_1, SLEEP_2, force_method="Welch's t-test")
    assert r["statistic"] == pytest.approx(-1.8608, abs=1e-3)
    assert r["p_value"] == pytest.approx(0.07939, abs=1e-4)


def test_anova_matches_R_plantgrowth():
    # R: F = 4.846 on 2 and 27 df, p-value = 0.01591
    r = compare_multiple_groups({"ctrl": CTRL, "trt1": TRT1, "trt2": TRT2}, force_method="One-way ANOVA")
    assert r["statistic"] == pytest.approx(4.846, abs=1e-3)
    assert r["p_value"] == pytest.approx(0.01591, abs=1e-4)


def test_kruskal_matches_R_plantgrowth():
    # R: Kruskal-Wallis chi-squared = 7.9882, df = 2, p-value = 0.01842
    r = compare_multiple_groups({"ctrl": CTRL, "trt1": TRT1, "trt2": TRT2}, force_method="Kruskal-Wallis test")
    assert r["statistic"] == pytest.approx(7.9882, abs=1e-3)
    assert r["p_value"] == pytest.approx(0.01842, abs=1e-4)


def test_tukey_matches_R_plantgrowth():
    # R TukeyHSD: trt1-ctrl p=0.3909, trt2-ctrl p=0.1980, trt2-trt1 p=0.0120
    t = pairwise_comparisons({"ctrl": CTRL, "trt1": TRT1, "trt2": TRT2}, test="tukey")
    p = {(a, b): v for a, b, v in zip(t.Group_1, t.Group_2, t.P_Value)}
    assert p[("ctrl", "trt1")] == pytest.approx(0.3909, abs=2e-3)
    assert p[("ctrl", "trt2")] == pytest.approx(0.1980, abs=2e-3)
    assert p[("trt1", "trt2")] == pytest.approx(0.0120, abs=1e-3)


# ------------------------------------------------------------ multiple-comparison correction
P = [0.01, 0.04, 0.03, 0.005]


def test_holm_matches_hand_calculation():
    # sorted .005,.01,.03,.04 -> x4,x3,x2,x1 = .02,.03,.06,.04 -> running max .02,.03,.06,.06
    assert adjust_pvalues(P, "holm").round(6).tolist() == [0.03, 0.06, 0.06, 0.02]


def test_bonferroni_matches_hand_calculation():
    assert adjust_pvalues(P, "bonferroni").round(6).tolist() == [0.04, 0.16, 0.12, 0.02]


def test_benjamini_hochberg_matches_hand_calculation():
    assert adjust_pvalues(P, "bh").round(6).tolist() == [0.02, 0.04, 0.04, 0.02]


def test_adjusted_p_never_below_raw_and_never_above_one():
    rng = np.random.default_rng(0)
    p = rng.random(50)
    for m in ("holm", "bonferroni", "bh"):
        a = adjust_pvalues(p, m)
        assert (a >= p - 1e-12).all() and (a <= 1).all()


def test_nan_pvalues_stay_nan_and_are_not_counted():
    out = adjust_pvalues([0.01, np.nan, 0.04], "bonferroni")
    assert out[0] == pytest.approx(0.02) and np.isnan(out[1])


def test_uncorrected_pairs_overstate_significance_and_corrected_do_not():
    """Four groups drawn from the SAME population: uncorrected pairwise tests flag
    'significant' pairs far more than 5% of the time; Holm must not."""
    rng = np.random.default_rng(42)
    raw_hits = holm_hits = 0
    runs = 300
    for _ in range(runs):
        groups = {f"g{i}": rng.normal(0, 1, 8) for i in range(4)}
        t = pairwise_comparisons(groups, test="welch", correction="none")
        raw_hits += (t["P_Value"] < 0.05).any()
        t2 = pairwise_comparisons(groups, test="welch", correction="holm")
        holm_hits += (t2["Adj_P_Value"] < 0.05).any()
    assert raw_hits / runs > 0.15          # the old behaviour: way above the 5% promise
    assert holm_hits / runs < 0.09         # corrected: close to 5%


# ------------------------------------------------------------ robustness (messy / extreme input)
def test_two_groups_with_missing_values_ignores_them():
    r = compare_two_groups([1, 2, 3, np.nan, 5], [2, 3, 4, 5, 6, np.nan])
    assert r["n_a"] == 4 and r["n_b"] == 5


def test_group_with_one_value_is_refused_clearly():
    with pytest.raises(ValueError, match="at least 2"):
        compare_two_groups([1.0], [1, 2, 3])


def test_identical_groups_give_nonsignificant_not_crash():
    r = compare_two_groups([5, 5, 5, 5], [5, 5, 5, 5])
    assert r["significant"] is False


def test_paired_with_all_zero_differences_does_not_crash():
    r = compare_two_groups([1, 2, 3, 4], [1, 2, 3, 4], paired=True)
    assert r["significant"] is False


def test_paired_length_mismatch_is_refused():
    with pytest.raises(ValueError, match="same number"):
        compare_two_groups([1, 2, 3], [1, 2], paired=True)


def test_tiny_groups_use_nonparametric_and_cannot_be_significant_with_n2():
    r = compare_two_groups([1, 2], [10, 11])
    assert r["method"] == "Mann-Whitney U test"
    assert r["p_value"] > 0.05      # with n=2 vs 2 no result can reach p < .05


def test_outlier_heavy_data_selects_nonparametric():
    rng = np.random.default_rng(1)
    a = np.append(rng.normal(10, 1, 30), 500)
    b = rng.normal(11, 1, 31)
    assert compare_two_groups(a, b)["method"] == "Mann-Whitney U test"


def test_multiple_groups_ignores_groups_with_fewer_than_two_values_and_says_so():
    r = compare_multiple_groups({"a": [1, 2, 3, 4], "b": [2, 3, 4, 5], "c": [9]})
    assert "c" in r["dropped_groups"] and set(r["group_means"]) == {"a", "b"}


def test_multiple_groups_all_identical_values_does_not_crash():
    r = compare_multiple_groups({"a": [3, 3, 3], "b": [3, 3, 3], "c": [3, 3, 3]})
    assert r["significant"] is False


def test_correlation_pairs_rows_before_dropping_missing():
    """Old bug: dropping NaN separately from x and y misaligns the pairs."""
    x = np.array([1, 2, 3, 4, 5, 6, np.nan, 8, 9, 10.0])
    y = np.array([2, 4, 6, 8, 10, 12, 14, np.nan, 18, 20.0])
    r = correlation_test(x, y)
    assert r["r"] == pytest.approx(1.0) and r["n"] == 8


def test_correlation_constant_variable_refused():
    with pytest.raises(ValueError, match="identical"):
        correlation_test([1, 1, 1, 1], [1, 2, 3, 4])


def test_chi_square_matches_hand_calculation():
    # [[30,10],[10,30]]: expected 20 in every cell. With Yates' continuity correction
    # (applied to 2x2 tables): 4 * (|30-20| - 0.5)^2 / 20 = 18.05
    r = chi_test(np.array([[30, 10], [10, 30]]))
    assert r["chi2"] == pytest.approx(18.05, abs=1e-9)
    assert r["p_value"] < 0.001


def test_small_2x2_uses_fisher_exact():
    r = chi_test(np.array([[3, 1], [1, 3]]))
    assert "Fisher" in r["method"]
    assert r["p_value"] == pytest.approx(0.4857, abs=1e-3)   # R fisher.test(matrix(c(3,1,1,3),2))


def test_normality_of_constant_data_does_not_crash():
    assert normality_test([2, 2, 2, 2])["is_normal"] is None
