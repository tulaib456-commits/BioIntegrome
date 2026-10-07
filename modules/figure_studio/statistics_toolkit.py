import numpy as np
from scipy import stats


def test_normality(data):
    """Shapiro-Wilk test -- the standard way to check if parametric
    tests (t-test, ANOVA, Pearson) are appropriate, or whether you
    should use their non-parametric equivalents instead."""

    data = np.asarray(data)
    data = data[~np.isnan(data)]

    if len(data) < 3:
        return {"statistic": None, "p_value": None, "is_normal": None,
                "note": "Need at least 3 values to test normality."}

    stat, p = stats.shapiro(data)

    return {
        "statistic": stat, "p_value": p, "is_normal": p > 0.05,
        "note": "Normally distributed (p > 0.05)" if p > 0.05
                else "NOT normally distributed (p ≤ 0.05) — consider a non-parametric test"
    }


def compare_two_groups(group_a, group_b, paired=False, force_method=None):
    """
    Auto-selects the correct test based on normality (like SPSS's
    guidance), or lets the user force a specific method. Returns a
    full result dict with everything needed for an SPSS-style
    report: test name, statistic, p-value, effect size.
    """

    group_a = np.asarray(group_a)
    group_b = np.asarray(group_b)

    norm_a = test_normality(group_a)
    norm_b = test_normality(group_b)
    both_normal = (norm_a["is_normal"] and norm_b["is_normal"])

    if force_method:
        method = force_method
    elif paired:
        method = "Paired t-test" if both_normal else "Wilcoxon signed-rank test"
    else:
        method = "Welch's t-test" if both_normal else "Mann-Whitney U test"

    if method == "Paired t-test":
        stat, p = stats.ttest_rel(group_a, group_b)
    elif method == "Wilcoxon signed-rank test":
        stat, p = stats.wilcoxon(group_a, group_b)
    elif method == "Welch's t-test":
        stat, p = stats.ttest_ind(group_a, group_b, equal_var=False)
    elif method == "Student's t-test (equal variance)":
        stat, p = stats.ttest_ind(group_a, group_b, equal_var=True)
    else:  # Mann-Whitney U
        stat, p = stats.mannwhitneyu(group_a, group_b)

    # Cohen's d effect size (standard, interpretable magnitude measure)
    pooled_std = np.sqrt((group_a.var(ddof=1) + group_b.var(ddof=1)) / 2)
    cohens_d = (group_a.mean() - group_b.mean()) / pooled_std if pooled_std > 0 else 0

    return {
        "method": method,
        "statistic": stat,
        "p_value": p,
        "significant": p < 0.05,
        "cohens_d": cohens_d,
        "mean_a": group_a.mean(), "mean_b": group_b.mean(),
        "normality_a": norm_a, "normality_b": norm_b
    }


def compare_multiple_groups(groups_dict, force_method=None):
    """groups_dict: {group_name: array_of_values}. ANOVA/Kruskal-Wallis
    equivalent of the above, for 3+ groups."""

    group_arrays = list(groups_dict.values())
    group_names = list(groups_dict.keys())

    normalities = [test_normality(g) for g in group_arrays]
    all_normal = all(n["is_normal"] for n in normalities)

    method = force_method or ("One-way ANOVA" if all_normal else "Kruskal-Wallis test")

    if method == "One-way ANOVA":
        stat, p = stats.f_oneway(*group_arrays)
    else:
        stat, p = stats.kruskal(*group_arrays)

    return {
        "method": method, "statistic": stat, "p_value": p,
        "significant": p < 0.05,
        "group_means": {name: np.mean(arr) for name, arr in groups_dict.items()},
        "normalities": dict(zip(group_names, normalities))
    }


def test_correlation(x, y, force_method=None):

    x, y = np.asarray(x), np.asarray(y)

    norm_x, norm_y = test_normality(x), test_normality(y)
    both_normal = norm_x["is_normal"] and norm_y["is_normal"]

    method = force_method or ("Pearson" if both_normal else "Spearman")

    if method == "Pearson":
        r, p = stats.pearsonr(x, y)
    else:
        r, p = stats.spearmanr(x, y)

    return {"method": method, "r": r, "p_value": p, "significant": p < 0.05}


def test_categorical_association(contingency_table):
    """Chi-square test of independence -- for comparing categorical
    variables (e.g. does treatment group relate to outcome category)."""

    chi2, p, dof, expected = stats.chi2_contingency(contingency_table)

    return {
        "method": "Chi-square test of independence",
        "chi2": chi2, "p_value": p, "dof": dof,
        "significant": p < 0.05
    }