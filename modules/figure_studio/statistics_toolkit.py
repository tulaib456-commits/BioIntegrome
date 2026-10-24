import warnings

import numpy as np
import pandas as pd
from scipy import stats

ALPHA = 0.05


# ---------------------------------------------------------------- helpers

def _clean(values, name="group"):
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    return arr


def _require_n(arr, name, minimum=2):
    if len(arr) < minimum:
        raise ValueError(
            f"{name} has {len(arr)} valid value(s); at least {minimum} are "
            "needed for a statistical test."
        )


def significance_stars(p):
    if p is None or not np.isfinite(p):
        return "n/a"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def adjust_pvalues(p_values, method="holm"):
    """
    Multiple-comparison correction. NaN p-values stay NaN and are NOT counted.
    method: 'holm' (default, uniformly better than Bonferroni), 'bonferroni',
            'bh' (Benjamini-Hochberg FDR), 'none'.
    """

    p = np.asarray(p_values, dtype=float)
    out = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    m = int(ok.sum())

    if m == 0 or method == "none":
        out[ok] = p[ok]
        return out

    pv = p[ok]
    order = np.argsort(pv)
    sorted_p = pv[order]

    if method == "bonferroni":
        adj_sorted = np.minimum(sorted_p * m, 1.0)
    elif method == "holm":
        adj_sorted = np.minimum(np.maximum.accumulate(sorted_p * (m - np.arange(m))), 1.0)
    elif method == "bh":
        raw = sorted_p * m / (np.arange(m) + 1)
        adj_sorted = np.minimum(np.minimum.accumulate(raw[::-1])[::-1], 1.0)
    else:
        raise ValueError(f"Unknown correction method: {method}")

    adj = np.empty(m)
    adj[order] = adj_sorted
    out[ok] = adj
    return out


# ---------------------------------------------------------------- normality

def test_normality(data):
    """Shapiro-Wilk test -- the standard way to check if parametric
    tests (t-test, ANOVA, Pearson) are appropriate, or whether you
    should use their non-parametric equivalents instead."""

    data = np.asarray(data, dtype=float)
    data = data[~np.isnan(data)]

    if len(data) < 3:
        return {"statistic": None, "p_value": None, "is_normal": None,
                "note": "Need at least 3 values to test normality."}

    if np.ptp(data) == 0:
        return {"statistic": None, "p_value": None, "is_normal": None,
                "note": "All values are identical -- normality cannot be tested."}

    stat, p = stats.shapiro(data)

    return {
        "statistic": stat, "p_value": p, "is_normal": bool(p > 0.05),
        "note": "Normally distributed (p > 0.05)" if p > 0.05
                else "NOT normally distributed (p ≤ 0.05) — consider a non-parametric test"
    }


# ---------------------------------------------------------------- two groups

def compare_two_groups(group_a, group_b, paired=False, force_method=None):
    """
    Auto-selects the correct test based on normality (like SPSS's
    guidance), or lets the user force a specific method. Returns a
    full result dict with everything needed for an SPSS-style
    report: test name, statistic, p-value, effect size.

    paired=True needs the two arrays aligned subject by subject (same length).
    For paired data the normality check is done on the DIFFERENCES, which is
    what the paired t-test actually assumes.
    """

    group_a = np.asarray(group_a, dtype=float)
    group_b = np.asarray(group_b, dtype=float)

    if paired:
        if len(group_a) != len(group_b):
            raise ValueError(
                "A paired test needs the same number of values in both groups "
                f"(got {len(group_a)} and {len(group_b)})."
            )
        keep = ~(np.isnan(group_a) | np.isnan(group_b))
        group_a, group_b = group_a[keep], group_b[keep]
    else:
        group_a, group_b = _clean(group_a), _clean(group_b)

    _require_n(group_a, "Group 1")
    _require_n(group_b, "Group 2")

    norm_a = test_normality(group_a)
    norm_b = test_normality(group_b)

    if paired:
        norm_diff = test_normality(group_a - group_b)
        both_normal = bool(norm_diff["is_normal"])
    else:
        norm_diff = None
        both_normal = bool(norm_a["is_normal"]) and bool(norm_b["is_normal"])

    if force_method:
        method = force_method
    elif paired:
        method = "Paired t-test" if both_normal else "Wilcoxon signed-rank test"
    else:
        method = "Welch's t-test" if both_normal else "Mann-Whitney U test"

    note = None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        try:
            if method == "Paired t-test":
                stat, p = stats.ttest_rel(group_a, group_b)
            elif method == "Wilcoxon signed-rank test":
                stat, p = stats.wilcoxon(group_a, group_b)
            elif method == "Welch's t-test":
                stat, p = stats.ttest_ind(group_a, group_b, equal_var=False)
            elif method == "Student's t-test (equal variance)":
                stat, p = stats.ttest_ind(group_a, group_b, equal_var=True)
            else:  # Mann-Whitney U
                stat, p = stats.mannwhitneyu(group_a, group_b, alternative="two-sided")
        except ValueError as error:
            stat, p = np.nan, 1.0
            note = f"The test could not distinguish the groups ({error})."

    if not np.isfinite(p):
        stat, p = np.nan, np.nan
        note = "The groups show no variation, so the test is not defined."

    # Effect size: Cohen's d (independent groups) or d_z (paired)
    if paired:
        diff = group_a - group_b
        sd = diff.std(ddof=1)
        effect = diff.mean() / sd if sd > 0 else np.nan
        effect_name = "Cohen's d (paired, d_z)"
    else:
        pooled = np.sqrt((group_a.var(ddof=1) + group_b.var(ddof=1)) / 2)
        effect = (group_a.mean() - group_b.mean()) / pooled if pooled > 0 else np.nan
        effect_name = "Cohen's d"

    return {
        "method": method,
        "statistic": stat,
        "p_value": p,
        "significant": bool(p < ALPHA) if np.isfinite(p) else False,
        "cohens_d": effect,
        "effect_size_name": effect_name,
        "n_a": len(group_a), "n_b": len(group_b),
        "mean_a": group_a.mean(), "mean_b": group_b.mean(),
        "normality_a": norm_a, "normality_b": norm_b,
        "normality_diff": norm_diff,
        "note": note,
    }


# ---------------------------------------------------------------- 3+ groups

def pairwise_comparisons(groups_dict, test="auto", correction="holm"):
    """
    All pairwise comparisons between groups WITH multiple-comparison control.

    test:
      'auto'        Shapiro on every group -> Welch's t-test if all look normal,
                    otherwise Mann-Whitney U
      'welch'       Welch's t-test
      'mannwhitney' Mann-Whitney U
      'tukey'       Tukey HSD (its own correction; needs >= 2 values per group)
    correction: 'holm' (default), 'bonferroni', 'bh', 'none'  (ignored for Tukey)

    Returns a table: one row per pair, with raw and ADJUSTED p-values.
    """

    cleaned = {}
    for name, values in groups_dict.items():
        arr = _clean(values)
        if len(arr) >= 2:
            cleaned[name] = arr

    if len(cleaned) < 2:
        raise ValueError("At least 2 groups with 2 or more valid values each are needed.")

    names = list(cleaned)

    if test == "auto":
        normal = all(bool(test_normality(cleaned[n])["is_normal"]) for n in names)
        test_used = "welch" if normal else "mannwhitney"
    else:
        test_used = test

    rows = []

    if test_used == "tukey":
        res = stats.tukey_hsd(*[cleaned[n] for n in names])
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                rows.append({
                    "Group_1": names[i], "Group_2": names[j],
                    "n1": len(cleaned[names[i]]), "n2": len(cleaned[names[j]]),
                    "Mean_1": cleaned[names[i]].mean(), "Mean_2": cleaned[names[j]].mean(),
                    "Mean_Diff": cleaned[names[i]].mean() - cleaned[names[j]].mean(),
                    "Test": "Tukey HSD", "Statistic": float(res.statistic[i, j]),
                    "P_Value": float(res.pvalue[i, j]),
                })
        table = pd.DataFrame(rows)
        table["Adj_P_Value"] = table["P_Value"]
        table["Correction"] = "Tukey (built in)"
    else:
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = cleaned[names[i]], cleaned[names[j]]
                res = compare_two_groups(
                    a, b,
                    force_method="Welch's t-test" if test_used == "welch" else "Mann-Whitney U test"
                )
                rows.append({
                    "Group_1": names[i], "Group_2": names[j],
                    "n1": len(a), "n2": len(b),
                    "Mean_1": a.mean(), "Mean_2": b.mean(),
                    "Mean_Diff": a.mean() - b.mean(),
                    "Test": res["method"], "Statistic": res["statistic"],
                    "P_Value": res["p_value"],
                })
        table = pd.DataFrame(rows)
        table["Adj_P_Value"] = adjust_pvalues(table["P_Value"].values, correction)
        table["Correction"] = {"holm": "Holm", "bonferroni": "Bonferroni",
                               "bh": "Benjamini-Hochberg", "none": "none (uncorrected)"}[correction]

    table["Stars"] = [significance_stars(p) for p in table["Adj_P_Value"]]
    table["Significant"] = table["Adj_P_Value"] < ALPHA
    return table


def compare_multiple_groups(groups_dict, force_method=None):
    """groups_dict: {group_name: array_of_values}. ANOVA/Kruskal-Wallis
    equivalent of the above, for 3+ groups. Also returns the post-hoc table
    (Tukey HSD after ANOVA; Mann-Whitney with Holm correction after Kruskal-Wallis)."""

    cleaned = {}
    dropped = []
    for name, values in groups_dict.items():
        arr = _clean(values)
        if len(arr) >= 2:
            cleaned[name] = arr
        else:
            dropped.append(str(name))

    if len(cleaned) < 2:
        raise ValueError("At least 2 groups with 2 or more valid values each are needed.")

    group_arrays = list(cleaned.values())
    group_names = list(cleaned.keys())

    normalities = [test_normality(g) for g in group_arrays]
    all_normal = all(bool(n["is_normal"]) for n in normalities)

    method = force_method or ("One-way ANOVA" if all_normal else "Kruskal-Wallis test")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if method == "One-way ANOVA":
            stat, p = stats.f_oneway(*group_arrays)
        else:
            try:
                stat, p = stats.kruskal(*group_arrays)
            except ValueError:
                stat, p = np.nan, 1.0

    if not np.isfinite(p):
        stat, p = np.nan, np.nan

    posthoc = None
    if len(cleaned) >= 3 and np.isfinite(p):
        posthoc = pairwise_comparisons(
            cleaned, test="tukey" if method == "One-way ANOVA" else "mannwhitney",
            correction="holm"
        )

    return {
        "method": method, "statistic": stat, "p_value": p,
        "significant": bool(p < ALPHA) if np.isfinite(p) else False,
        "group_means": {name: float(np.mean(arr)) for name, arr in cleaned.items()},
        "group_n": {name: len(arr) for name, arr in cleaned.items()},
        "normalities": dict(zip(group_names, normalities)),
        "posthoc": posthoc,
        "dropped_groups": dropped,
    }


# ---------------------------------------------------------------- correlation / chi-square

def test_correlation(x, y, force_method=None):
    """x and y must be the SAME rows; rows with a missing value in either are dropped."""

    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)

    if len(x) != len(y):
        raise ValueError("x and y must have the same number of values.")

    keep = ~(np.isnan(x) | np.isnan(y))
    x, y = x[keep], y[keep]

    if len(x) < 3:
        raise ValueError(f"Only {len(x)} complete pair(s); at least 3 are needed for a correlation.")
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        raise ValueError("One of the variables has identical values in every row, so a correlation is not defined.")

    norm_x, norm_y = test_normality(x), test_normality(y)
    both_normal = bool(norm_x["is_normal"]) and bool(norm_y["is_normal"])

    method = force_method or ("Pearson" if both_normal else "Spearman")

    if method == "Pearson":
        r, p = stats.pearsonr(x, y)
    else:
        r, p = stats.spearmanr(x, y)

    return {"method": method, "r": r, "p_value": p, "n": len(x),
            "significant": bool(p < ALPHA)}


def test_categorical_association(contingency_table):
    """
    Chi-square test of independence -- for comparing categorical variables
    (e.g. does treatment group relate to outcome category).
    Small tables: when any expected count is below 5 the chi-square
    approximation is unreliable, so a 2x2 table uses Fisher's exact test instead.
    """

    table = np.asarray(contingency_table, dtype=float)

    if table.ndim != 2 or table.shape[0] < 2 or table.shape[1] < 2:
        raise ValueError("Both variables need at least 2 different categories.")

    chi2, p, dof, expected = stats.chi2_contingency(table)

    n = table.sum()
    cramers_v = np.sqrt(chi2 / (n * (min(table.shape) - 1))) if n > 0 else np.nan

    method = "Chi-square test of independence"
    low_expected = bool((expected < 5).any())
    note = None

    if low_expected and table.shape == (2, 2):
        _, p = stats.fisher_exact(table)
        method = "Fisher's exact test (expected counts < 5)"
    elif low_expected:
        note = ("Some expected counts are below 5, so the chi-square p-value is "
                "approximate. Combine rare categories or collect more data.")

    return {
        "method": method,
        "chi2": chi2, "p_value": p, "dof": dof,
        "cramers_v": cramers_v,
        "significant": bool(p < ALPHA),
        "note": note,
    }
