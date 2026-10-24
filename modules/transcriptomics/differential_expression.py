import numpy as np
import pandas as pd

from scipy import stats
from statsmodels.stats.multitest import multipletests

from modules.transcriptomics.validator import (
    detect_gene_column,
    detect_sample_columns
)
from modules.transcriptomics.input_checks import (
    check_samples_numeric, check_no_negative_values,
    check_no_empty_libraries, check_group_sizes
)


def suggest_groups(samples):
    """
    Try to auto-split sample names into two groups by finding a
    common underscore-separated token that takes exactly 2 distinct
    values across all samples (e.g. 'mock' vs 'CoV2').

    Returns (group_a, group_b) lists of sample names, or (None, None)
    if no clean 2-way split is found.
    """

    tokenized = [name.split("_") for name in samples]

    max_tokens = max(len(tokens) for tokens in tokenized)

    for position in range(max_tokens):

        values = []

        for tokens in tokenized:

            if position < len(tokens):
                values.append(tokens[position])
            else:
                values.append(None)

        unique_values = sorted(set(values))

        if len(unique_values) == 2 and None not in unique_values:

            group_a = [
                sample
                for sample, value in zip(samples, values)
                if value == unique_values[0]
            ]

            group_b = [
                sample
                for sample, value in zip(samples, values)
                if value == unique_values[1]
            ]

            return group_a, group_b

    return None, None


class DifferentialExpression:
    """
    Runs gene-by-gene differential expression between two groups
    of samples (Welch's t-test, Benjamini-Hochberg FDR).

    Raw counts (already_normalized=False): CPM + log2 is applied internally.
    Already normalized / log-ratio data (already_normalized=True): used as is.

    Safety behaviour (verified by the test-suite):
      * a group with fewer than 2 samples is refused with a clear message
      * negative values with already_normalized=False are refused (they would
        silently turn into NaN and p = 1.0)
      * missing values are handled gene by gene (scipy nan_policy="omit")
      * a gene that cannot be tested (fewer than 2 valid values in a group, or
        no variation at all) gets p = NaN and is EXCLUDED from the FDR
        correction, instead of being silently set to p = 1.0
    """

    def __init__(self, dataframe, group_a, group_b, already_normalized=False):

        self.gene_column = detect_gene_column(dataframe)

        self.group_a = list(group_a)
        self.group_b = list(group_b)

        check_group_sizes(self.group_a, self.group_b)
        check_samples_numeric(dataframe, self.group_a + self.group_b)
        check_no_negative_values(
            dataframe, self.group_a + self.group_b, already_normalized
        )
        check_no_empty_libraries(
            dataframe, self.group_a + self.group_b, already_normalized
        )

        self.raw = dataframe.copy()

        if already_normalized:
            self.df = dataframe.copy()
        else:
            self.df = self._cpm_log2(dataframe)

    def _cpm_log2(self, dataframe):

        df = dataframe.copy()

        all_samples = self.group_a + self.group_b

        totals = df[all_samples].sum()

        df[all_samples] = (df[all_samples] / totals) * 1_000_000

        df[all_samples] = np.log2(df[all_samples] + 1)

        return df

    def run(self, fdr_threshold=0.05, log2fc_threshold=1.0):

        import warnings

        genes = self.df[self.gene_column].values

        a_values = self.df[self.group_a].values.astype(float)
        b_values = self.df[self.group_b].values.astype(float)

        n_a = np.sum(~np.isnan(a_values), axis=1)
        n_b = np.sum(~np.isnan(b_values), axis=1)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)

            mean_a = np.nanmean(a_values, axis=1)
            mean_b = np.nanmean(b_values, axis=1)
            var_a = np.nanvar(a_values, axis=1, ddof=1)
            var_b = np.nanvar(b_values, axis=1, ddof=1)

            _, p_values = stats.ttest_ind(
                b_values, a_values, axis=1,
                equal_var=False, nan_policy="omit"
            )

        log2fc = mean_b - mean_a

        no_variation = (var_a == 0) & (var_b == 0)

        testable = (
            (n_a >= 2) & (n_b >= 2)
            & ~no_variation
            & np.isfinite(p_values)
        )

        p_values = np.where(testable, p_values, np.nan)

        adjusted = np.full(len(p_values), np.nan)
        if testable.any():
            adjusted[testable] = multipletests(
                p_values[testable], method="fdr_bh"
            )[1]

        results = pd.DataFrame({
            "Gene": genes,
            "Mean_GroupA": mean_a,
            "Mean_GroupB": mean_b,
            "Log2FC": log2fc,
            "P_Value": p_values,
            "Adj_P_Value": adjusted,
            "N_GroupA": n_a,
            "N_GroupB": n_b
        })

        results["Significant"] = (
            (results["Adj_P_Value"] < fdr_threshold)
            & (results["Log2FC"].abs() >= log2fc_threshold)
        ).fillna(False)

        self.n_untestable = int((~testable).sum())

        results = results.sort_values(
            "Adj_P_Value", na_position="last"
        ).reset_index(drop=True)

        return results
