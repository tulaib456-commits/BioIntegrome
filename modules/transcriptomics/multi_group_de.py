import numpy as np
import pandas as pd
from scipy import stats

from modules.transcriptomics.input_checks import (
    check_samples_numeric, check_no_negative_values, check_no_empty_libraries
)


def suggest_multi_groups(samples, max_groups=8):
    """
    Auto-detects a grouping variable with 2-8 distinct values from
    underscore-separated sample name tokens (e.g. tissue type,
    treatment condition). Returns {sample: group_label} or None.
    """

    tokenized = [name.split("_") for name in samples]
    max_tokens = max(len(tokens) for tokens in tokenized)

    for position in range(max_tokens):

        values = []

        for tokens in tokenized:
            values.append(tokens[position] if position < len(tokens) else None)

        unique_values = sorted(set(values))

        if None not in unique_values and 2 <= len(unique_values) <= max_groups:
            return {sample: value for sample, value in zip(samples, values)}

    return None


class MultiGroupDE:
    """
    One-way ANOVA across 2+ groups, per gene, fully vectorized
    across all genes at once (no per-gene Python loop — tested at
    ~28,000 genes in 0.02s vs 23s for the naive loop version).
    Runs on CPM + log2 normalized values, computed internally so
    results don't depend on whichever normalization method was
    picked in the Normalization tab.
    """

    def __init__(self, dataframe, group_map, gene_column, already_normalized=False):
        self.gene_column = gene_column
        self.group_map = group_map
        self.samples = list(group_map.keys())
        self.groups = sorted(set(group_map.values()))

        if len(self.groups) < 2:
            raise ValueError("ANOVA needs at least 2 groups.")
        if len(self.samples) <= len(self.groups):
            raise ValueError(
                "ANOVA needs more samples than groups (at least one group "
                "must have replicates)."
            )
        check_samples_numeric(dataframe, self.samples)
        check_no_negative_values(dataframe, self.samples, already_normalized)
        check_no_empty_libraries(dataframe, self.samples, already_normalized)

        if already_normalized:
            self.df = dataframe.copy()
        else:
            self.df = self._cpm_log2(dataframe)

    def _cpm_log2(self, dataframe):
        df = dataframe.copy()
        totals = df[self.samples].sum()
        df[self.samples] = (df[self.samples] / totals) * 1_000_000
        df[self.samples] = np.log2(df[self.samples] + 1)
        return df

    def run(self, fdr_threshold=0.05, min_effect_size=0.0):
        """
        One-way ANOVA for every gene at once (vectorized). Missing values are
        handled gene by gene: each gene uses the valid values it has. A gene
        that cannot be tested (fewer than 2 groups with data, no degrees of
        freedom, or no variation inside any group) gets p = NaN and is left
        out of the FDR correction.
        """

        from statsmodels.stats.multitest import multipletests

        genes = self.df[self.gene_column].values
        values = self.df[self.samples].values.astype(float)

        sample_index = {s: i for i, s in enumerate(self.samples)}
        valid = ~np.isnan(values)
        filled = np.where(valid, values, 0.0)

        n_genes = values.shape[0]

        n_total = valid.sum(axis=1).astype(float)
        grand_mean = np.where(n_total > 0, filled.sum(axis=1) / np.maximum(n_total, 1), np.nan)

        ss_between = np.zeros(n_genes)
        ss_within = np.zeros(n_genes)
        k_present = np.zeros(n_genes)
        group_means = {}

        for group in self.groups:

            idx = [sample_index[s] for s in self.samples if self.group_map[s] == group]
            g_valid = valid[:, idx]
            g_filled = filled[:, idx]
            n_g = g_valid.sum(axis=1).astype(float)

            with np.errstate(invalid="ignore", divide="ignore"):
                mean_g = np.where(n_g > 0, g_filled.sum(axis=1) / np.maximum(n_g, 1), np.nan)

            group_means[group] = mean_g
            present = n_g > 0
            k_present += present

            ss_between += np.where(present, n_g * (mean_g - grand_mean) ** 2, 0.0)
            dev = np.where(g_valid, g_filled - np.nan_to_num(mean_g)[:, None], 0.0)
            ss_within += (dev ** 2).sum(axis=1)

        df_between = k_present - 1
        df_within = n_total - k_present

        testable = (df_between >= 1) & (df_within >= 1)

        with np.errstate(divide="ignore", invalid="ignore"):
            ms_between = ss_between / np.where(df_between >= 1, df_between, np.nan)
            ms_within = ss_within / np.where(df_within >= 1, df_within, np.nan)
            f_stats = ms_between / ms_within

        testable &= np.isfinite(f_stats) & (ms_within > 0)

        p_values = np.full(n_genes, np.nan)
        p_values[testable] = stats.f.sf(
            f_stats[testable], df_between[testable], df_within[testable]
        )
        f_out = np.where(testable, f_stats, np.nan)

        adjusted_p_values = np.full(n_genes, np.nan)
        if testable.any():
            adjusted_p_values[testable] = multipletests(
                p_values[testable], method="fdr_bh"
            )[1]

        means_matrix = np.column_stack([group_means[g] for g in self.groups])
        with np.errstate(invalid="ignore"):
            max_mean_diff = np.nanmax(means_matrix, axis=1) - np.nanmin(means_matrix, axis=1)

        results = pd.DataFrame({
            "Gene": genes,
            "F_Statistic": f_out,
            "P_Value": p_values,
            "Adj_P_Value": adjusted_p_values,
            "Max_Mean_Diff": max_mean_diff
        })

        for group in self.groups:
            results[f"Mean_{group}"] = group_means[group]

        results["Significant"] = (
            (results["Adj_P_Value"] < fdr_threshold)
            & (results["Max_Mean_Diff"] >= min_effect_size)
        ).fillna(False)

        self.n_untestable = int((~testable).sum())

        return results.sort_values(
            "Adj_P_Value", na_position="last"
        ).reset_index(drop=True)

    def posthoc(self, gene_name):
        """
        Tukey HSD pairwise comparison for ONE gene, computed on
        demand so it stays instant regardless of dataset size.
        """

        from statsmodels.stats.multicomp import pairwise_tukeyhsd

        gene_row = self.df[self.df[self.gene_column] == gene_name]

        if gene_row.empty:
            return None

        row = gene_row.iloc[0]

        values = np.array([float(row[s]) for s in self.samples])
        labels = np.array([self.group_map[s] for s in self.samples])

        keep = ~np.isnan(values)
        values, labels = values[keep], labels[keep]

        if len(set(labels)) < 2:
            return None

        tukey_result = pairwise_tukeyhsd(endog=values, groups=labels)
        summary = tukey_result.summary()

        return pd.DataFrame(
            data=summary.data[1:],
            columns=summary.data[0]
        )

    def gene_expression_table(self, gene_name):
        """Long-format (Sample, Group, Expression) table for one gene — feeds a boxplot."""

        gene_row = self.df[self.df[self.gene_column] == gene_name]

        if gene_row.empty:
            return None

        row = gene_row.iloc[0]

        return pd.DataFrame({
            "Sample": self.samples,
            "Group": [self.group_map[s] for s in self.samples],
            "Expression": [float(row[s]) for s in self.samples]
        })