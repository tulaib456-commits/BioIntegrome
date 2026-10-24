import numpy as np
import pandas as pd


class MissingValueAnalyzer:
    """
    Proteomics missing values are NOT like RNA-seq zero counts.
    Low-abundance proteins are more likely to go undetected (below
    the instrument's detection limit) -- this is Missing Not At
    Random (MNAR), the standard understanding in the field (same
    logic used by Perseus, the most widely used companion tool to
    MaxQuant). Missing values carry real biological information
    ("likely very low or absent"), so imputing with something like
    the sample mean would be actively wrong.
    """

    def __init__(self, dataframe, protein_column, samples):
        self.df = dataframe
        self.protein_column = protein_column
        self.samples = samples

    def missingness_summary(self):

        values = self.df[self.samples]
        missing_fraction = values.isna().mean(axis=1)

        return pd.DataFrame({
            self.protein_column: self.df[self.protein_column],
            "Missing_Fraction": missing_fraction,
            "Valid_Count": values.notna().sum(axis=1),
            "Total_Samples": len(self.samples)
        })

    def per_group_detection(self, group_map):
        """
        Fraction of samples with a valid value, per protein, PER
        GROUP -- what filtering/imputation decisions should
        actually be based on, not overall missingness.
        """

        groups = sorted(set(group_map.values()))
        result = pd.DataFrame({self.protein_column: self.df[self.protein_column]})

        for group in groups:
            group_samples = [s for s in self.samples if group_map.get(s) == group]
            group_values = self.df[group_samples]
            result[f"Valid_Fraction_{group}"] = group_values.notna().mean(axis=1)

        return result


class ProteomicsFilter:
    """
    Filters by detection rate WITHIN groups, not overall -- a
    protein detected in 100% of Group A but 0% of Group B is
    exactly the kind of biologically interesting, all-or-nothing
    signal a naive overall filter would wrongly discard.
    """

    def __init__(self, dataframe, protein_column, samples):
        self.df = dataframe
        self.protein_column = protein_column
        self.samples = samples

    def filter_by_group_detection(self, group_map, min_valid_fraction=0.7):

        groups = sorted(set(group_map.values()))
        keep_mask = pd.Series(False, index=self.df.index)

        for group in groups:
            group_samples = [s for s in self.samples if group_map.get(s) == group]
            valid_fraction = self.df[group_samples].notna().mean(axis=1)
            keep_mask |= (valid_fraction >= min_valid_fraction)

        return self.df[keep_mask].reset_index(drop=True)


class Imputer:
    """
    Downshifted-normal-distribution imputation -- the standard method for
    MNAR proteomics data (Perseus / FragPipe-Analyst "Perseus-style").

    For EACH SAMPLE (column) separately, missing values are drawn from a
    normal distribution with
        mean = (mean of that sample's observed values) - downshift * SD
        SD   = width * (SD of that sample's observed values)
    with the Perseus defaults downshift = 1.8 and width = 0.3.
    The logic: a missing protein was probably below the detection limit, so it
    is replaced by a LOW value taken from the low end of the whole sample's
    intensity distribution -- not from that protein's own average.

    If a sample has fewer than 3 observed values (or no spread), the
    distribution of ALL observed values is used for it instead.

    Operates on log-transformed intensities.
    """

    def __init__(self, downshift=1.8, width=0.3, random_state=42):
        self.downshift = downshift
        self.width = width
        self.random_state = random_state

    def impute(self, dataframe, samples):

        rng = np.random.default_rng(self.random_state)

        df = dataframe.copy()
        values = df[samples].values.astype(float)

        observed_all = values[~np.isnan(values)]

        if observed_all.size < 3:
            raise ValueError(
                "Fewer than 3 observed values in the whole table -- "
                "imputation is not possible."
            )

        global_mean = observed_all.mean()
        global_sd = observed_all.std(ddof=1)

        if not global_sd > 0:
            raise ValueError("All observed values are identical -- imputation is not possible.")

        for j in range(values.shape[1]):

            column = values[:, j]
            missing = np.isnan(column)

            n_missing = int(missing.sum())
            if n_missing == 0:
                continue

            observed = column[~missing]

            if observed.size >= 3 and observed.std(ddof=1) > 0:
                col_mean, col_sd = observed.mean(), observed.std(ddof=1)
            else:
                col_mean, col_sd = global_mean, global_sd

            column[missing] = rng.normal(
                col_mean - self.downshift * col_sd,
                self.width * col_sd,
                n_missing
            )

            values[:, j] = column

        df[samples] = values

        return df
