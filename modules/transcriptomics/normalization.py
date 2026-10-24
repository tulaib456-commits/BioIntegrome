import numpy as np

from modules.transcriptomics.validator import (

    detect_gene_column,

    detect_sample_columns

)


class Normalizer:

    def __init__(self, dataframe):

        self.df = dataframe.copy()

        self.gene = detect_gene_column(dataframe)

        self.samples = detect_sample_columns(dataframe)

    def cpm(self):

        df = self.df.copy()

        totals = df[self.samples].sum()

        df[self.samples] = (

            df[self.samples]

            / totals

        ) * 1000000

        return df

    def log2_cpm(self):

        df = self.cpm()

        df[self.samples] = np.log2(

            df[self.samples] + 1

        )

        return df

    def median_of_ratios(self):
        """
        DESeq2-style median-of-ratios normalization. More robust than simple
        CPM because it isn't thrown off by a handful of very highly expressed genes.

        Standard version: reference = geometric mean of genes that have a
        non-zero count in EVERY sample. Small or sparse datasets often have no
        such gene; DESeq2 then switches to its "poscounts" variant, and so does
        this function (it never returns a silent wall of NaN).
        """

        df = self.df.copy()
        counts = df[self.samples].values.astype(float)

        if (counts < 0).any():
            raise ValueError("Counts must not be negative for median-of-ratios normalization.")

        totals = counts.sum(axis=0)
        empty = [s for s, t in zip(self.samples, totals) if not t > 0]
        if empty:
            raise ValueError(
                f"These samples have zero total counts and cannot be normalized: {empty[:5]}"
            )

        positive = counts > 0
        all_positive_genes = positive.all(axis=1)

        with np.errstate(divide="ignore", invalid="ignore"):
            log_counts = np.where(positive, np.log(counts), np.nan)

        if all_positive_genes.sum() >= 1:
            self.size_factor_method = "ratio"
            gene_log_means = log_counts[all_positive_genes].mean(axis=1)
            ratios = log_counts[all_positive_genes] - gene_log_means[:, None]
            log_size_factors = np.median(ratios, axis=0)
        else:
            # poscounts: sum of log over positive counts / number of samples
            self.size_factor_method = "poscounts"
            gene_log_means = np.nansum(log_counts, axis=1) / counts.shape[1]
            usable = positive.any(axis=1)
            ratios = log_counts[usable] - gene_log_means[usable][:, None]
            log_size_factors = np.nanmedian(ratios, axis=0)

        size_factors = np.exp(log_size_factors)

        if not np.isfinite(size_factors).all() or (size_factors <= 0).any():
            raise ValueError(
                "Could not estimate library size factors (too many zeros). "
                "Filter low-count genes first, or use CPM normalization."
            )

        df[self.samples] = counts / size_factors[None, :]

        self.size_factors = dict(zip(self.samples, size_factors))

        return df
