"""
Shared safety checks for the statistics engines.

Principle: bad input must produce a CLEAR message in plain words, never a
silent wrong answer (such as every p-value quietly becoming 1.0).
"""

import numpy as np
import pandas as pd


def check_samples_numeric(dataframe, samples):
    missing = [s for s in samples if s not in dataframe.columns]
    if missing:
        raise ValueError(
            f"These samples are not columns of the dataset: {missing[:5]}"
        )
    bad = [s for s in samples if not pd.api.types.is_numeric_dtype(dataframe[s])]
    if bad:
        raise ValueError(
            f"These sample columns are not numeric (text found inside): {bad[:5]}. "
            "Remove annotation columns or fix the text entries."
        )


def check_no_negative_values(dataframe, samples, already_normalized):
    """Raw-count style analysis (CPM + log2) is mathematically invalid on negative
    numbers (log-ratios, z-scores, centered data). Refuse instead of returning
    NaN -> p = 1.0 -> 'zero significant results'."""
    if already_normalized:
        return
    values = dataframe[samples].to_numpy(dtype=float)
    if np.nanmin(values) < 0:
        raise ValueError(
            "The data contain negative values (for example log-ratios), so they "
            "cannot be treated as raw counts. Tell the analysis the data are "
            "already normalized / log-transformed (already_normalized=True)."
        )


def check_no_empty_libraries(dataframe, samples, already_normalized):
    if already_normalized:
        return
    totals = dataframe[samples].sum()
    empty = [s for s in samples if not totals[s] > 0]
    if empty:
        raise ValueError(
            f"These samples have zero total counts and cannot be normalized: "
            f"{empty[:5]}. Remove them from the analysis."
        )


def check_group_sizes(group_a, group_b, minimum=2):
    if set(group_a) & set(group_b):
        raise ValueError("A sample cannot be in both groups.")
    for name, group in (("Group A", group_a), ("Group B", group_b)):
        if len(group) < minimum:
            raise ValueError(
                f"{name} has {len(group)} sample(s). Each group needs at least "
                f"{minimum} samples (replicates) for a statistical test."
            )
