"""
Turning the many layouts researchers really use into the one shape the
plots and tests need ("long": one column names the group, one holds the value).
"""

import numpy as np
import pandas as pd

from modules.figure_studio.data_loader import detect_column_types

_ID_WORDS = ("id", "subject", "patient", "sample", "animal", "case", "participant", "mouse")


def looks_like_id(column_name, series):
    name = str(column_name).lower()
    if any(w == name or name.startswith(w) or name.endswith(w) for w in _ID_WORDS):
        return True
    if pd.api.types.is_integer_dtype(series) and series.is_unique and series.min() in (0, 1):
        return True
    return False


def suggest_layout(dataframe):
    """'wide' when there is no column that could name a group but there are
    two or more numeric columns (e.g. ID | Before | After); otherwise 'long'."""

    numeric_cols, categorical_cols = detect_column_types(dataframe)
    if not categorical_cols and len(numeric_cols) >= 2 and not code_like_columns(dataframe):
        return "wide"
    return "long"


def wide_to_long(dataframe, value_cols, id_col=None,
                 group_label="Group", value_label="Value"):
    """
    Wide (Before | After) -> long (ID | Group | Value).
    Names are made unique so they never collide with existing columns.
    """

    if len(value_cols) < 2:
        raise ValueError("Pick at least 2 columns to compare.")
    if id_col is not None and id_col in value_cols:
        raise ValueError("The ID column cannot also be a value column.")

    taken = set(dataframe.columns)
    while group_label in taken:
        group_label += "_"
    while value_label in taken or value_label == group_label:
        value_label += "_"

    id_vars = [id_col] if id_col else []
    long = dataframe.melt(
        id_vars=id_vars, value_vars=list(value_cols),
        var_name=group_label, value_name=value_label
    )

    if id_col is None:
        # keep the pairing information: row number within the original table
        long.insert(0, "Row", np.tile(np.arange(1, len(dataframe) + 1), len(value_cols)))

    # keep the user's column order for the groups
    long[group_label] = pd.Categorical(long[group_label], categories=list(value_cols), ordered=True)
    long[group_label] = long[group_label].astype(str)

    long = long.dropna(subset=[value_label]).reset_index(drop=True)
    return long, group_label, value_label


def code_like_columns(dataframe, max_levels=6):
    """Numeric columns that are probably group CODES (e.g. 0/1/2), offered to the
    user as 'treat as categories?'"""

    out = []
    for col in dataframe.columns:
        s = dataframe[col]
        if not pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
            continue
        clean = s.dropna()
        if clean.empty:
            continue
        integer_valued = np.all(np.isclose(clean, np.round(clean)))
        n_unique = clean.nunique()
        if integer_valued and 2 <= n_unique <= max_levels and n_unique < len(clean) / 2:
            out.append(col)
    return out


def as_categories(dataframe, columns):
    """Turn numeric code columns (0/1/2 ...) into text labels so they act as groups."""

    df = dataframe.copy()
    for c in columns:
        values = df[c]
        if values.dropna().empty:
            continue
        if np.all(np.isclose(values.dropna(), np.round(values.dropna()))):
            text = values.astype("Int64").astype(str)
        else:
            text = values.astype(str)
        df[c] = text.where(values.notna(), np.nan)
    return df


def paired_arrays(dataframe, id_col, group_col, value_col, group_1, group_2):
    """
    Aligns two conditions BY SUBJECT ID and keeps only subjects measured in both.
    (Pairing by row position breaks as soon as one value is missing.)
    Returns (values_1, values_2, n_dropped).
    """

    if len({id_col, group_col, value_col}) < 3:
        raise ValueError(
            "The Subject ID, the Group and the Value must be three DIFFERENT columns "
            "(for example ID | Group | Value)."
        )

    sub = dataframe[[id_col, group_col, value_col]].dropna(subset=[id_col])
    sub = sub[sub[group_col].isin([group_1, group_2])]

    if sub.duplicated(subset=[id_col, group_col]).any():
        raise ValueError(
            f"Some subjects have more than one value for the same condition in "
            f"'{id_col}'. A paired test needs exactly one value per subject and condition."
        )

    wide = sub.pivot(index=id_col, columns=group_col, values=value_col)

    if group_1 not in wide.columns or group_2 not in wide.columns:
        raise ValueError("One of the two conditions has no data.")

    n_subjects = len(wide)
    complete = wide.dropna(subset=[group_1, group_2])

    return (
        complete[group_1].to_numpy(dtype=float),
        complete[group_2].to_numpy(dtype=float),
        n_subjects - len(complete),
    )
