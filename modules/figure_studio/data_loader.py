import io
import re

import numpy as np
import pandas as pd


def detect_column_types(dataframe):
    """
    Splits columns into numeric (for values/axes) and categorical
    (for grouping/coloring) -- drives which columns the UI offers
    in each dropdown.
    """

    numeric_cols = []
    categorical_cols = []

    for col in dataframe.columns:
        if pd.api.types.is_bool_dtype(dataframe[col]):
            categorical_cols.append(col)
        elif pd.api.types.is_numeric_dtype(dataframe[col]):
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)

    return numeric_cols, categorical_cols


# ---------------------------------------------------------------- cleaning

_NUMBER_WITH_COMMA_DECIMAL = re.compile(r"^-?\d+,\d+$")
_NUMBER_WITH_THOUSANDS = re.compile(r"^-?\d{1,3}(,\d{3})+(\.\d+)?$")


def _is_text_column(series):
    """True for text columns in every pandas version.
    (pandas 3 stores text as a new 'str' type, so `dtype == object` is no longer enough.)"""
    return (
        pd.api.types.is_object_dtype(series)
        or pd.api.types.is_string_dtype(series)
    ) and not pd.api.types.is_bool_dtype(series)


def _try_numeric(series):
    """
    Returns (numeric_series, how) if a text column is really numbers stored
    as text (e.g. '12,5' or '1,234' or ' 7.1 '), otherwise (None, None).
    Needs >= 90 % of the non-empty cells to convert, so real categories
    such as 'Control' are never touched, and ID codes like '007' are kept.
    """

    text = series.dropna().astype(str).str.strip()
    text = text[text != ""]

    if text.empty:
        return None, None

    if text.str.match(r"^0\d+$").any():          # '007' style codes: keep as text
        return None, None

    cleaned = series.astype(str).str.strip().where(series.notna())

    with_comma = text[text.str.contains(",", regex=False)]
    all_thousands_form = (
        len(with_comma) > 0 and with_comma.str.match(_NUMBER_WITH_THOUSANDS).all()
    )

    if all_thousands_form:
        # '1,234' is ambiguous (1234 or 1.234). When EVERY value with a comma has
        # exactly 3 digits after it, thousands separators are the likelier reading;
        # the user is told what was assumed.
        candidate = cleaned.str.replace(",", "", regex=False)
        how = "comma read as thousands separator, e.g. 1,234 = 1234 -- check this is right"
    elif text.str.match(_NUMBER_WITH_COMMA_DECIMAL).mean() > 0.5:
        candidate = cleaned.str.replace(",", ".", regex=False)
        how = "decimal comma, e.g. 12,5 = 12.5"
    else:
        candidate = cleaned
        how = "text"

    converted = pd.to_numeric(candidate, errors="coerce")
    ok_fraction = converted.notna().sum() / max(len(text), 1)

    if ok_fraction >= 0.9:
        return converted, how

    return None, None


def clean_dataframe(dataframe):
    """
    Makes real-world files safe to analyse and tells the user what changed.
    Returns (cleaned_dataframe, notes).
    """

    df = dataframe.copy()
    notes = []

    df.columns = [str(c).strip() for c in df.columns]

    # duplicate column names -> make unique, never silently overwrite
    seen = {}
    new_names = []
    for c in df.columns:
        if c in seen:
            seen[c] += 1
            new_names.append(f"{c}_{seen[c]}")
            notes.append(f"Duplicate column name '{c}' was renamed to '{new_names[-1]}'.")
        else:
            seen[c] = 0
            new_names.append(c)
    df.columns = new_names

    for col in df.columns:
        if _is_text_column(df[col]):
            df[col] = df[col].map(lambda v: v.strip() if isinstance(v, str) else v)
            df[col] = df[col].replace("", np.nan)

    before = df.shape
    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
    if df.shape != before:
        notes.append(
            f"Removed {before[0] - df.shape[0]} empty row(s) and "
            f"{before[1] - df.shape[1]} empty column(s)."
        )

    for col in df.columns:
        if _is_text_column(df[col]):
            converted, how = _try_numeric(df[col])
            if converted is not None:
                df[col] = converted
                notes.append(f"Column '{col}' contained numbers stored as text ({how}); converted to numbers.")

    return df.reset_index(drop=True), notes


# ---------------------------------------------------------------- loading

def list_sheets(uploaded_file):
    """Sheet names of an Excel file ([] for CSV/TSV/TXT)."""

    name = uploaded_file.name.lower()
    if not name.endswith((".xlsx", ".xls")):
        return []

    uploaded_file.seek(0)
    sheets = pd.ExcelFile(uploaded_file).sheet_names
    uploaded_file.seek(0)
    return sheets


def _read_text_table(uploaded_file):
    raw = uploaded_file.read()
    uploaded_file.seek(0)

    last_error = None
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError as error:
            last_error = error
    else:
        raise ValueError(f"Could not read the file text: {last_error}")

    first_lines = "\n".join(text.splitlines()[:20])
    candidates = [",", "\t", ";", "|"]
    counts = {c: first_lines.count(c) for c in candidates}
    sep = max(counts, key=counts.get) if max(counts.values()) > 0 else ","

    return pd.read_csv(io.StringIO(text), sep=sep, engine="python", dtype=str)


def load_file(uploaded_file, sheet_name=None):
    """
    Generic loader -- any CSV/TSV/TXT/Excel. Detects the separator
    (comma, tab, semicolon, pipe), the text encoding, and cleans the table.
    What was changed is listed in df.attrs["notes"].
    """

    name = uploaded_file.name.lower()
    uploaded_file.seek(0)

    if name.endswith((".xlsx", ".xls")):
        raw = pd.read_excel(uploaded_file, sheet_name=sheet_name or 0)
    else:
        raw = _read_text_table(uploaded_file)

    if raw.empty or raw.shape[1] == 0:
        raise ValueError("The file contains no data.")

    df, notes = clean_dataframe(raw)

    if df.empty:
        raise ValueError("The file contains no data after removing empty rows and columns.")

    df.attrs["notes"] = notes
    return df

# END OF FILE: modules/figure_studio/data_loader.py