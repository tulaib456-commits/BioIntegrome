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
        if pd.api.types.is_numeric_dtype(dataframe[col]):
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)

    return numeric_cols, categorical_cols


def load_file(uploaded_file):
    """Generic loader -- any CSV/TSV/Excel, unlike the domain-specific modules."""

    name = uploaded_file.name.lower()

    if name.endswith(".xlsx"):
        return pd.read_excel(uploaded_file)
    elif name.endswith(".tsv"):
        return pd.read_csv(uploaded_file, sep="\t")
    else:
        return pd.read_csv(uploaded_file)