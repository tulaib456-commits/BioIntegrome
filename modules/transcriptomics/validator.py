import pandas as pd

from config.constants import GENE_COLUMN_CANDIDATES


def detect_gene_column(dataframe):

    for column in dataframe.columns:

        if column in GENE_COLUMN_CANDIDATES:

            return column

    return dataframe.columns[0]


def detect_sample_columns(dataframe):
    """
    Every NUMERIC column except the gene/protein column.
    Text columns (e.g. a 'Description' or 'Biotype' annotation next to the
    counts) are not samples and are skipped instead of breaking the maths.
    """

    gene_column = detect_gene_column(dataframe)

    return [

        column

        for column in dataframe.columns

        if column != gene_column
        and pd.api.types.is_numeric_dtype(dataframe[column])

    ]


def count_missing_values(dataframe):

    return dataframe.isna().sum().sum()


def duplicate_gene_count(dataframe):

    gene_column = detect_gene_column(dataframe)

    return dataframe[gene_column].duplicated().sum()