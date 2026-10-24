import streamlit as st

LARGE_TABLE_CELLS = 1_000_000


def display_results_table(dataframe, key_prefix, long_text_columns=None, default_index=2):
    """
    Reusable table with a "how many rows" selector (10/30/50/100/All).
    default_index: 0=10, 1=30, 2=50, 3=100, 4=All.

    Safety: very large tables (over ~1 million cells) default to 100 rows
    instead of "All", because sending everything to the browser can freeze
    the app. The user can still choose "All" on purpose.
    """

    options = [10, 30, 50, 100, "All"]

    n_cells = dataframe.shape[0] * dataframe.shape[1]
    is_large = n_cells > LARGE_TABLE_CELLS

    if is_large and default_index == 4:
        default_index = 3

    display_count = st.selectbox(
        "Show rows", options, index=default_index,
        key=f"{key_prefix}_display_count"
    )

    if display_count == "All" and is_large:
        st.warning(
            f"Loading all {len(dataframe):,} rows x {dataframe.shape[1]:,} "
            "columns. This is a very large table and may be slow."
        )

    display_df = (
        dataframe if display_count == "All"
        else dataframe.head(display_count)
    )

    column_config = {}

    if long_text_columns:
        for col in long_text_columns:
            if col in display_df.columns:
                column_config[col] = st.column_config.TextColumn(col, width="large")

    st.dataframe(
        display_df,
        width="stretch",
        height=450,
        column_config=column_config if column_config else None,
        key=f"{key_prefix}_table"
    )

    st.caption(f"Showing {len(display_df):,} of {len(dataframe):,} total rows.")