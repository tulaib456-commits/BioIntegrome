import streamlit as st
import tempfile

from modules.auth import require_login
from modules.figure_studio.data_loader import load_file, detect_column_types
from modules.figure_studio.plot_builder import FigureBuilder
from modules.figure_studio.themes import PALETTES, THEMES, FONT_FAMILIES, apply_theme

require_login()

st.title("🎨 Figure Studio")
st.caption(
    "Build publication-quality, fully editable plots from any data — "
    "no R, no code. Export as high-res PNG, vector SVG, or PDF."
)

uploaded = st.file_uploader("Upload your data", type=["csv", "tsv", "xlsx"])

if uploaded is not None:

    df = load_file(uploaded)
    st.session_state.figure_studio_data = df

    st.dataframe(df.head())

    numeric_cols, categorical_cols = detect_column_types(df)
    builder = FigureBuilder(df)

    st.divider()

    plot_type = st.selectbox(
        "Plot Type",
        [
            "Bar Chart", "Box Plot", "Violin Plot", "Scatter Plot",
            "Line Chart", "Histogram", "Correlation Heatmap",
            "Grouped Comparison (with significance stats)",
            "Kaplan-Meier Survival Curve", "Forest Plot",
            "Venn Diagram (2-set)", "Paired/Connected Dot Plot"
        ]
    )

    col1, col2 = st.columns(2)

    with col1:
    palette_name = st.selectbox(
        "Color Palette", list(PALETTES.keys()) + ["Custom..."]
    )

    if palette_name == "Custom...":
        n_colors = st.slider("Number of colors", 2, 8, 3)
        custom_colors = []
        cols = st.columns(n_colors)
        for i in range(n_colors):
            with cols[i]:
                color = st.color_picker(f"Color {i+1}", "#1f77b4", key=f"custom_color_{i}")
                custom_colors.append(color)
        active_palette = custom_colors
    else:
        active_palette = PALETTES[palette_name]
    with col2:
        theme_name = st.selectbox("Theme", list(THEMES.keys()))

    col3, col4 = st.columns(2)
    with col3:
        font_family = st.selectbox("Font", FONT_FAMILIES)
    with col4:
        font_size = st.slider("Font Size", 8, 24, 14)

    title = st.text_input("Plot Title", "")

    fig = None

    if plot_type == "Bar Chart":
        x = st.selectbox("X axis", categorical_cols + numeric_cols, key="bar_x")
        y = st.selectbox("Y axis", numeric_cols, key="bar_y")
        group = st.selectbox("Group/Color by (optional)", [None] + categorical_cols, key="bar_g")
        fig = builder.bar_chart(x, y, group, active_palette, title=title)

    elif plot_type == "Box Plot":
        x = st.selectbox("X axis (group)", categorical_cols, key="box_x")
        y = st.selectbox("Y axis (value)", numeric_cols, key="box_y")
        group = st.selectbox("Color by (optional)", [None] + categorical_cols, key="box_g")
        fig = builder.box_plot(x, y, group, active_palette, title=title)

    elif plot_type == "Violin Plot":
        x = st.selectbox("X axis (group)", categorical_cols, key="vio_x")
        y = st.selectbox("Y axis (value)", numeric_cols, key="vio_y")
        group = st.selectbox("Color by (optional)", [None] + categorical_cols, key="vio_g")
        fig = builder.violin_plot(x, y, group, active_palette, title=title)

    elif plot_type == "Scatter Plot":
        x = st.selectbox("X axis", numeric_cols, key="sc_x")
        y = st.selectbox("Y axis", numeric_cols, key="sc_y")
        group = st.selectbox("Color by (optional)", [None] + categorical_cols, key="sc_g")
        trendline = st.checkbox("Show trendline (linear fit)")
        fig = builder.scatter_plot(
            x, y, group, palette=active_palette, title=title,
            trendline="ols" if trendline else None
        )

    elif plot_type == "Line Chart":
        x = st.selectbox("X axis", numeric_cols + categorical_cols, key="ln_x")
        y = st.selectbox("Y axis", numeric_cols, key="ln_y")
        group = st.selectbox("Group by (optional)", [None] + categorical_cols, key="ln_g")
        fig = builder.line_chart(x, y, group, active_palette, title=title)

    elif plot_type == "Histogram":
        x = st.selectbox("Value column", numeric_cols, key="hist_x")
        group = st.selectbox("Color by (optional)", [None] + categorical_cols, key="hist_g")
        bins = st.slider("Number of bins", 5, 100, 30)
        fig = builder.histogram(x, group, active_palette, bins, title=title)

    elif plot_type == "Correlation Heatmap":
        cols = st.multiselect("Columns to correlate", numeric_cols, default=numeric_cols[:10])
        if cols:
            fig = builder.correlation_heatmap(cols, title=title)

    elif plot_type == "Grouped Comparison (with significance stats)":
        value_col = st.selectbox("Value column", numeric_cols, key="gc_val")
        group_col = st.selectbox("Group column", categorical_cols, key="gc_grp")
        sub_type = st.radio("Style", ["box", "violin"], horizontal=True)
        fig = builder.grouped_comparison_with_stats(
            value_col, group_col, active_palette, title=title, plot_type=sub_type
        )
    elif plot_type == "Kaplan-Meier Survival Curve":
        duration_col = st.selectbox("Duration/Time column", numeric_cols, key="km_dur")
        event_col = st.selectbox("Event column (1=event, 0=censored)", numeric_cols, key="km_ev")
        group_col = st.selectbox("Group by (optional)", [None] + categorical_cols, key="km_grp")
        fig, log_rank_result = builder.kaplan_meier_plot(duration_col, event_col, group_col, active_palette, title=title)
        if log_rank_result:
            st.caption(f"Log-rank test: chi²={log_rank_result['chi_square']:.2f}, p={log_rank_result['p_value']:.4f}")

    elif plot_type == "Forest Plot":
        label_col = st.selectbox("Label column", categorical_cols, key="fp_label")
        estimate_col = st.selectbox("Estimate column", numeric_cols, key="fp_est")
        ci_lower_col = st.selectbox("CI Lower column", numeric_cols, key="fp_lo")
        ci_upper_col = st.selectbox("CI Upper column", numeric_cols, key="fp_hi")
        fig = builder.forest_plot(label_col, estimate_col, ci_lower_col, ci_upper_col, title=title)

    elif plot_type == "Venn Diagram (2-set)":
        set_col = st.selectbox("Category column", categorical_cols, key="venn_col")
        item_col = st.selectbox("Item/ID column", categorical_cols + numeric_cols, key="venn_item")
        categories = df[set_col].dropna().unique().tolist()
        if len(categories) >= 2:
            cat_a = st.selectbox("Set A", categories, index=0, key="venn_a")
            cat_b = st.selectbox("Set B", categories, index=1, key="venn_b")
            set_a = set(df[df[set_col] == cat_a][item_col])
            set_b = set(df[df[set_col] == cat_b][item_col])
            fig = builder.venn_diagram_2set(set_a, set_b, str(cat_a), str(cat_b), title=title)

    elif plot_type == "Paired/Connected Dot Plot":
        id_col = st.selectbox("Subject/ID column", categorical_cols, key="pd_id")
        time_col = st.selectbox("Time/Condition column", categorical_cols, key="pd_time")
        value_col = st.selectbox("Value column", numeric_cols, key="pd_val")
        fig = builder.paired_dot_plot(id_col, time_col, value_col, active_palette, title=title)

    if fig is not None:

        fig = apply_theme(fig, theme_name, font_family, font_size)

        st.plotly_chart(fig, use_container_width=True, key="figure_studio_main")
        st.session_state.figure_studio_fig = fig

        st.divider()
        st.subheader("Export")

        export_format = st.selectbox("Format", ["PNG (300 DPI)", "SVG (vector, editable)", "PDF"])

        if st.button("Generate Export File"):

            with tempfile.NamedTemporaryFile(delete=False, suffix=".tmp") as tmp:
                tmp_path = tmp.name

            try:
                if export_format.startswith("PNG"):
                    builder.export_png(fig, tmp_path)
                    mime, ext = "image/png", "png"
                elif export_format.startswith("SVG"):
                    builder.export_svg(fig, tmp_path)
                    mime, ext = "image/svg+xml", "svg"
                else:
                    builder.export_pdf(fig, tmp_path)
                    mime, ext = "application/pdf", "pdf"

                with open(tmp_path, "rb") as f:
                    st.session_state.figure_studio_export = (f.read(), mime, ext)

            except Exception as error:
                st.error(f"Export failed: {error}")

        if st.session_state.get("figure_studio_export"):
            data_bytes, mime, ext = st.session_state.figure_studio_export
            st.download_button(
                f"Download {ext.upper()}", data_bytes,
                file_name=f"figure.{ext}", mime=mime
            )

else:
    st.info("Upload a CSV, TSV, or Excel file to begin.")