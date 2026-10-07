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
            inner_cols = st.columns(n_colors)
            for i in range(n_colors):
                with inner_cols[i]:
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


    if fig is not None:

        fig = apply_theme(fig, theme_name, font_family, font_size)

        st.session_state.figure_studio_fig = fig

        st.divider()
        with st.expander("Customize Axes, Fonts & Layout", expanded=False):

            col_a, col_b = st.columns(2)
            with col_a:
                custom_x_title = st.text_input("X Axis Title (leave blank to keep default)")
                x_tick_size = st.slider("X Tick Font Size", 8, 24, 12)
            with col_b:
                custom_y_title = st.text_input("Y Axis Title (leave blank to keep default)")
                y_tick_size = st.slider("Y Tick Font Size", 8, 24, 12)

            axis_title_size = st.slider("Axis Title Font Size", 10, 30, 16)
            legend_position = st.selectbox("Legend Position", ["right", "top", "bottom"])
            show_legend = st.checkbox("Show Legend", value=True)

            col_c, col_d = st.columns(2)
            with col_c:
                export_width = st.number_input("Figure Width (px)", 400, 3000, 1200)
            with col_d:
                export_height = st.number_input("Figure Height (px)", 300, 3000, 800)

            fig = builder.customize_layout(
                fig,
                x_axis_title=custom_x_title or None,
                y_axis_title=custom_y_title or None,
                x_tick_font_size=x_tick_size, y_tick_font_size=y_tick_size,
                axis_title_font_size=axis_title_size,
                legend_position=legend_position, show_legend=show_legend,
                width=export_width, height=export_height
            )

        st.plotly_chart(
            fig, use_container_width=True, key="figure_studio_main",
            config={
                "toImageButtonOptions": {
                    "format": "png", "filename": "figure",
                    "height": 1200, "width": 1800, "scale": 3
                },
                "displaylogo": False
            }
        )
        st.caption(
            "💡 Use the camera icon in the top-right of the chart above "
            "for a reliable, instant PNG export — no server processing needed."
        )

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
            except Exception as error:
                st.error(
                    "Server-side export failed — this is a known issue with "
                    "the cloud environment's Chrome sandbox, not your figure. "
                    "Use the camera icon on the chart above for a reliable PNG, "
                    "or right-click → 'Save image as' on the chart."
                )

    st.divider()
    with st.expander("Run a Statistical Test (SPSS-style report)", expanded=False):

        from modules.figure_studio.statistics_toolkit import (
            compare_two_groups, compare_multiple_groups, test_correlation,
            test_categorical_association
        )

        test_kind = st.selectbox(
            "Test Type",
            ["Compare 2 Groups", "Compare 3+ Groups", "Correlation", "Categorical Association (Chi-square)"]
        )

        if test_kind == "Compare 2 Groups":
            value_col = st.selectbox("Value column", numeric_cols, key="stat_val2")
            group_col = st.selectbox("Group column", categorical_cols, key="stat_grp2")
            groups = df[group_col].dropna().unique().tolist()
            if len(groups) >= 2:
                g1 = st.selectbox("Group 1", groups, index=0, key="stat_g1")
                g2 = st.selectbox("Group 2", groups, index=1, key="stat_g2")
                paired = st.checkbox("Paired/repeated measures?")
                if st.button("Run Test"):
                    result = compare_two_groups(
                        df[df[group_col] == g1][value_col].dropna().values,
                        df[df[group_col] == g2][value_col].dropna().values,
                        paired=paired
                    )
                    st.write(f"**Method (auto-selected):** {result['method']}")
                    st.write(f"**Statistic:** {result['statistic']:.4f}")
                    st.write(f"**P-value:** {result['p_value']:.6f}")
                    st.write(f"**Significant (p<0.05):** {result['significant']}")
                    st.write(f"**Cohen's d (effect size):** {result['cohens_d']:.3f}")
                    st.caption(f"Group 1 normality: {result['normality_a']['note']}")
                    st.caption(f"Group 2 normality: {result['normality_b']['note']}")

        elif test_kind == "Compare 3+ Groups":
            value_col = st.selectbox("Value column", numeric_cols, key="stat_val3")
            group_col = st.selectbox("Group column", categorical_cols, key="stat_grp3")
            if st.button("Run Test", key="stat_run3"):
                groups_dict = {g: df[df[group_col] == g][value_col].dropna().values
                              for g in df[group_col].dropna().unique()}
                result = compare_multiple_groups(groups_dict)
                st.write(f"**Method (auto-selected):** {result['method']}")
                st.write(f"**Statistic:** {result['statistic']:.4f}")
                st.write(f"**P-value:** {result['p_value']:.6f}")
                st.write(f"**Significant (p<0.05):** {result['significant']}")
                st.dataframe(result["group_means"])

        elif test_kind == "Correlation":
            x_col = st.selectbox("X variable", numeric_cols, key="stat_corr_x")
            y_col = st.selectbox("Y variable", numeric_cols, key="stat_corr_y")
            if st.button("Run Test", key="stat_run_corr"):
                result = test_correlation(df[x_col].dropna(), df[y_col].dropna())
                st.write(f"**Method (auto-selected):** {result['method']}")
                st.write(f"**r:** {result['r']:.4f}")
                st.write(f"**P-value:** {result['p_value']:.6f}")

        else:
            cat_a = st.selectbox("Categorical variable A", categorical_cols, key="stat_cat_a")
            cat_b = st.selectbox("Categorical variable B", categorical_cols, key="stat_cat_b")
            if st.button("Run Test", key="stat_run_chi"):
                table = pd.crosstab(df[cat_a], df[cat_b])
                result = test_categorical_association(table)
                st.write(f"**Chi-square:** {result['chi2']:.4f}")
                st.write(f"**P-value:** {result['p_value']:.6f}")
                st.write(f"**Degrees of freedom:** {result['dof']}")

else:
    st.info("Upload a CSV, TSV, or Excel file to begin.")