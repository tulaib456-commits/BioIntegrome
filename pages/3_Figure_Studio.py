import tempfile

import pandas as pd
import streamlit as st

from modules.auth import require_login
from modules.figure_studio.data_loader import load_file, detect_column_types
from modules.figure_studio.plot_builder import FigureBuilder
from modules.figure_studio.themes import PALETTES, THEMES, FONT_FAMILIES, apply_theme
from modules.figure_studio.export_component import render_exportable_plot
from modules.transcriptomics.result_display import display_results_table

require_login()

st.title("🎨 Figure Studio")
st.caption(
    "Build publication-quality, fully editable plots from any data — "
    "no R, no code. Export as high-res PNG or vector SVG straight from "
    "your browser."
)

uploaded = st.file_uploader("Upload your data", type=["csv", "tsv", "xlsx"])

if uploaded is not None:

    df = load_file(uploaded)
    st.session_state.figure_studio_data = df

    display_results_table(df, "fs_data", default_index=4)

    numeric_cols, categorical_cols = detect_column_types(df)
    all_cols = list(df.columns)
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
    km_result = None
    plot_note = None

    try:

        if plot_type == "Bar Chart":
            x = st.selectbox("X axis", categorical_cols + numeric_cols, key="bar_x")
            y = st.selectbox("Y axis", numeric_cols, key="bar_y")
            group = st.selectbox("Group/Color by (optional)", [None] + categorical_cols, key="bar_g")
            bar_summary = st.selectbox(
                "Each bar shows",
                ["Mean ± SD", "Mean ± SEM", "Mean only", "Median", "Sum"],
                key="bar_sum"
            )
            st.caption(
                "Every bar summarises all rows of that category. SD = spread of "
                "the data; SEM = precision of the mean (SD divided by √n). "
                "State which one you used in your figure legend."
            )
            fig = builder.bar_chart(
                x, y, group, active_palette, title=title, summary=bar_summary
            )

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
            st.caption(
                "Needs a time column (days/months), an event column "
                "(1 = event happened, 0 = censored) and optionally a group column."
            )
            duration_col = st.selectbox("Time column", numeric_cols, key="km_dur")
            event_col = st.selectbox("Event column (1 = event, 0 = censored)", numeric_cols, key="km_evt")
            group = st.selectbox("Group column (optional)", [None] + categorical_cols, key="km_grp")

            event_values = set(df[event_col].dropna().unique().tolist())
            if not event_values.issubset({0, 1}):
                st.error(
                    f"The event column '{event_col}' must contain only 0 and 1 "
                    f"(found: {sorted(event_values)[:8]}). Pick a different column."
                )
            else:
                km_data = df[[duration_col, event_col] + ([group] if group else [])].dropna()
                km_builder = FigureBuilder(km_data)
                fig, km_result = km_builder.kaplan_meier_plot(
                    duration_col, event_col, group, active_palette, title=title
                )
                if group and km_result is None:
                    plot_note = (
                        "Log-rank p-value is shown only when there are exactly "
                        "2 groups. With 3+ groups, compare pairs of groups separately."
                    )

        elif plot_type == "Forest Plot":
            st.caption(
                "One row per study/variable: a label, the estimate "
                "(e.g. hazard ratio) and its lower and upper confidence limits."
            )
            label_col = st.selectbox("Label column", all_cols, key="fo_lab")
            est_col = st.selectbox("Estimate column", numeric_cols, key="fo_est")
            lo_col = st.selectbox("Lower CI column", numeric_cols, key="fo_lo")
            hi_col = st.selectbox("Upper CI column", numeric_cols, key="fo_hi")
            default_ref = 1.0 if df[lo_col].min() > 0 else 0.0
            ref_line = st.number_input(
                "Reference line (0 for differences, 1 for ratios such as HR/OR)",
                value=default_ref, key=f"fo_ref_{default_ref}"
            )
            fig = builder.forest_plot(
                label_col, est_col, lo_col, hi_col, title=title,
                reference_line=ref_line
            )

        elif plot_type == "Venn Diagram (2-set)":
            st.caption(
                "Pick two columns. Each column is treated as a list of items "
                "(for example gene names); empty cells are ignored."
            )
            col_a = st.selectbox("Set A column", all_cols, index=0, key="ve_a")
            col_b = st.selectbox(
                "Set B column", all_cols, index=min(1, len(all_cols) - 1), key="ve_b"
            )
            label_a = st.text_input("Label for set A", str(col_a), key="ve_la")
            label_b = st.text_input("Label for set B", str(col_b), key="ve_lb")
            set_a = set(df[col_a].dropna().astype(str).str.strip())
            set_b = set(df[col_b].dropna().astype(str).str.strip())
            fig = builder.venn_diagram_2set(
                set_a, set_b, label_a=label_a, label_b=label_b, title=title
            )
            plot_note = (
                f"Set A: {len(set_a):,} unique items · Set B: {len(set_b):,} unique "
                f"items · Shared: {len(set_a & set_b):,}"
            )

        elif plot_type == "Paired/Connected Dot Plot":
            st.caption(
                "One line per subject (e.g. patient) connecting its values "
                "across time points or conditions."
            )
            id_col = st.selectbox("Subject ID column", all_cols, key="pa_id")
            time_col = st.selectbox("Time / condition column", all_cols, key="pa_time")
            value_col = st.selectbox("Value column", numeric_cols, key="pa_val")
            time_order = st.multiselect(
                "Order of time points / conditions (left to right)",
                df[time_col].dropna().unique().tolist(),
                default=df[time_col].dropna().unique().tolist(),
                key=f"pa_order_{time_col}"
            )
            fig = builder.paired_dot_plot(
                id_col, time_col, value_col, active_palette, title=title
            )
            if time_order and time_col not in numeric_cols:
                fig.update_xaxes(
                    type="category", categoryorder="array",
                    categoryarray=[str(t) for t in time_order]
                )
                st.caption(
                    "Paired Wilcoxon / paired t-test can be run in the "
                    "'Run a Statistical Test' box below (tick 'Paired')."
                )

    except Exception as error:
        st.error(
            "Could not build this plot with the selected columns. "
            f"Details: {error}"
        )
        fig = None

    if fig is not None:

        fig = apply_theme(fig, theme_name, font_family, font_size)

        st.session_state.figure_studio_fig = fig

        st.divider()
        with st.expander("Customize Axes, Fonts & Layout", expanded=False):

            custom_title = st.text_input("Figure Title")
            col_t1, col_t2, col_t3 = st.columns(3)
            with col_t1:
                title_bold = st.checkbox("Bold Title", value=True)
            with col_t2:
                title_size = st.slider("Title Font Size", 10, 36, 20)
            with col_t3:
                title_pos = st.selectbox("Title Position", ["left", "center", "right"], index=1)

            col_a, col_b = st.columns(2)
            with col_a:
                custom_x_title = st.text_input("X Axis Title")
                x_tick_size = st.slider("X Tick Font Size", 8, 24, 12)
                x_scale = st.selectbox("X Axis Scale", ["linear", "log"])
            with col_b:
                custom_y_title = st.text_input("Y Axis Title")
                y_tick_size = st.slider("Y Tick Font Size", 8, 24, 12)
                y_scale = st.selectbox("Y Axis Scale", ["linear", "log"])

            col_c, col_d, col_e = st.columns(3)
            with col_c:
                axis_title_size = st.slider("Axis Title Font Size", 10, 30, 16)
            with col_d:
                axis_bold = st.checkbox("Bold Axis Titles", value=False)
            with col_e:
                axis_color = st.color_picker("Axis Title Color", "#000000")

            legend_position = st.selectbox("Legend Position", ["right", "top", "bottom"])
            show_legend = st.checkbox("Show Legend", value=True)

            col_f, col_g = st.columns(2)
            with col_f:
                export_width = st.number_input("Figure Width (px)", 400, 3000, 1200)
            with col_g:
                export_height = st.number_input("Figure Height (px)", 300, 3000, 800)

            fig = builder.customize_layout(
                fig,
                x_axis_title=custom_x_title or None, y_axis_title=custom_y_title or None,
                x_tick_font_size=x_tick_size, y_tick_font_size=y_tick_size,
                axis_title_font_size=axis_title_size, axis_title_bold=axis_bold,
                axis_title_color=axis_color,
                legend_position=legend_position, show_legend=show_legend,
                width=export_width, height=export_height,
                x_scale=x_scale, y_scale=y_scale,
                title_text=custom_title or None, title_bold=title_bold,
                title_font_size=title_size, title_position=title_pos
            )

        # Kaplan-Meier: show the log-rank test result next to the figure
        if km_result is not None:
            p_val = km_result["p_value"]
            st.info(
                f"**Log-rank test (2 groups):** χ² = {km_result['chi_square']:.3f}, "
                f"p = {p_val:.4g}"
                + (" — significant at p < 0.05." if p_val < 0.05
                   else " — not significant at p < 0.05.")
            )
        if plot_note:
            st.caption(plot_note)

        # The chart itself: camera icon (PNG) + floppy icon (SVG) + two buttons
        render_exportable_plot(
            fig, key="figure_studio_main", filename="figure",
            export_width=int(fig.layout.width or 1200),
            export_height=int(fig.layout.height or 800),
            png_scale=3
        )
        st.caption(
            "💡 Download PNG or SVG with the two buttons under the chart, or "
            "with the camera (PNG) and floppy-disk (SVG) icons in the chart's "
            "top-right toolbar. The file is created in your browser, so it "
            "always works. The SVG opens in Illustrator, Inkscape or "
            "PowerPoint with editable text and shapes."
        )

        with st.expander("Other formats (PDF) — server-side, may not work on the cloud"):

            st.caption(
                "PDF needs a server-side Chrome that the free cloud servers "
                "often cannot start. If it fails, download the SVG above and "
                "save it as PDF from Inkscape or Illustrator."
            )

            export_format = st.selectbox(
                "Format", ["PDF", "SVG (server-side)", "PNG 300 DPI (server-side)"]
            )

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

                except Exception:
                    st.error(
                        "Server-side export failed — this is a known issue with "
                        "the cloud environment's Chrome sandbox, not your figure. "
                        "Use the PNG / SVG buttons under the chart instead."
                    )

            if st.session_state.get("figure_studio_export"):
                data_bytes, mime, ext = st.session_state.figure_studio_export
                st.download_button(
                    f"Download {ext.upper()}", data_bytes,
                    file_name=f"figure.{ext}", mime=mime
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
                    values_1 = df[df[group_col] == g1][value_col].dropna().values
                    values_2 = df[df[group_col] == g2][value_col].dropna().values
                    if paired and len(values_1) != len(values_2):
                        st.error(
                            "A paired test needs the same number of values in "
                            f"both groups (Group 1 has {len(values_1)}, "
                            f"Group 2 has {len(values_2)})."
                        )
                    else:
                        result = compare_two_groups(values_1, values_2, paired=paired)
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
                pair = df[[x_col, y_col]].dropna()
                result = test_correlation(pair[x_col], pair[y_col])
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
