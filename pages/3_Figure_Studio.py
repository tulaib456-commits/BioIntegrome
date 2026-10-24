import tempfile

import pandas as pd
import streamlit as st

from modules.auth import require_login
from modules.figure_studio.data_loader import load_file, list_sheets, detect_column_types
from modules.figure_studio.data_prep import (
    suggest_layout, wide_to_long, code_like_columns, as_categories,
    paired_arrays, looks_like_id
)
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

uploaded = st.file_uploader("Upload your data", type=["csv", "tsv", "txt", "xlsx"])

if uploaded is not None:

    sheet = None
    try:
        sheets = list_sheets(uploaded)
        if len(sheets) > 1:
            sheet = st.selectbox("Which Excel sheet?", sheets)
        raw_df = load_file(uploaded, sheet_name=sheet)
    except Exception as error:
        st.error(f"Could not read this file. Details: {error}")
        st.stop()

    for note in raw_df.attrs.get("notes", []):
        st.caption(f"ℹ️ {note}")

    st.subheader("1. Your data")
    display_results_table(raw_df, "fs_data", default_index=4)

    # ---------------- how is the data arranged?
    st.subheader("2. How is your data arranged?")

    suggested = suggest_layout(raw_df)
    layout = st.radio(
        "Layout",
        [
            "Long — one column says which group each row belongs to (e.g. Group | Value)",
            "Wide — each group or time point is its own column (e.g. ID | Before | After)",
        ],
        index=1 if suggested == "wide" else 0,
        key="fs_layout",
        help="Most Excel sheets for paired data (Before/After) are 'wide'. "
             "Figure Studio converts them automatically."
    )

    df = raw_df
    wide_id_col = None

    if layout.startswith("Wide"):

        raw_numeric, raw_categorical = detect_column_types(raw_df)

        id_options = [None] + list(raw_df.columns)
        id_default = next(
            (c for c in raw_df.columns if looks_like_id(c, raw_df[c])), None
        )
        wide_id_col = st.selectbox(
            "Subject / sample ID column (optional, needed for paired tests)",
            id_options,
            index=id_options.index(id_default) if id_default in id_options else 0,
            format_func=lambda c: "(none)" if c is None else str(c),
            key="fs_wide_id"
        )

        candidates = [c for c in raw_numeric if c != wide_id_col]
        value_cols = st.multiselect(
            "Columns to compare (each becomes one group)",
            candidates, default=candidates, key=f"fs_wide_cols_{wide_id_col}"
        )

        if len(value_cols) < 2:
            st.warning("Pick at least 2 columns to compare.")
            st.stop()

        try:
            df, _group_name, _value_name = wide_to_long(
                raw_df, value_cols, id_col=wide_id_col
            )
        except ValueError as error:
            st.error(str(error))
            st.stop()

        st.caption(
            f"Reshaped to long format: {len(df)} rows with the columns "
            f"{', '.join(map(str, df.columns))}. Use '{_group_name}' as the "
            f"group and '{_value_name}' as the value in the plots below."
        )

    else:
        codes = code_like_columns(raw_df)
        if codes:
            as_cat = st.multiselect(
                "These numeric columns look like group codes (e.g. 0/1/2). "
                "Treat them as categories?",
                codes, key="fs_codes"
            )
            if as_cat:
                df = as_categories(raw_df, as_cat)

    st.session_state.figure_studio_data = df

    numeric_cols, categorical_cols = detect_column_types(df)

    # After reshaping a wide table, offer the new Value column first so every
    # dropdown starts on the right column (not on the ID column).
    if layout.startswith("Wide") and _value_name in numeric_cols:
        numeric_cols = [_value_name] + [c for c in numeric_cols if c != _value_name]

    all_cols = list(df.columns)
    builder = FigureBuilder(df)

    st.subheader("3. Plot")

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

        if plot_type != "Venn Diagram (2-set)" and not numeric_cols:
            raise ValueError(
                "this plot needs at least one numeric column, and your table has none."
            )

        if plot_type in ("Box Plot", "Violin Plot", "Grouped Comparison (with significance stats)") \
                and not categorical_cols:
            raise ValueError(
                "this plot needs a text column that names the groups. Choose the "
                "'Wide' layout above if each group is its own column."
            )

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

            col_s1, col_s2 = st.columns(2)
            with col_s1:
                gc_test_label = st.selectbox(
                    "Statistical test between groups",
                    ["Automatic (Shapiro-Wilk decides)", "Welch's t-test",
                     "Mann-Whitney U (non-parametric)", "Tukey HSD (after ANOVA)"],
                    key="gc_test"
                )
            with col_s2:
                gc_corr_label = st.selectbox(
                    "Multiple-comparison correction",
                    ["Holm (recommended)", "Bonferroni", "Benjamini-Hochberg (FDR)",
                     "None (not recommended)"],
                    key="gc_corr",
                    disabled=gc_test_label.startswith("Tukey")
                )
            gc_show_ns = st.checkbox("Show non-significant (ns) brackets", value=True, key="gc_ns")

            gc_test = {"Automatic": "auto", "Welch": "welch",
                       "Mann-Whitney": "mannwhitney", "Tukey": "tukey"}[
                next(k for k in ("Automatic", "Welch", "Mann-Whitney", "Tukey")
                     if gc_test_label.startswith(k))]
            gc_corr = {"Holm": "holm", "Bonferroni": "bonferroni",
                       "Benjamini-Hochberg": "bh", "None": "none"}[
                next(k for k in ("Holm", "Bonferroni", "Benjamini-Hochberg", "None")
                     if gc_corr_label.startswith(k))]

            fig = builder.grouped_comparison_with_stats(
                value_col, group_col, active_palette, title=title,
                plot_type=sub_type, test=gc_test, correction=gc_corr,
                show_ns=gc_show_ns
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
            _pa_id_default = wide_id_col if (layout.startswith("Wide") and wide_id_col) else (
                "Row" if (layout.startswith("Wide") and "Row" in all_cols) else all_cols[0])
            _pa_time_default = _group_name if layout.startswith("Wide") else all_cols[min(1, len(all_cols) - 1)]
            id_col = st.selectbox("Subject ID column", all_cols,
                                  index=all_cols.index(_pa_id_default), key="pa_id")
            time_col = st.selectbox("Time / condition column", all_cols,
                                    index=all_cols.index(_pa_time_default), key="pa_time")
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

        if x_scale == "log" and fig.layout.xaxis.type != "log":
            st.warning("Log scale was NOT applied to the X axis: it needs numeric values that are all above 0.")
        if y_scale == "log" and fig.layout.yaxis.type != "log":
            st.warning("Log scale was NOT applied to the Y axis: it needs numeric values that are all above 0.")

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

        pairwise_table = getattr(builder, "last_pairwise_table", None)
        if plot_type.startswith("Grouped Comparison") and pairwise_table is not None:
            st.subheader("Statistics behind the stars")
            shown = pairwise_table[
                ["Group_1", "Group_2", "n1", "n2", "Mean_Diff", "Test",
                 "P_Value", "Adj_P_Value", "Correction", "Stars"]
            ]
            st.dataframe(
                shown.style.format({"Mean_Diff": "{:.3f}", "P_Value": "{:.4g}", "Adj_P_Value": "{:.4g}"}),
                width="stretch"
            )
            st.caption(
                "Stars are based on the ADJUSTED p-value: * < 0.05, ** < 0.01, "
                "*** < 0.001, ns = not significant. Report the test and the correction "
                "in your figure legend. Groups with fewer than 2 values are left out."
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

        test_options = ["Compare 2 Groups", "Compare 3+ Groups", "Correlation",
                        "Categorical Association (Chi-square)"]
        test_kind = st.selectbox("Test Type", test_options)

        if not numeric_cols and not test_kind.startswith("Categorical"):
            st.info("This test needs at least one numeric column, and your table has none.")
            st.stop()

        no_groups_msg = (
            "This test needs a column that says which group each row belongs to, "
            "and your table has none. If each group is its own column "
            "(e.g. Before | After), choose the 'Wide' layout in step 2 above."
        )

        try:

            if test_kind == "Compare 2 Groups":

                if not categorical_cols:
                    st.info(no_groups_msg)
                else:
                    value_col = st.selectbox("Value column", numeric_cols, key="stat_val2")
                    group_col = st.selectbox("Group column", categorical_cols, key="stat_grp2")
                    groups = df[group_col].dropna().unique().tolist()

                    if len(groups) < 2:
                        st.info(f"'{group_col}' has only one group, so there is nothing to compare.")
                    else:
                        g1 = st.selectbox("Group 1", groups, index=0, key="stat_g1")
                        g2 = st.selectbox("Group 2", groups, index=1, key="stat_g2")
                        paired = st.checkbox(
                            "Paired / repeated measures (same subject measured twice)",
                            value=(layout.startswith("Wide") and wide_id_col is not None),
                            key="stat_paired"
                        )

                        id_for_pairs = None
                        if paired:
                            id_default = (wide_id_col if (layout.startswith("Wide") and wide_id_col)
                                          else "Row" if (layout.startswith("Wide") and "Row" in all_cols)
                                          else next((c for c in all_cols if looks_like_id(c, df[c])), all_cols[0]))
                            id_for_pairs = st.selectbox(
                                "Subject ID column (links the two measurements of one subject)",
                                all_cols, index=all_cols.index(id_default), key="stat_pair_id"
                            )

                        if g1 == g2:
                            st.warning("Choose two different groups.")
                        elif st.button("Run Test", key="stat_run2"):

                            if paired:
                                values_1, values_2, n_dropped = paired_arrays(
                                    df, id_for_pairs, group_col, value_col, g1, g2
                                )
                                if n_dropped:
                                    st.caption(
                                        f"{n_dropped} subject(s) lacked a value in one of the two "
                                        "conditions and were left out of the paired test."
                                    )
                            else:
                                values_1 = df[df[group_col] == g1][value_col].dropna().values
                                values_2 = df[df[group_col] == g2][value_col].dropna().values

                            result = compare_two_groups(values_1, values_2, paired=paired)

                            st.write(f"**Method (auto-selected):** {result['method']}")
                            st.write(f"**Sample size:** n = {result['n_a']} vs {result['n_b']}")
                            st.write(f"**Means:** {result['mean_a']:.4g} vs {result['mean_b']:.4g}")
                            st.write(f"**Statistic:** {result['statistic']:.4f}")
                            st.write(f"**P-value:** {result['p_value']:.6f}")
                            st.write(f"**Significant (p<0.05):** {result['significant']}")
                            st.write(f"**{result['effect_size_name']} (effect size):** {result['cohens_d']:.3f}")
                            if paired and result["normality_diff"]:
                                st.caption(f"Differences normality: {result['normality_diff']['note']}")
                            else:
                                st.caption(f"Group 1 normality: {result['normality_a']['note']}")
                                st.caption(f"Group 2 normality: {result['normality_b']['note']}")
                            if result["note"]:
                                st.warning(result["note"])

            elif test_kind == "Compare 3+ Groups":

                if not categorical_cols:
                    st.info(no_groups_msg)
                else:
                    value_col = st.selectbox("Value column", numeric_cols, key="stat_val3")
                    group_col = st.selectbox("Group column", categorical_cols, key="stat_grp3")

                    if st.button("Run Test", key="stat_run3"):
                        groups_dict = {
                            g: df[df[group_col] == g][value_col].dropna().values
                            for g in df[group_col].dropna().unique()
                        }
                        result = compare_multiple_groups(groups_dict)
                        st.write(f"**Method (auto-selected):** {result['method']}")
                        st.write(f"**Statistic:** {result['statistic']:.4f}")
                        st.write(f"**P-value:** {result['p_value']:.6f}")
                        st.write(f"**Significant (p<0.05):** {result['significant']}")
                        if result["dropped_groups"]:
                            st.warning(
                                "Left out (fewer than 2 values): "
                                + ", ".join(result["dropped_groups"])
                            )
                        summary_table = pd.DataFrame({
                            "Group": list(result["group_means"]),
                            "n": [result["group_n"][g] for g in result["group_means"]],
                            "Mean": list(result["group_means"].values()),
                        })
                        st.dataframe(summary_table, width="stretch")

                        if result["posthoc"] is not None:
                            st.write("**Post-hoc pairwise comparisons** "
                                     "(Tukey HSD after ANOVA; Mann-Whitney with Holm correction after Kruskal-Wallis):")
                            st.dataframe(
                                result["posthoc"][["Group_1", "Group_2", "Mean_Diff", "Test",
                                                   "P_Value", "Adj_P_Value", "Stars"]],
                                width="stretch"
                            )
                        elif result["significant"] is False:
                            st.caption("No post-hoc tests are shown because the overall test is not significant.")

            elif test_kind == "Correlation":

                if len(numeric_cols) < 2:
                    st.info("Correlation needs at least 2 numeric columns.")
                else:
                    x_col = st.selectbox("X variable", numeric_cols, key="stat_corr_x")
                    y_col = st.selectbox("Y variable", numeric_cols,
                                         index=min(1, len(numeric_cols) - 1), key="stat_corr_y")
                    if st.button("Run Test", key="stat_run_corr"):
                        pair = df[[x_col, y_col]].dropna()
                        result = test_correlation(pair[x_col].values, pair[y_col].values)
                        st.write(f"**Method (auto-selected):** {result['method']}")
                        st.write(f"**Complete pairs used:** n = {result['n']}")
                        st.write(f"**r:** {result['r']:.4f}")
                        st.write(f"**P-value:** {result['p_value']:.6f}")

            else:

                if len(categorical_cols) < 2:
                    st.info(
                        "The chi-square test compares two categorical columns "
                        "(e.g. Treatment and Outcome). Your table has fewer than two. "
                        "Numeric 0/1/2 codes can be turned into categories in step 2."
                    )
                else:
                    cat_a = st.selectbox("Categorical variable A", categorical_cols, key="stat_cat_a")
                    cat_b = st.selectbox("Categorical variable B", categorical_cols,
                                         index=1, key="stat_cat_b")
                    if cat_a == cat_b:
                        st.warning("Choose two different columns.")
                    elif st.button("Run Test", key="stat_run_chi"):
                        table = pd.crosstab(df[cat_a], df[cat_b])
                        result = test_categorical_association(table.values)
                        st.dataframe(table, width="stretch")
                        st.write(f"**Method:** {result['method']}")
                        st.write(f"**Chi-square:** {result['chi2']:.4f}")
                        st.write(f"**P-value:** {result['p_value']:.6f}")
                        st.write(f"**Degrees of freedom:** {result['dof']}")
                        st.write(f"**Cramér's V (effect size):** {result['cramers_v']:.3f}")
                        if result["note"]:
                            st.warning(result["note"])

        except ValueError as error:
            st.error(f"This test cannot be run on the selected data: {error}")
        except Exception as error:
            st.error(
                "Something unexpected went wrong while running this test. "
                f"Please send this message to support: {error}"
            )

else:
    st.info("Upload a CSV, TSV, or Excel file to begin.")
