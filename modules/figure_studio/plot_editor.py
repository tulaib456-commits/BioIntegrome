import streamlit as st
import plotly.graph_objects as go

from modules.figure_studio.themes import PALETTES, THEMES, FONT_FAMILIES, apply_theme
from modules.figure_studio.plot_builder import FigureBuilder

RECOLORABLE_TYPES = {"scatter", "scattergl", "bar", "box", "violin", "histogram"}


def render_editable_plot(fig, key_prefix, default_title="", enable_recolor=True):
    """
    Wraps ANY Plotly figure with the Figure Studio edit panel and
    shows it with a built-in PNG camera export.
    key_prefix must be unique for every chart on a page.
    """

    fig = go.Figure(fig)  # copy, so edits never stack up across reruns

    with st.expander("✏️ Edit this figure", expanded=False):

        col1, col2, col3 = st.columns(3)
        with col1:
            theme_name = st.selectbox("Theme", list(THEMES.keys()), key=f"{key_prefix}_theme")
        with col2:
            font_family = st.selectbox("Font", FONT_FAMILIES, key=f"{key_prefix}_font")
        with col3:
            font_size = st.slider("Base font size", 8, 24, 14, key=f"{key_prefix}_fsize")

        title_text = st.text_input("Figure title", default_title, key=f"{key_prefix}_title")
        col_t1, col_t2, col_t3 = st.columns(3)
        with col_t1:
            title_bold = st.checkbox("Bold title", True, key=f"{key_prefix}_tbold")
        with col_t2:
            title_size = st.slider("Title size", 10, 36, 20, key=f"{key_prefix}_tsize")
        with col_t3:
            title_pos = st.selectbox("Title position", ["left", "center", "right"], index=1, key=f"{key_prefix}_tpos")

        col_a, col_b = st.columns(2)
        with col_a:
            x_title = st.text_input("X axis title", key=f"{key_prefix}_xt")
            x_tick = st.slider("X tick size", 8, 24, 12, key=f"{key_prefix}_xts")
            x_scale = st.selectbox("X scale", ["linear", "log"], key=f"{key_prefix}_xsc")
        with col_b:
            y_title = st.text_input("Y axis title", key=f"{key_prefix}_yt")
            y_tick = st.slider("Y tick size", 8, 24, 12, key=f"{key_prefix}_yts")
            y_scale = st.selectbox("Y scale", ["linear", "log"], key=f"{key_prefix}_ysc")

        col_c, col_d, col_e = st.columns(3)
        with col_c:
            axis_size = st.slider("Axis title size", 10, 30, 16, key=f"{key_prefix}_ats")
        with col_d:
            axis_bold = st.checkbox("Bold axis titles", False, key=f"{key_prefix}_abold")
        with col_e:
            axis_color = st.color_picker("Axis title color", "#000000", key=f"{key_prefix}_acol")

        col_f, col_g = st.columns(2)
        with col_f:
            legend_pos = st.selectbox("Legend position", ["right", "top", "bottom"], key=f"{key_prefix}_lpos")
        with col_g:
            show_legend = st.checkbox("Show legend", True, key=f"{key_prefix}_lshow")

        col_h, col_i = st.columns(2)
        with col_h:
            width = st.number_input("Width (px)", 400, 3000, 1000, key=f"{key_prefix}_w")
        with col_i:
            height = st.number_input("Height (px)", 300, 3000, 600, key=f"{key_prefix}_h")

        recolor = False
        palette = None
        if enable_recolor:
            recolor = st.checkbox(
                "Recolor series with a palette (leave off for heatmaps or "
                "charts where color has meaning, e.g. volcano up/down)",
                False, key=f"{key_prefix}_recolor"
            )
            if recolor:
                palette_name = st.selectbox("Palette", list(PALETTES.keys()), key=f"{key_prefix}_pal")
                palette = PALETTES[palette_name]

    if recolor and palette:
        for i, trace in enumerate(fig.data):
            if trace.type in RECOLORABLE_TYPES:
                color = palette[i % len(palette)]
                if trace.type == "scatter" and trace.mode == "lines":
                    trace.update(line=dict(color=color))
                else:
                    trace.update(marker=dict(color=color))

    fig = apply_theme(fig, theme_name, font_family, font_size)

    fig = FigureBuilder(None).customize_layout(
        fig,
        x_axis_title=x_title or None, y_axis_title=y_title or None,
        x_tick_font_size=x_tick, y_tick_font_size=y_tick,
        axis_title_font_size=axis_size, axis_title_bold=axis_bold,
        axis_title_color=axis_color,
        legend_position=legend_pos, show_legend=show_legend,
        width=int(width), height=int(height),
        x_scale=x_scale, y_scale=y_scale,
        title_text=title_text or None, title_bold=title_bold,
        title_font_size=title_size, title_position=title_pos
    )

    st.plotly_chart(
        fig, width="content", key=f"{key_prefix}_chart",
        config={
            "toImageButtonOptions": {
                "format": "png", "filename": key_prefix,
                "height": int(height), "width": int(width), "scale": 3
            },
            "displaylogo": False
        }
    )

    return fig