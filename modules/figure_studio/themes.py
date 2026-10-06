"""
Publication-standard color palettes and visual themes.
Palette hex values verified against published standards
(Okabe-Ito colorblind-safe palette, viridis).
"""

PALETTES = {
    "Colorblind Safe (Okabe-Ito)": [
        "#E69F00", "#56B4E9", "#009E73", "#F0E442",
        "#0072B2", "#D55E00", "#CC79A7", "#000000"
    ],
    "Viridis": [
        "#440154", "#443983", "#31688e", "#21918c",
        "#35b779", "#90d743", "#fde725"
    ],
    "Nature-style (muted)": [
        "#4C72B0", "#DD8452", "#55A868", "#C44E52",
        "#8172B2", "#937860", "#DA8BC3", "#8C8C8C"
    ],
    "Science-style (bold)": [
        "#3B4992", "#EE0000", "#008B45", "#631879",
        "#008280", "#BB0021", "#5F559B", "#A20056"
    ],
    "Pastel": [
        "#A1C9F4", "#FFB482", "#8DE5A1", "#FF9F9B",
        "#D0BBFF", "#DEBB9B", "#FAB0E4", "#CFCFCF"
    ],
    "Grayscale (print-safe)": [
        "#1a1a1a", "#4d4d4d", "#808080", "#b3b3b3", "#e6e6e6"
    ]
}

THEMES = {
    "Classic (white background, black axes)": {
        "plot_bgcolor": "white",
        "paper_bgcolor": "white",
        "gridcolor": "#e0e0e0",
        "show_grid": True,
        "axis_line_color": "black",
        "axis_line_width": 1.5
    },
    "Minimal (no gridlines)": {
        "plot_bgcolor": "white",
        "paper_bgcolor": "white",
        "gridcolor": "white",
        "show_grid": False,
        "axis_line_color": "black",
        "axis_line_width": 1
    },
    "Dark": {
        "plot_bgcolor": "#1e1e1e",
        "paper_bgcolor": "#1e1e1e",
        "gridcolor": "#404040",
        "show_grid": True,
        "axis_line_color": "white",
        "axis_line_width": 1
    }
}

FONT_FAMILIES = [
    "Arial", "Helvetica", "Times New Roman", "Georgia", "Courier New"
]


def apply_theme(fig, theme_name, font_family="Arial", font_size=14):
    """Applies a named theme + font settings to any Plotly figure."""

    theme = THEMES.get(theme_name, THEMES["Classic (white background, black axes)"])

    fig.update_layout(
        plot_bgcolor=theme["plot_bgcolor"],
        paper_bgcolor=theme["paper_bgcolor"],
        font=dict(family=font_family, size=font_size, color=(
            "white" if theme_name == "Dark" else "black"
        )),
        xaxis=dict(
            showgrid=theme["show_grid"], gridcolor=theme["gridcolor"],
            showline=True, linecolor=theme["axis_line_color"],
            linewidth=theme["axis_line_width"]
        ),
        yaxis=dict(
            showgrid=theme["show_grid"], gridcolor=theme["gridcolor"],
            showline=True, linecolor=theme["axis_line_color"],
            linewidth=theme["axis_line_width"]
        )
    )

    return fig