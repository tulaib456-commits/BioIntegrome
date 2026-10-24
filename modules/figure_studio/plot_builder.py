import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

from modules.figure_studio.themes import apply_theme


class FigureBuilder:
    """
    Generic, fully-customizable publication plotting -- works on
    ANY tabular data, not tied to a specific scientific domain.
    """

    def __init__(self, dataframe):
        self.df = dataframe

    def bar_chart(self, x, y, group_col=None, palette=None, barmode="group",
                  title="", x_label="", y_label="", summary="Mean ± SD"):
        """
        Bar chart of a SUMMARY of y for every category (and group).

        IMPORTANT: Plotly's own bar chart stacks (adds up) every row, which
        gives meaningless sums for experimental data. Journals expect
        mean +/- error bars, so that is the default here.

        summary: "Mean ± SD", "Mean ± SEM", "Mean only", "Median" or "Sum".
        SD  = standard deviation (spread of the data).
        SEM = standard error of the mean = SD / sqrt(n) (precision of the mean).
        """

        if group_col is None or group_col == x:
            keys = [x]
        else:
            keys = [x, group_col]

        grouped = self.df.groupby(keys, sort=False, dropna=True)[y]
        n = grouped.count()
        sd = grouped.std(ddof=1)

        error = None
        if summary == "Median":
            value = grouped.median()
            label = f"Median of {y}"
        elif summary == "Sum":
            value = grouped.sum()
            label = f"Sum of {y}"
        else:
            value = grouped.mean()
            label = f"Mean of {y}"
            if "SD" in summary:
                error = sd
                label += " (± SD)"
            elif "SEM" in summary:
                error = sd / np.sqrt(n)
                label += " (± SEM)"

        table = value.rename("_value").to_frame()
        if error is not None:
            table["_error"] = error
        table = table.reset_index()

        fig = px.bar(
            table, x=x, y="_value", color=group_col, barmode=barmode,
            error_y="_error" if error is not None else None,
            color_discrete_sequence=palette, title=title,
            labels={"_value": label, "_error": "Error"}
        )

        fig.update_layout(
            xaxis_title=x_label or x, yaxis_title=y_label or label
        )

        return fig

    def box_plot(self, x, y, group_col=None, palette=None,
                 title="", x_label="", y_label="", show_points="outliers"):

        fig = px.box(
            self.df, x=x, y=y, color=group_col,
            color_discrete_sequence=palette, title=title,
            points=show_points
        )

        fig.update_layout(xaxis_title=x_label or x, yaxis_title=y_label or y)

        return fig

    def violin_plot(self, x, y, group_col=None, palette=None,
                    title="", x_label="", y_label="", show_box=True, show_points=False):

        fig = px.violin(
            self.df, x=x, y=y, color=group_col,
            color_discrete_sequence=palette, title=title,
            box=show_box, points="all" if show_points else False
        )

        fig.update_layout(xaxis_title=x_label or x, yaxis_title=y_label or y)

        return fig

    def scatter_plot(self, x, y, group_col=None, size_col=None, palette=None,
                     title="", x_label="", y_label="", trendline=None, point_size=8):

        fig = px.scatter(
            self.df, x=x, y=y, color=group_col, size=size_col,
            color_discrete_sequence=palette, title=title,
            trendline=trendline
        )

        if not size_col:
            fig.update_traces(marker=dict(size=point_size))

        fig.update_layout(xaxis_title=x_label or x, yaxis_title=y_label or y)

        return fig

    def line_chart(self, x, y, group_col=None, palette=None,
                   title="", x_label="", y_label="", markers=True):

        fig = px.line(
            self.df, x=x, y=y, color=group_col,
            color_discrete_sequence=palette, title=title,
            markers=markers
        )

        fig.update_layout(xaxis_title=x_label or x, yaxis_title=y_label or y)

        return fig

    def histogram(self, x, group_col=None, palette=None, bins=30,
                  title="", x_label="", y_label="Count", barmode="overlay"):

        fig = px.histogram(
            self.df, x=x, color=group_col, nbins=bins,
            color_discrete_sequence=palette, title=title,
            barmode=barmode, opacity=0.7 if group_col else 1.0
        )

        fig.update_layout(xaxis_title=x_label or x, yaxis_title=y_label)

        return fig

    def correlation_heatmap(self, columns, method="pearson", palette_scale="RdBu_r", title=""):

        corr = self.df[columns].corr(method=method)

        fig = px.imshow(
            corr, color_continuous_scale=palette_scale,
            title=title, text_auto=".2f", zmin=-1, zmax=1
        )

        return fig

    def grouped_comparison_with_stats(self, value_col, group_col, palette=None,
                                       title="", x_label="", y_label="",
                                       plot_type="box", compare_all_pairs=True,
                                       test="auto", correction="holm",
                                       show_ns=True):
        """
        Box/violin plot with significance brackets between every pair of groups.

        Statistics (what the stars mean):
          test        'auto' (Shapiro-Wilk decides Welch t / Mann-Whitney),
                      'welch', 'mannwhitney' or 'tukey' (Tukey HSD)
          correction  'holm' (default), 'bonferroni', 'bh', 'none' --
                      applied across ALL pairs, because testing many pairs
                      without correction produces false positives.
        The stars are based on the ADJUSTED p-values. The full table is kept
        in self.last_pairwise_table so the page can show it under the figure.
        """

        from modules.figure_studio.statistics_toolkit import pairwise_comparisons

        data = self.df[[group_col, value_col]].dropna()

        # one fixed order, used by BOTH the plot and the brackets
        groups = list(dict.fromkeys(data[group_col].tolist()))
        order = {group_col: groups}

        if plot_type == "violin":
            fig = px.violin(
                data, x=group_col, y=value_col, color=group_col,
                color_discrete_sequence=palette, box=True, title=title,
                category_orders=order
            )
        else:
            fig = px.box(
                data, x=group_col, y=value_col, color=group_col,
                color_discrete_sequence=palette, title=title,
                category_orders=order
            )

        fig.update_layout(
            xaxis_title=x_label or group_col,
            yaxis_title=y_label or value_col,
            showlegend=False
        )

        self.last_pairwise_table = None

        usable = [g for g in groups if (data[data[group_col] == g][value_col].notna().sum() >= 2)]
        if len(usable) < 2 or not compare_all_pairs:
            return fig

        table = pairwise_comparisons(
            {g: data[data[group_col] == g][value_col].values for g in usable},
            test=test, correction=correction
        )
        self.last_pairwise_table = table

        max_val = data[value_col].max()
        min_val = data[value_col].min()
        value_range = (max_val - min_val) or 1.0

        bracket_level = 0

        for _, row in table.iterrows():

            if not show_ns and not row["Significant"]:
                continue

            x_a = groups.index(row["Group_1"])
            x_b = groups.index(row["Group_2"])

            y_bracket = max_val + value_range * (0.08 + bracket_level * 0.12)
            bracket_level += 1

            for xs, ys in (
                ((x_a, x_a), (y_bracket - value_range * 0.02, y_bracket)),
                ((x_b, x_b), (y_bracket - value_range * 0.02, y_bracket)),
                ((x_a, x_b), (y_bracket, y_bracket)),
            ):
                fig.add_shape(
                    type="line", x0=xs[0], x1=xs[1], y0=ys[0], y1=ys[1],
                    line=dict(color="black", width=1.2)
                )

            fig.add_annotation(
                x=(x_a + x_b) / 2, y=y_bracket + value_range * 0.02,
                text=row["Stars"], showarrow=False,
                font=dict(size=16, color="black")
            )

        fig.update_yaxes(
            range=[min_val - value_range * 0.05,
                   max_val + value_range * (0.15 + bracket_level * 0.12)]
        )

        return fig

    def export_png(self, fig, filepath, width=1200, height=800, scale=3):
        """scale=3 at 1200x800 gives ~300 DPI equivalent, standard journal requirement."""
        fig.write_image(filepath, format="png", width=width, height=height, scale=scale)

    def export_svg(self, fig, filepath, width=1200, height=800):
        """Vector format -- fully editable in Illustrator/Inkscape after export."""
        fig.write_image(filepath, format="svg", width=width, height=height)

    def export_pdf(self, fig, filepath, width=1200, height=800):
        fig.write_image(filepath, format="pdf", width=width, height=height)

    def kaplan_meier_plot(self, duration_col, event_col, group_col=None,
                          palette=None, title="", x_label="Time", y_label="Survival Probability"):

        from modules.figure_studio.survival_analysis import kaplan_meier, log_rank_test

        fig = go.Figure()
        palette = palette or ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]

        if group_col:
            groups = sorted(self.df[group_col].dropna().unique())
        else:
            groups = [None]

        group_data = {}

        for i, group in enumerate(groups):

            if group is not None:
                subset = self.df[self.df[group_col] == group]
                label = str(group)
            else:
                subset = self.df
                label = "All"

            km = kaplan_meier(subset[duration_col].values, subset[event_col].values)
            group_data[label] = (subset[duration_col].values, subset[event_col].values)

            fig.add_trace(go.Scatter(
                x=km["time"], y=km["survival"], mode="lines",
                line=dict(shape="hv", color=palette[i % len(palette)], width=2),
                name=label
            ))

        fig.update_layout(
            title=title, xaxis_title=x_label, yaxis_title=y_label,
            yaxis=dict(range=[0, 1.05])
        )

        log_rank_result = None
        if group_col and len(groups) == 2:
            labels = list(group_data.keys())
            d_a, e_a = group_data[labels[0]]
            d_b, e_b = group_data[labels[1]]
            chi_sq, p_val = log_rank_test(d_a, e_a, d_b, e_b)
            log_rank_result = {"chi_square": chi_sq, "p_value": p_val}

        return fig, log_rank_result

    def forest_plot(self, label_col, estimate_col, ci_lower_col, ci_upper_col,
                    title="", x_label="Effect Size", reference_line=0):

        if len({label_col, estimate_col, ci_lower_col, ci_upper_col}) < 4:
            raise ValueError(
                "Label, Estimate, Lower CI and Upper CI must be four DIFFERENT columns."
            )

        data = self.df[[label_col, estimate_col, ci_lower_col, ci_upper_col]].dropna()

        bad = (data[ci_lower_col] > data[estimate_col]) | (data[ci_upper_col] < data[estimate_col])
        if bad.any():
            raise ValueError(
                "Lower CI must be <= estimate <= Upper CI in every row "
                f"(check row(s): {', '.join(str(i + 2) for i in data.index[bad][:5])} "
                "of your file, counting the header as row 1)."
            )

        data = data.sort_values(estimate_col)

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=data[estimate_col], y=data[label_col], mode="markers",
            marker=dict(size=10, color="black"),
            showlegend=False, name="Estimate",
            error_x=dict(
                type="data",
                symmetric=False,
                array=data[ci_upper_col] - data[estimate_col],
                arrayminus=data[estimate_col] - data[ci_lower_col],
                color="black", thickness=1.5
            )
        ))

        fig.add_vline(x=reference_line, line_dash="dash", line_color="gray")

        fig.update_layout(
            title=title, xaxis_title=x_label, yaxis_title="",
            showlegend=False
        )

        return fig

    def venn_diagram_2set(self, set_a, set_b, label_a="Set A", label_b="Set B", title=""):

        only_a = len(set_a - set_b)
        only_b = len(set_b - set_a)
        both = len(set_a & set_b)

        fig = go.Figure()

        fig.add_shape(type="circle", x0=0, y0=0, x1=4, y1=4,
                      line=dict(color="#0072B2", width=2), fillcolor="#0072B2", opacity=0.3)
        fig.add_shape(type="circle", x0=2.3, y0=0, x1=6.3, y1=4,
                      line=dict(color="#D55E00", width=2), fillcolor="#D55E00", opacity=0.3)

        fig.add_annotation(x=1.3, y=2, text=str(only_a), showarrow=False, font=dict(size=20))
        fig.add_annotation(x=5, y=2, text=str(only_b), showarrow=False, font=dict(size=20))
        fig.add_annotation(x=3.15, y=2, text=str(both), showarrow=False, font=dict(size=20))

        fig.add_annotation(x=1.3, y=4.4, text=f"<b>{label_a}</b>", showarrow=False)
        fig.add_annotation(x=5, y=4.4, text=f"<b>{label_b}</b>", showarrow=False)

        fig.update_xaxes(visible=False, range=[-1, 7.3])
        fig.update_yaxes(visible=False, range=[-1, 5], scaleanchor="x")
        fig.update_layout(title=title, plot_bgcolor="white")

        return fig

    def paired_dot_plot(self, id_col, time_col, value_col, palette=None,
                        title="", x_label="", y_label=""):

        fig = go.Figure()
        palette = palette or ["#999999"]

        for subject_id in self.df[id_col].unique():
            subset = self.df[self.df[id_col] == subject_id].sort_values(time_col)
            fig.add_trace(go.Scatter(
                x=subset[time_col], y=subset[value_col], mode="lines+markers",
                line=dict(color=palette[0], width=1), marker=dict(size=8, color=palette[0]),
                opacity=0.6, showlegend=False
            ))

        fig.update_layout(
            title=title, xaxis_title=x_label or time_col, yaxis_title=y_label or value_col
        )

        return fig

    @staticmethod
    def _axis_is_numeric(fig, axis):
        """
        True only if every trace's data on this axis is numeric, so a
        log scale makes sense. Text categories, dates and heatmap labels
        return False (forcing a log/linear type on those makes the plot vanish).
        """
        found = False
        for trace in fig.data:
            values = getattr(trace, axis, None)
            if values is None:
                continue
            try:
                arr = np.asarray(values)
            except Exception:
                return False
            if arr.size == 0:
                continue
            if not np.issubdtype(arr.dtype, np.number):
                return False
            found = True
        return found

    @staticmethod
    def _axis_positive(fig, axis):
        """A log scale needs every value > 0; otherwise Plotly silently drops points."""
        for trace in fig.data:
            values = getattr(trace, axis, None)
            if values is None:
                continue
            arr = np.asarray(values, dtype=float)
            arr = arr[~np.isnan(arr)]
            if arr.size and arr.min() <= 0:
                return False
        return True

    def customize_layout(self, fig, x_axis_title=None, y_axis_title=None,
                         x_tick_font_size=12, y_tick_font_size=12,
                         axis_title_font_size=16, axis_title_bold=False,
                         axis_title_color="black", tick_color=None,
                         legend_title=None, legend_position="right", show_legend=True,
                         x_range=None, y_range=None, width=None, height=None,
                         x_scale="linear", y_scale="linear",
                         title_text=None, title_bold=True, title_font_size=20,
                         title_position="center",
                         margin_l=80, margin_r=40, margin_t=60, margin_b=80):

        legend_positions = {
            "right": dict(x=1.02, y=1, xanchor="left"),
            "top": dict(x=0.5, y=1.1, xanchor="center", orientation="h"),
            "bottom": dict(x=0.5, y=-0.2, xanchor="center", orientation="h"),
        }
        title_x_map = {"left": 0.02, "center": 0.5, "right": 0.98}

        fig.update_layout(
            showlegend=show_legend,
            legend=dict(title=legend_title, **legend_positions.get(legend_position, {})),
            margin=dict(l=margin_l, r=margin_r, t=margin_t, b=margin_b)
        )

        if title_text is not None:
            title_display = f"<b>{title_text}</b>" if title_bold else title_text
            fig.update_layout(
                title=dict(
                    text=title_display, font=dict(size=title_font_size),
                    x=title_x_map.get(title_position, 0.5)
                )
            )

        # Tick fonts always apply, so the tick-size sliders work even
        # when no axis title was typed.
        tick_font_x = dict(size=x_tick_font_size)
        tick_font_y = dict(size=y_tick_font_size)
        if tick_color:
            tick_font_x["color"] = tick_color
            tick_font_y["color"] = tick_color
        fig.update_xaxes(tickfont=tick_font_x)
        fig.update_yaxes(tickfont=tick_font_y)

        if x_axis_title:
            x_title_display = f"<b>{x_axis_title}</b>" if axis_title_bold else x_axis_title
            fig.update_xaxes(
                title_text=x_title_display,
                title_font=dict(size=axis_title_font_size, color=axis_title_color)
            )
        if y_axis_title:
            y_title_display = f"<b>{y_axis_title}</b>" if axis_title_bold else y_axis_title
            fig.update_yaxes(
                title_text=y_title_display,
                title_font=dict(size=axis_title_font_size, color=axis_title_color)
            )

        # The axis type is touched ONLY when the user asks for a log scale
        # AND the axis is numeric. Otherwise Plotly keeps its own
        # auto-detected type (category / date / linear).
        if x_scale == "log" and self._axis_is_numeric(fig, "x") and self._axis_positive(fig, "x"):
            fig.update_xaxes(type="log")
        if y_scale == "log" and self._axis_is_numeric(fig, "y") and self._axis_positive(fig, "y"):
            fig.update_yaxes(type="log")

        if x_range:
            fig.update_xaxes(range=x_range)
        if y_range:
            fig.update_yaxes(range=y_range)
        if width:
            fig.update_layout(width=width)
        if height:
            fig.update_layout(height=height)

        return fig    