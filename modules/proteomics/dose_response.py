"""
Dose-response analysis: how much of a compound is needed for an effect?

Model (four-parameter logistic, the field standard):
    response = Bottom + (Top - Bottom) / (1 + (conc / IC50) ** Hill)

  Bottom / Top  the lowest / highest plateau of the curve
  IC50 (EC50)   concentration giving a response half-way between Bottom and Top
  Hill          steepness (1 = classic one-site binding)

The fit uses log10(IC50) internally (much more stable), and the 95% confidence
interval comes from the fit's covariance with a t-distribution.
Concentration 0 (untreated control) cannot be plotted on a log scale, so it is
used only to normalise (optional) and is left out of the fit.
"""

import numpy as np
import pandas as pd
from scipy import optimize, stats


def four_pl(x, bottom, top, log_ic50, hill):
    return bottom + (top - bottom) / (1.0 + (x / 10.0 ** log_ic50) ** hill)


def normalise_to_control(df, conc_col, resp_col, group_col=None):
    """Express responses as % of the mean of the untreated (concentration 0) wells, per group."""
    out = df.copy()
    keys = [group_col] if group_col else [None]
    for key in keys:
        mask = out[group_col] == key if key is not None else pd.Series(True, index=out.index)
        control = out.loc[mask & (out[conc_col] == 0), resp_col].mean()
        if not np.isfinite(control) or control == 0:
            raise ValueError("Normalising needs untreated control wells (concentration = 0) with a non-zero mean.")
        out.loc[mask, resp_col] = out.loc[mask, resp_col] / control * 100.0
    return out


def fit_four_parameter(conc, resp):
    """Returns a dict with the fitted parameters, 95% CIs, R^2 and plain-language warnings."""
    conc = np.asarray(conc, dtype=float)
    resp = np.asarray(resp, dtype=float)
    keep = np.isfinite(conc) & np.isfinite(resp) & (conc > 0)
    conc, resp = conc[keep], resp[keep]

    n_levels = len(np.unique(conc))
    if n_levels < 4:
        raise ValueError(f"Only {n_levels} distinct concentration(s) above zero; at least 4 are needed to fit a curve (6-8 is better).")
    if len(conc) < 5:
        raise ValueError("Too few data points for a 4-parameter fit (need at least 5).")
    if np.ptp(resp) == 0:
        raise ValueError("All responses are identical; there is no curve to fit.")

    # starting values: plateaus from the two ends, IC50 where the response crosses the middle
    order = np.argsort(conc)
    xs, ys = conc[order], resp[order]
    quarter = max(2, len(ys) // 4)
    low_end, high_end = ys[:quarter].mean(), ys[-quarter:].mean()
    crossing = np.argmin(np.abs(ys - (low_end + high_end) / 2))
    start = [low_end, high_end, np.log10(xs[crossing]), 1.0]      # Bottom, Top, logIC50, Hill

    span = np.ptp(resp)
    lower = [resp.min() - span, resp.min() - span, np.log10(conc.min()) - 3, 0.1]
    upper = [resp.max() + span, resp.max() + span, np.log10(conc.max()) + 3, 10.0]

    best = None
    for hill0 in (1.0, 0.5, 2.0):
        for plateaus in ((start[0], start[1]), (start[1], start[0])):
            p0 = [plateaus[0], plateaus[1], start[2], hill0]
            p0 = np.clip(p0, np.array(lower) + 1e-9, np.array(upper) - 1e-9)
            try:
                params, cov = optimize.curve_fit(four_pl, conc, resp, p0=p0, bounds=(lower, upper), maxfev=20000)
            except (RuntimeError, ValueError):
                continue
            sse = float(np.sum((resp - four_pl(conc, *params)) ** 2))
            if best is None or sse < best[0]:
                best = (sse, params, cov)
    if best is None:
        raise ValueError("The curve could not be fitted to these data (no stable solution). Check the values and concentration units.")

    sse, params, cov = best
    bottom, top, log_ic50, hill = params
    dof = max(len(conc) - 4, 1)
    se = np.sqrt(np.diag(cov)) if np.all(np.isfinite(cov)) else np.full(4, np.nan)
    t = stats.t.ppf(0.975, dof)
    ss_total = float(np.sum((resp - resp.mean()) ** 2))
    r_squared = 1 - sse / ss_total if ss_total > 0 else np.nan

    warnings = []
    ic50 = 10.0 ** log_ic50
    if ic50 < conc.min():
        warnings.append("IC50 is below the lowest tested concentration: extrapolated, add lower doses.")
    if ic50 > conc.max():
        warnings.append("IC50 is above the highest tested concentration: extrapolated, add higher doses.")
    if not np.isfinite(se[2]):
        warnings.append("The confidence interval could not be computed (the data do not define the curve well).")
    if dof < 3:
        warnings.append("Very few degrees of freedom: treat the confidence interval with caution.")
    if r_squared < 0.9:
        warnings.append(f"Poor fit (R² = {r_squared:.2f}); check outliers, the concentration units and whether the response is dose-dependent.")

    return {
        "IC50": ic50,
        "IC50_low": 10.0 ** (log_ic50 - t * se[2]) if np.isfinite(se[2]) else np.nan,
        "IC50_high": 10.0 ** (log_ic50 + t * se[2]) if np.isfinite(se[2]) else np.nan,
        "Hill": hill, "Hill_se": se[3], "Bottom": bottom, "Top": top,
        "R_squared": r_squared, "n_points": int(len(conc)), "n_concentrations": int(n_levels),
        "warnings": warnings, "params": params,
    }


def fit_table(df, conc_col, resp_col, group_col=None):
    """Fit every compound/group separately. Returns (summary table, {group: fit dict}, {group: error text})."""
    fits, errors, rows = {}, {}, []
    groups = [(None, df)] if group_col is None else list(df.groupby(group_col, sort=False))
    for name, sub in groups:
        label = "All data" if name is None else str(name)
        try:
            fit = fit_four_parameter(sub[conc_col], sub[resp_col])
            fits[label] = fit
            rows.append({"Compound": label, "IC50": fit["IC50"], "IC50_95%_low": fit["IC50_low"],
                         "IC50_95%_high": fit["IC50_high"], "Hill": fit["Hill"], "Bottom": fit["Bottom"],
                         "Top": fit["Top"], "R_squared": fit["R_squared"], "Points": fit["n_points"],
                         "Warnings": " | ".join(fit["warnings"])})
        except ValueError as error:
            errors[label] = str(error)
    return pd.DataFrame(rows), fits, errors


def dose_response_figure(df, conc_col, resp_col, fits, group_col=None, unit="µM", y_label="Response"):
    import plotly.graph_objects as go

    fig = go.Figure()
    positive = df[df[conc_col] > 0]
    groups = [(None, positive)] if group_col is None else list(positive.groupby(group_col, sort=False))
    for name, sub in groups:
        label = "All data" if name is None else str(name)
        fig.add_trace(go.Scatter(x=sub[conc_col], y=sub[resp_col], mode="markers", name=f"{label} (data)",
                                 marker=dict(size=8), legendgroup=label))
        if label in fits:
            fit = fits[label]
            xs = np.logspace(np.log10(sub[conc_col].min()) - 0.3, np.log10(sub[conc_col].max()) + 0.3, 200)
            fig.add_trace(go.Scatter(x=xs, y=four_pl(xs, *fit["params"]), mode="lines",
                                     name=f"{label}: IC50 = {fit['IC50']:.3g} {unit}", legendgroup=label))
            fig.add_vline(x=fit["IC50"], line_dash="dot", line_color="grey", opacity=0.5)
    fig.update_layout(xaxis=dict(type="log", title=f"Concentration ({unit})"), yaxis_title=y_label,
                      height=480, plot_bgcolor="white", margin=dict(l=10, r=10, t=30, b=10))
    return fig

# END OF FILE: modules/proteomics/dose_response.py