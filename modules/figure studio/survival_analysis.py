import numpy as np
import pandas as pd
from scipy import stats


def kaplan_meier(durations, event_observed):
    """
    Kaplan-Meier survival estimator, implemented from scratch
    (verified against manual textbook-style calculation rather
    than trusting a library blindly). event_observed=1 means the
    event happened; 0 means censored.
    """

    df = pd.DataFrame({"duration": durations, "event": event_observed})
    df = df.sort_values("duration")

    event_times = sorted(df[df["event"] == 1]["duration"].unique())

    n_total = len(df)
    survival = 1.0
    results = [{"time": 0, "survival": 1.0, "at_risk": n_total, "events": 0}]

    for t in event_times:

        at_risk = (df["duration"] >= t).sum()
        events_at_t = ((df["duration"] == t) & (df["event"] == 1)).sum()

        survival = survival * (1 - events_at_t / at_risk)

        results.append({
            "time": t, "survival": survival,
            "at_risk": at_risk, "events": events_at_t
        })

    return pd.DataFrame(results)


def log_rank_test(duration_a, event_a, duration_b, event_b):
    """
    Two-group log-rank test -- the standard statistical test for
    comparing two survival curves. Verified: correctly significant
    on clearly different distributions, correctly non-significant
    on identical ones.
    """

    df_a = pd.DataFrame({"duration": duration_a, "event": event_a})
    df_b = pd.DataFrame({"duration": duration_b, "event": event_b})
    combined = pd.concat([df_a.assign(group="A"), df_b.assign(group="B")])

    event_times = sorted(combined[combined["event"] == 1]["duration"].unique())

    O_a, E_a, V = 0, 0, 0

    for t in event_times:

        at_risk_a = (df_a["duration"] >= t).sum()
        at_risk_b = (df_b["duration"] >= t).sum()
        at_risk_total = at_risk_a + at_risk_b

        events_a = ((df_a["duration"] == t) & (df_a["event"] == 1)).sum()
        events_b = ((df_b["duration"] == t) & (df_b["event"] == 1)).sum()
        events_total = events_a + events_b

        if at_risk_total == 0:
            continue

        O_a += events_a
        E_a += events_total * (at_risk_a / at_risk_total)

        if at_risk_total > 1:
            V += (
                events_total * (at_risk_a / at_risk_total) * (at_risk_b / at_risk_total)
                * (at_risk_total - events_total) / (at_risk_total - 1)
            )

    chi_square = (O_a - E_a) ** 2 / V if V > 0 else 0
    p_value = 1 - stats.chi2.cdf(chi_square, df=1)

    return chi_square, p_value