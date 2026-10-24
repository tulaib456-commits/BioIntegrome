"""
Runs the real Figure Studio page headlessly (Streamlit's AppTest) with uploaded
test files and drives its widgets by key. Catches wiring errors that unit tests
of the calculation code cannot: wrong default column, crash on empty dropdown, etc.
"""
import io
import os
import sys
import types

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAGE = os.path.join(ROOT, "pages", "3_Figure_Studio.py")
HEMO = os.path.join(ROOT, "tests", "data", "Hemoglobin.xlsx")


class Upload(io.BytesIO):
    def __init__(self, data, name):
        super().__init__(data if isinstance(data, bytes) else data.encode("utf-8"))
        self.name = name


@pytest.fixture(autouse=True)
def fake_login(monkeypatch):
    """The page needs a login; tests replace it with a no-op (real login is Supabase)."""
    fake = types.ModuleType("modules.auth")
    fake.require_login = lambda: None
    fake.require_active_plan = lambda: None
    monkeypatch.setitem(sys.modules, "modules.auth", fake)


def run_page(file_bytes, name):
    import streamlit as st
    real = st.file_uploader
    st.file_uploader = lambda *a, **k: Upload(file_bytes, name)
    try:
        at = AppTest.from_file(PAGE, default_timeout=60)
        at.run()
    finally:
        st.file_uploader = real
    return at


def run_again(at, file_bytes, name):
    import streamlit as st
    real = st.file_uploader
    st.file_uploader = lambda *a, **k: Upload(file_bytes, name)
    try:
        at.run()
    finally:
        st.file_uploader = real
    return at


def csv_bytes(df):
    return df.to_csv(index=False).encode()


def no_crash(at):
    assert not at.exception, [e.value for e in at.exception]


def texts(at):
    return " ".join([str(e.value) for e in at.error] + [str(e.value) for e in at.warning] +
                    [str(e.value) for e in at.info] + [str(e.value) for e in at.markdown] +
                    [str(e.value) for e in at.caption])


def set_select(at, key, value):
    at.selectbox(key=key).set_value(value)


# ------------------------------------------------------------------ the user's file
def test_hemoglobin_opens_as_wide_and_every_plot_type_survives():
    data = open(HEMO, "rb").read()
    at = run_page(data, "Hemoglobin.xlsx")
    no_crash(at)
    assert "Reshaped to long format" in texts(at)
    plot_select = [s for s in at.selectbox if s.label == "Plot Type"][0]
    for plot in plot_select.options:
        plot_select.set_value(plot)
        at = run_again(at, data, "Hemoglobin.xlsx")
        plot_select = [s for s in at.selectbox if s.label == "Plot Type"][0]
        no_crash(at)


def test_hemoglobin_paired_test_gives_scipy_answer():
    from scipy import stats
    df = pd.read_excel(HEMO)
    t, p = stats.ttest_rel(df.Before, df.After)
    data = open(HEMO, "rb").read()
    at = run_page(data, "Hemoglobin.xlsx")
    next(b for b in at.button if b.key == "stat_run2").click()
    at = run_again(at, data, "Hemoglobin.xlsx")
    no_crash(at)
    body = texts(at)
    assert "Paired t-test" in body and f"{p:.6f}" in body and f"{t:.4f}" in body


def test_stats_box_on_data_without_any_text_column_gives_guidance_not_crash():
    df = pd.DataFrame({"a": np.arange(20.0), "b": np.arange(20.0) ** 2})
    at = run_page(csv_bytes(df), "numbers_only.csv")
    no_crash(at)


# ------------------------------------------------------------------ plot types with their own inputs
def test_kaplan_meier_flow_shows_logrank():
    rng = np.random.default_rng(0)
    n = 40
    df = pd.DataFrame({"Time": np.concatenate([rng.exponential(10, n), rng.exponential(25, n)]).round(1),
                       "Event": rng.binomial(1, .8, 2 * n), "Arm": ["A"] * n + ["B"] * n})
    data = csv_bytes(df)
    at = run_page(data, "surv.csv")
    set_select(at, "km_dur", "Time") if False else None
    plot = [s for s in at.selectbox if s.label == "Plot Type"][0]
    plot.set_value("Kaplan-Meier Survival Curve"); at = run_again(at, data, "surv.csv")
    at.selectbox(key="km_dur").set_value("Time")
    at.selectbox(key="km_evt").set_value("Event")
    at.selectbox(key="km_grp").set_value("Arm")
    at = run_again(at, data, "surv.csv")
    no_crash(at)
    assert "Log-rank test" in texts(at)


def test_kaplan_meier_with_non_binary_event_column_gives_clear_message():
    df = pd.DataFrame({"Time": [1.0, 2.5, 3.1, 4.7], "Event": [1, 2, 3, 4], "G": list("aabb")})
    data = csv_bytes(df)
    at = run_page(data, "s.csv")
    [s for s in at.selectbox if s.label == "Plot Type"][0].set_value("Kaplan-Meier Survival Curve")
    at = run_again(at, data, "s.csv")
    at.selectbox(key="km_evt").set_value("Event")
    at = run_again(at, data, "s.csv")
    no_crash(at)
    assert "0 and 1" in texts(at)


def test_forest_plot_flow_and_same_column_message():
    df = pd.DataFrame({"Study": list("ABCD"), "HR": [.8, 1.2, .6, 1.0], "Lower": [.6, .9, .4, .7], "Upper": [1.1, 1.6, .9, 1.4]})
    data = csv_bytes(df)
    at = run_page(data, "f.csv")
    [s for s in at.selectbox if s.label == "Plot Type"][0].set_value("Forest Plot")
    at = run_again(at, data, "f.csv")
    # default: estimate/lower/upper all = first numeric column -> must be a clear message
    no_crash(at)
    assert "DIFFERENT" in texts(at)
    at.selectbox(key="fo_est").set_value("HR"); at.selectbox(key="fo_lo").set_value("Lower"); at.selectbox(key="fo_hi").set_value("Upper")
    at = run_again(at, data, "f.csv")
    no_crash(at)
    assert "DIFFERENT" not in texts(at)


def test_venn_flow_reports_overlap_counts():
    a = [f"G{i}" for i in range(1, 41)]
    b = [f"G{i}" for i in range(26, 66)]
    df = pd.DataFrame({"ListA": a, "ListB": b})
    data = csv_bytes(df)
    at = run_page(data, "v.csv")
    [s for s in at.selectbox if s.label == "Plot Type"][0].set_value("Venn Diagram (2-set)")
    at = run_again(at, data, "v.csv")
    at.selectbox(key="ve_b").set_value("ListB")
    at = run_again(at, data, "v.csv")
    no_crash(at)
    assert "Shared: 15" in texts(at)


def test_grouped_comparison_all_test_and_correction_combinations():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"G": np.repeat(["Control", "A", "B", "C"], 10), "V": rng.normal(10, 2, 40)})
    data = csv_bytes(df)
    at = run_page(data, "g.csv")
    [s for s in at.selectbox if s.label == "Plot Type"][0].set_value("Grouped Comparison (with significance stats)")
    at = run_again(at, data, "g.csv")
    for test in at.selectbox(key="gc_test").options:
        for corr in at.selectbox(key="gc_corr").options:
            at.selectbox(key="gc_test").set_value(test)
            if not test.startswith("Tukey"):          # correction box is disabled for Tukey (built-in)
                at.selectbox(key="gc_corr").set_value(corr)
            at = run_again(at, data, "g.csv")
            no_crash(at)
            assert at.dataframe, f"no statistics table for {test} / {corr}"


def test_numeric_group_codes_can_be_used_as_groups():
    rng = np.random.default_rng(2)
    df = pd.DataFrame({"Treatment": np.repeat([0, 1, 2], 10), "Value": rng.normal(size=30)})
    data = csv_bytes(df)
    at = run_page(data, "c.csv")
    no_crash(at)
    at.multiselect(key="fs_codes").set_value(["Treatment"])
    at = run_again(at, data, "c.csv")
    no_crash(at)
    [s for s in at.selectbox if s.label == "Plot Type"][0].set_value("Box Plot")
    at = run_again(at, data, "c.csv")
    no_crash(at)
    assert "Treatment" in at.selectbox(key="box_x").options


def test_messy_files_never_crash_the_page():
    cases = {
        "semicolon_decimal_comma.csv": "Group;Value\nA;1,5\nA;2,5\nB;3,5\nB;4,5\n",
        "one_group_only.csv": "Group,Value\nA,1\nA,2\nA,3\n",
        "single_column.csv": "Value\n1\n2\n3\n",
        "all_text.csv": "a,b\nx,y\nz,w\n",
        "with_nan.csv": "Group,Value\nA,1\nA,\nB,3\nB,4\n",
    }
    for name, text in cases.items():
        at = run_page(text.encode(), name)
        assert not at.exception, (name, [e.value for e in at.exception])
