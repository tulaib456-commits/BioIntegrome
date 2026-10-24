"""
Dataset-zoo tests: messy real-world files and every plot type.
'No crash' is the minimum; most tests also check the numbers drawn.
"""
import io
import os
import time

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from modules.figure_studio.data_loader import load_file, list_sheets, clean_dataframe, detect_column_types
from modules.figure_studio.data_prep import (
    suggest_layout, wide_to_long, code_like_columns, as_categories, paired_arrays
)
from modules.figure_studio.plot_builder import FigureBuilder
from modules.figure_studio.themes import PALETTES, THEMES, apply_theme
from modules.figure_studio.export_component import build_export_html

HERE = os.path.dirname(__file__)
HEMO = os.path.join(HERE, "data", "Hemoglobin.xlsx")


class Upload(io.BytesIO):
    """Behaves like Streamlit's uploaded file object."""
    def __init__(self, data, name):
        super().__init__(data if isinstance(data, bytes) else data.encode("utf-8"))
        self.name = name


def full_pipeline(fig, **layout):
    """Everything the page does to a figure after building it."""
    fig = apply_theme(fig, list(THEMES)[0], "Arial", 14)
    fig = FigureBuilder(None).customize_layout(fig, **layout)
    html, height = build_export_html(fig, "t", 1200, 800, 3)
    assert "<script>" in html and height > 0
    return fig


# ================================================================ the user's real file
def test_hemoglobin_file_loads_and_is_recognised_as_wide():
    with open(HEMO, "rb") as f:
        df = load_file(Upload(f.read(), "Hemoglobin.xlsx"))
    assert df.shape == (50, 3)
    assert suggest_layout(df) == "wide"


def test_hemoglobin_wide_to_long_keeps_every_pair():
    with open(HEMO, "rb") as f:
        df = load_file(Upload(f.read(), "Hemoglobin.xlsx"))
    long, g, v = wide_to_long(df, ["Before", "After"], id_col="ID")
    assert len(long) == 100 and set(long[g]) == {"Before", "After"}
    a, b, dropped = paired_arrays(long, "ID", g, v, "Before", "After")
    assert dropped == 0 and len(a) == 50
    assert np.allclose(a, df["Before"]) and np.allclose(b, df["After"])


def test_hemoglobin_paired_test_equals_scipy_on_the_original_columns():
    from modules.figure_studio.statistics_toolkit import compare_two_groups
    with open(HEMO, "rb") as f:
        df = load_file(Upload(f.read(), "Hemoglobin.xlsx"))
    r = compare_two_groups(df["Before"], df["After"], paired=True, force_method="Paired t-test")
    t, p = stats.ttest_rel(df["Before"], df["After"])
    assert r["statistic"] == pytest.approx(t) and r["p_value"] == pytest.approx(p)


# ================================================================ messy file formats
def test_semicolon_csv_with_decimal_commas():
    text = "Group;Value\nA;1,5\nA;2,5\nB;3,5\nB;4,5\n"
    df = load_file(Upload(text, "x.csv"))
    assert df["Value"].tolist() == [1.5, 2.5, 3.5, 4.5]
    assert detect_column_types(df)[0] == ["Value"]


def test_tab_separated_txt():
    df = load_file(Upload("a\tb\n1\t2\n3\t4\n", "x.txt"))
    assert df.shape == (2, 2)


def test_utf8_bom_and_latin1_encodings():
    df1 = load_file(Upload("\ufeffGroup,Value\nA,1\nB,2\n".encode("utf-8"), "x.csv"))
    assert list(df1.columns) == ["Group", "Value"]
    df2 = load_file(Upload("Gruppe,Wert\nÄ,1\nÖ,2\n".encode("latin-1"), "y.csv"))
    assert df2["Gruppe"].tolist() == ["Ä", "Ö"]


def test_numbers_stored_as_text_are_converted_but_categories_and_codes_are_not():
    text = "Code,Group,Conc\n007,Control,\"1,234\"\n008,Control,\"2,345\"\n009,Drug,\"3,456\"\n"
    df = load_file(Upload(text, "x.csv"))
    assert df["Conc"].tolist() == [1234, 2345, 3456]
    assert df["Group"].tolist() == ["Control", "Control", "Drug"]
    assert df["Code"].tolist() == ["007", "008", "009"]            # leading zeros preserved


def test_empty_rows_columns_and_padding_spaces_are_cleaned():
    text = "Group, Value ,\nControl ,1,\n ,,\nDrug,2,\n"
    df = load_file(Upload(text, "x.csv"))
    assert list(df.columns) == ["Group", "Value"] and len(df) == 2
    assert df["Group"].tolist() == ["Control", "Drug"]


def test_duplicate_column_names_are_renamed_not_overwritten():
    df, notes = clean_dataframe(pd.DataFrame([[1, 2], [3, 4]], columns=["x", "x"]))
    assert df.shape == (2, 2) and len(set(df.columns)) == 2 and notes


def test_empty_file_is_refused_with_clear_message():
    with pytest.raises(Exception):
        load_file(Upload("", "x.csv"))


def test_excel_sheet_selection(tmp_path):
    path = tmp_path / "two.xlsx"
    with pd.ExcelWriter(path) as w:
        pd.DataFrame({"a": [1, 2]}).to_excel(w, sheet_name="First", index=False)
        pd.DataFrame({"b": [3, 4, 5]}).to_excel(w, sheet_name="Second", index=False)
    up = Upload(path.read_bytes(), "two.xlsx")
    assert list_sheets(up) == ["First", "Second"]
    assert load_file(up, sheet_name="Second").shape == (3, 1)


def test_unicode_column_names_survive_the_whole_pipeline():
    df = pd.DataFrame({"Gruppe": ["a", "a", "b", "b"], "Konzentration (µM)": [1.0, 2.0, 3.0, 4.0]})
    fig = FigureBuilder(df).box_plot("Gruppe", "Konzentration (µM)")
    full_pipeline(fig)


# ================================================================ group codes
def test_numeric_group_codes_are_offered_as_categories():
    df = pd.DataFrame({"Treatment": [0, 1, 2] * 10, "Value": np.random.default_rng(0).normal(size=30),
                       "ID": range(30)})
    assert code_like_columns(df) == ["Treatment"]          # ID (unique) and Value (continuous) are not
    out = as_categories(df, ["Treatment"])
    assert out["Treatment"].tolist()[:3] == ["0", "1", "2"]
    assert detect_column_types(out)[1] == ["Treatment"]


# ================================================================ wide -> long
def test_wide_to_long_three_columns_unequal_lengths():
    df = pd.DataFrame({"Control": [1, 2, 3, np.nan], "A": [4, 5, np.nan, np.nan], "B": [7, 8, 9, 10]})
    long, g, v = wide_to_long(df, ["Control", "A", "B"])
    assert long.groupby(g)[v].count().to_dict() == {"A": 2, "B": 4, "Control": 3}


def test_wide_to_long_name_collision_is_avoided():
    df = pd.DataFrame({"Group": ["x", "y"], "Before": [1, 2], "After": [3, 4]})
    long, g, v = wide_to_long(df, ["Before", "After"], id_col="Group")
    assert g != "Group" and g in long.columns


def test_paired_alignment_survives_missing_values():
    """Row-position pairing misaligns as soon as one value is missing; ID pairing must not."""
    df = pd.DataFrame({"ID": [1, 2, 3, 4], "Pre": [10, 20, np.nan, 40.0], "Post": [11, np.nan, 31, 41.0]})
    long, g, v = wide_to_long(df, ["Pre", "Post"], id_col="ID")
    a, b, dropped = paired_arrays(long, "ID", g, v, "Pre", "Post")
    assert a.tolist() == [10, 40] and b.tolist() == [11, 41] and dropped == 2


def test_duplicate_subject_values_refused_for_paired():
    long = pd.DataFrame({"ID": [1, 1, 2, 2], "G": ["a", "a", "a", "b"], "V": [1, 2, 3, 4.0]})
    with pytest.raises(ValueError, match="more than one"):
        paired_arrays(long, "ID", "G", "V", "a", "b")


# ================================================================ plots x data
@pytest.fixture(scope="module")
def general():
    rng = np.random.default_rng(1)
    g = np.repeat(["Control", "Drug_A", "Drug_B"], 20)
    val = np.concatenate([rng.normal(10, 2, 20), rng.normal(14, 2, 20), rng.normal(10.5, 2, 20)])
    return pd.DataFrame({"Group": g, "Sex": ["M", "F"] * 30, "Expression": val,
                         "Age": rng.integers(30, 70, 60), "Weight": rng.normal(70, 10, 60)})


@pytest.mark.parametrize("summary", ["Mean ± SD", "Mean ± SEM", "Mean only", "Median", "Sum"])
def test_bar_chart_heights_equal_pandas_statistics(general, summary):
    fig = FigureBuilder(general).bar_chart("Group", "Expression", None, None, summary=summary)
    t = fig.data[0]
    grouped = general.groupby("Group")["Expression"]
    expected = {"Mean ± SD": grouped.mean(), "Mean ± SEM": grouped.mean(), "Mean only": grouped.mean(),
                "Median": grouped.median(), "Sum": grouped.sum()}[summary]
    assert np.allclose(np.asarray(t.y, dtype=float), expected.loc[list(t.x)].values)
    if summary == "Mean ± SD":
        assert np.allclose(np.asarray(t.error_y.array, dtype=float), grouped.std(ddof=1).loc[list(t.x)].values)
    if summary == "Mean ± SEM":
        sem = grouped.std(ddof=1) / np.sqrt(grouped.count())
        assert np.allclose(np.asarray(t.error_y.array, dtype=float), sem.loc[list(t.x)].values)


def test_bar_chart_with_group_column_and_with_x_equal_group(general):
    f1 = FigureBuilder(general).bar_chart("Group", "Expression", "Sex", None)
    assert {t.name for t in f1.data} == {"M", "F"}
    f2 = FigureBuilder(general).bar_chart("Group", "Expression", "Group", None)
    assert len(f2.data) == 3


@pytest.mark.parametrize("kind", ["box", "violin"])
def test_every_basic_plot_type_builds_and_exports(general, kind):
    b = FigureBuilder(general)
    figs = [
        b.box_plot("Group", "Expression", "Sex"), b.violin_plot("Group", "Expression", None),
        b.scatter_plot("Age", "Weight", "Group", trendline="ols"), b.line_chart("Age", "Weight", "Group"),
        b.histogram("Expression", "Group", None, 20), b.correlation_heatmap(["Expression", "Age", "Weight"]),
        b.grouped_comparison_with_stats("Expression", "Group", plot_type=kind),
    ]
    for fig in figs:
        assert len(fig.data) > 0
        full_pipeline(fig, width=1000, height=600)


def test_grouped_comparison_brackets_sit_over_the_right_boxes_even_if_data_order_is_not_alphabetical():
    """Old bug: brackets used sorted groups, boxes used first-appearance order."""
    rng = np.random.default_rng(3)
    df = pd.DataFrame({
        "G": ["Treated"] * 15 + ["Control"] * 15 + ["Mid"] * 15,
        "V": np.concatenate([rng.normal(20, 1, 15), rng.normal(10, 1, 15), rng.normal(10.2, 1, 15)])})
    b = FigureBuilder(df)
    fig = b.grouped_comparison_with_stats("V", "G")
    axis_order = list(fig.layout.xaxis.categoryarray)
    assert axis_order == ["Treated", "Control", "Mid"]
    tbl = b.last_pairwise_table
    # the (Treated, Control) bracket must span x = 0..1, centre 0.5, with the star text of that pair
    star = tbl[(tbl.Group_1 == "Treated") & (tbl.Group_2 == "Control")]["Stars"].iloc[0]
    centres = {round(a.x, 3): a.text for a in fig.layout.annotations}
    assert centres[0.5] == star == "***"
    # Control vs Mid (x 1..2) is NOT different -> must read ns
    assert centres[1.5] == "ns"


def test_grouped_comparison_stars_use_corrected_pvalues():
    rng = np.random.default_rng(9)
    df = pd.DataFrame({"G": np.repeat(list("ABCDE"), 8), "V": rng.normal(0, 1, 40)})
    b = FigureBuilder(df)
    b.grouped_comparison_with_stats("V", "G", test="welch", correction="holm")
    t = b.last_pairwise_table
    assert (t["Adj_P_Value"] >= t["P_Value"] - 1e-12).all()
    assert t["Correction"].iloc[0] == "Holm"


def test_grouped_comparison_group_with_one_value_is_skipped_not_crash():
    df = pd.DataFrame({"G": ["A"] * 5 + ["B"] * 5 + ["C"], "V": list(range(5)) + list(range(3, 8)) + [99]})
    b = FigureBuilder(df)
    fig = b.grouped_comparison_with_stats("V", "G")
    assert "C" not in set(b.last_pairwise_table.Group_1) | set(b.last_pairwise_table.Group_2)
    full_pipeline(fig)


def test_grouped_comparison_with_nan_values():
    df = pd.DataFrame({"G": ["A"] * 6 + ["B"] * 6, "V": [1, 2, np.nan, 4, 5, 6, 3, 4, 5, np.nan, 7, 8.0]})
    full_pipeline(FigureBuilder(df).grouped_comparison_with_stats("V", "G"))


def test_log_scale_on_negative_data_is_not_applied_instead_of_blanking_the_plot(general):
    df = general.assign(Expression=general["Expression"] - 12)         # contains negatives
    fig = FigureBuilder(df).box_plot("Group", "Expression")
    fig = FigureBuilder(None).customize_layout(fig, y_scale="log")
    assert fig.layout.yaxis.type != "log"


def test_log_scale_on_positive_data_is_applied(general):
    fig = FigureBuilder(general).scatter_plot("Age", "Weight")
    fig = FigureBuilder(None).customize_layout(fig, y_scale="log", x_scale="log")
    assert fig.layout.yaxis.type == "log" and fig.layout.xaxis.type == "log"


def test_kaplan_meier_all_censored_no_events():
    df = pd.DataFrame({"T": [5, 6, 7, 8, 9, 10], "E": [0] * 6, "G": list("aaabbb")})
    fig, lr = FigureBuilder(df).kaplan_meier_plot("T", "E", "G")
    full_pipeline(fig)
    assert lr["p_value"] == pytest.approx(1.0)


def test_kaplan_meier_three_groups_has_no_logrank_but_draws_all_curves():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"T": rng.exponential(10, 90), "E": rng.binomial(1, .8, 90), "G": np.repeat(list("abc"), 30)})
    fig, lr = FigureBuilder(df).kaplan_meier_plot("T", "E", "G")
    assert lr is None and len(fig.data) == 3


def test_forest_plot_rejects_impossible_confidence_intervals():
    df = pd.DataFrame({"S": ["a", "b"], "E": [1.0, 2.0], "L": [1.5, 1.0], "U": [2.0, 3.0]})  # a: lower > estimate
    with pytest.raises(ValueError, match="Lower"):
        FigureBuilder(df).forest_plot("S", "E", "L", "U")


def test_venn_with_empty_overlap_and_with_identical_sets():
    b = FigureBuilder(None)
    full_pipeline(b.venn_diagram_2set({"a", "b"}, {"c"}))
    full_pipeline(b.venn_diagram_2set({"a", "b"}, {"a", "b"}))


def test_paired_dot_plot_with_missing_timepoints():
    df = pd.DataFrame({"ID": [1, 1, 2, 3, 3], "T": ["Pre", "Post", "Pre", "Pre", "Post"], "V": [1, 2, 3, 4, 5.0]})
    full_pipeline(FigureBuilder(df).paired_dot_plot("ID", "T", "V"))


def test_large_dataset_builds_quickly():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"G": rng.choice(list("ABCD"), 100_000), "V": rng.normal(size=100_000)})
    t0 = time.time()
    FigureBuilder(df).bar_chart("G", "V", None, None)
    FigureBuilder(df).histogram("V", "G", None, 50)
    assert time.time() - t0 < 10


def test_export_html_survives_hostile_text_in_title_and_labels():
    df = pd.DataFrame({"G": ["a</script><script>alert(1)</script>", "b"], "V": [1.0, 2.0]})
    fig = FigureBuilder(df).bar_chart("G", "V", None, None, title="</script><b>x</b>")
    html, _ = build_export_html(fig)
    assert html.count("<script>") == 1 and html.count("</script>") == 1


def test_forest_plot_same_column_chosen_twice_gives_clear_message():
    df = pd.DataFrame({"S": ["a", "b"], "E": [1.0, 2.0]})
    with pytest.raises(ValueError, match="DIFFERENT"):
        FigureBuilder(df).forest_plot("S", "E", "E", "E")


def test_group_codes_plus_value_is_suggested_as_long_not_wide():
    df = pd.DataFrame({"Treatment": [0, 1, 2] * 10, "Value": np.random.default_rng(0).normal(size=30)})
    assert suggest_layout(df) == "long"
