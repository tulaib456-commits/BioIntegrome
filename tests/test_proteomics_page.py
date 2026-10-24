"""Drives the real Proteomics page: upload -> missing values -> normalization -> DE -> ML."""
import numpy as np
import pandas as pd
import pytest

from tests.page_helpers import PageDriver, install_fake_login
from tests.sim_data import make_proteomics

NO_IMPUTE = "Use the values each protein has (no imputation)"
PERSEUS = "Perseus-style imputation (replace with low values)"


def csv_of(df):
    return df.to_csv(index=False).encode()


@pytest.fixture(scope="module")
def prot_csv():
    df, groups, truth = make_proteomics(n_prot=500, n_per_group=6, n_de=50, missing_rate=0.08, seed=5)
    return csv_of(df)


def make_driver(data, name, tmp_path, monkeypatch):
    install_fake_login()
    monkeypatch.chdir(tmp_path)
    d = PageDriver("2_Proteomics.py", {"Upload dataset": (data, name)})
    d.run()
    return d


def ok(d):
    assert not d.exceptions(), d.exceptions()


def ready_to_normalize(d, method):
    d.assign_groups("proteomics_mv", {"A_": "A", "B_": "B"})
    d.set_radio("How should the remaining missing values be handled?", method)
    d.click("Filter and Handle Missing Values"); ok(d)
    assert "proteins retained" in d.text(), d.text()


@pytest.mark.parametrize("method", [NO_IMPUTE, PERSEUS])
def test_full_flow_to_differential_expression(prot_csv, tmp_path, monkeypatch, method):
    d = make_driver(prot_csv, "prot.csv", tmp_path, monkeypatch); ok(d)
    assert "Uploaded: 500 proteins, 12 samples" in d.text()
    ready_to_normalize(d, method)
    d.click("Apply Normalization"); ok(d)
    samples = list(d.at.multiselect(key="prot_group_a").options)
    d.set_ms_key("prot_group_a", [s for s in samples if s.startswith("A_")]); ok(d)
    d.set_ms_key("prot_group_b", [s for s in samples if s.startswith("B_")]); ok(d)
    assert "significant proteins" in d.text()


def test_quantile_normalization_with_missing_values(prot_csv, tmp_path, monkeypatch):
    d = make_driver(prot_csv, "prot.csv", tmp_path, monkeypatch)
    ready_to_normalize(d, NO_IMPUTE)
    d.set_selectbox("Normalization Method", "Quantile Normalization")
    d.click("Apply Normalization"); ok(d)


def test_half_way_through_choosing_groups_never_crashes(prot_csv, tmp_path, monkeypatch):
    d = make_driver(prot_csv, "prot.csv", tmp_path, monkeypatch)
    ready_to_normalize(d, PERSEUS)
    d.click("Apply Normalization")
    samples = list(d.at.multiselect(key="prot_group_a").options)
    d.set_ms_key("prot_group_a", samples[:1]); ok(d)
    d.set_ms_key("prot_group_b", [s for s in samples if s.startswith("B_")]); ok(d)
    assert "at least 2" in d.text()


def test_anova_path(prot_csv, tmp_path, monkeypatch):
    d = make_driver(prot_csv, "prot.csv", tmp_path, monkeypatch)
    ready_to_normalize(d, PERSEUS)
    d.click("Apply Normalization")
    d.set_radio_key("prot_de_mode", "Multi-Group (ANOVA, 3+ conditions)"); ok(d)
    d.assign_groups("prot_anova", {"A_": "A", "B_": "B"})
    d.click_key("prot_anova_run"); ok(d)


def test_machine_learning_cross_validation_runs_without_pre_selection(prot_csv, tmp_path, monkeypatch):
    d = make_driver(prot_csv, "prot.csv", tmp_path, monkeypatch)
    ready_to_normalize(d, PERSEUS)
    d.click("Apply Normalization")
    samples = list(d.at.multiselect(key="prot_group_a").options)
    d.set_ms_key("prot_group_a", [s for s in samples if s.startswith("A_")])
    d.set_ms_key("prot_group_b", [s for s in samples if s.startswith("B_")]); ok(d)
    d.click_key("prot_ml_run"); ok(d)
    assert "Cross-validation complete" in d.text()
    d.click_key("prot_shap_btn"); ok(d)


def test_two_samples_per_group_does_not_crash_the_fold_slider(tmp_path, monkeypatch):
    df, groups, truth = make_proteomics(n_prot=300, n_per_group=2, n_de=30, missing_rate=0.0, seed=2)
    d = make_driver(csv_of(df), "tiny.csv", tmp_path, monkeypatch)
    d.assign_groups("proteomics_mv", {"A_": "A", "B_": "B"})
    d.set_radio("How should the remaining missing values be handled?", NO_IMPUTE)
    d.click("Filter and Handle Missing Values"); ok(d)
    d.click("Apply Normalization")
    samples = list(d.at.multiselect(key="prot_group_a").options)
    d.set_ms_key("prot_group_a", [s for s in samples if s.startswith("A_")])
    d.set_ms_key("prot_group_b", [s for s in samples if s.startswith("B_")]); ok(d)
    assert "only 2 samples" in d.text()


# ------------------------------------------------------------------ messy uploads
def test_semicolon_decimal_comma_file_is_read(tmp_path, monkeypatch):
    df, _, _ = make_proteomics(n_prot=40, n_per_group=3, n_de=5, missing_rate=0.0)
    text = df.round(3).to_csv(index=False, sep=";", decimal=",")
    d = make_driver(text.encode(), "euro.csv", tmp_path, monkeypatch); ok(d)
    assert "Uploaded: 40 proteins, 6 samples" in d.text()


def test_file_with_too_few_sample_columns_gets_clear_message(tmp_path, monkeypatch):
    df = pd.DataFrame({"Gene": ["a", "b"], "S1": [1.0, 2.0], "S2": [2.0, 3.0]})
    d = make_driver(csv_of(df), "few.csv", tmp_path, monkeypatch); ok(d)
    assert "at least 4 numeric sample columns" in d.text()


def test_text_annotation_column_next_to_samples_is_not_treated_as_a_sample(tmp_path, monkeypatch):
    df, _, _ = make_proteomics(n_prot=60, n_per_group=3, n_de=5, missing_rate=0.0)
    df["Description"] = "some protein"
    d = make_driver(csv_of(df), "annot.csv", tmp_path, monkeypatch); ok(d)
    assert "6 samples" in d.text()
