"""Drives the real Transcriptomics page tab by tab with simulated RNA-seq data."""
import os

import numpy as np
import pytest

from tests.page_helpers import PageDriver, install_fake_login
from tests.sim_data import make_rnaseq


@pytest.fixture(scope="module")
def counts_csv():
    df, groups, de, _ = make_rnaseq(n_genes=900, n_per_group=5, n_de=60, fold=4.0, seed=3)
    return df.to_csv(index=False).encode()


@pytest.fixture()
def driver(counts_csv, tmp_path, monkeypatch):
    install_fake_login()
    monkeypatch.chdir(tmp_path)           # uploads/ and models/ folders are created here
    d = PageDriver("1_Transcriptomics.py", {"RNA Seq Count Matrix": (counts_csv, "counts.csv")})
    d.run()
    return d


def ok(d):
    assert not d.exceptions(), d.exceptions()


def test_upload_and_all_tabs_render(driver):
    ok(driver)
    assert "Upload Successful" in driver.text()


def test_qc_buttons(driver):
    driver.click("Compute Sample Correlation"); ok(driver)
    driver.click("Show Expression Distribution"); ok(driver)


@pytest.mark.parametrize("method", ["CPM", "Log2 CPM", "Median-of-Ratios (DESeq2-style, recommended)"])
def test_filter_then_each_normalization(driver, method):
    driver.click("Apply Filter"); ok(driver)
    driver.set_selectbox("Normalization Method", method)
    driver.click("Apply Normalization"); ok(driver)
    driver.click("Show RLE Diagnostic Plot"); ok(driver)


def test_raw_count_filter_path(driver):
    driver.set_radio("Choose filtering approach", "Raw count threshold (simple)"); ok(driver)
    driver.click("Apply Filter"); ok(driver)


def test_two_group_de_with_proper_groups_then_degenerate_selections(driver):
    driver.click("Apply Filter")
    samples = [s for s in driver.at.multiselect[0].options]
    a = [s for s in samples if s.startswith("Ctrl")]
    b = [s for s in samples if s.startswith("Treat")]
    driver.set_multiselect("Group A (baseline)", a); ok(driver)
    driver.set_multiselect("Group B (comparison)", b); ok(driver)
    assert "significant genes found" in driver.text()
    # the dangerous moments: user is half-way through choosing samples
    driver.set_multiselect("Group A (baseline)", a[:1]); ok(driver)
    assert "at least 2" in driver.text()
    driver.set_multiselect("Group A (baseline)", []); ok(driver)


def test_anova_button(driver):
    driver.click("Apply Filter")
    driver.set_radio("Comparison Type", "Multi-Group (ANOVA, 3+ conditions)"); ok(driver)
    assert any(b.label == "Run Multi-Group ANOVA" for b in driver.at.button), driver.text()
    driver.click("Run Multi-Group ANOVA"); ok(driver)
    assert "significant genes out of" in driver.text()


def test_machine_learning_cross_validation_is_honest_and_page_has_no_leaky_option(driver):
    driver.click("Apply Filter")
    samples = [s for s in driver.at.multiselect[0].options]
    a = [s for s in samples if s.startswith("Ctrl")]
    b = [s for s in samples if s.startswith("Treat")]
    driver.set_multiselect("Group A (baseline)", a)
    driver.set_multiselect("Group B (comparison)", b); ok(driver)
    assert "Use significant genes from DE/ANOVA" not in str([r.options for r in driver.at.radio])
    driver.click("Run Cross-Validation"); ok(driver)
    assert "Cross-validation complete" in driver.text()


def test_pdf_report_button(driver):
    driver.click("Apply Filter")
    labels = [b.label for b in driver.at.button]
    assert "Generate PDF Report" in labels
    driver.click("Generate PDF Report"); ok(driver)


def test_two_samples_per_group_does_not_crash_the_fold_slider(tmp_path, monkeypatch):
    df, groups, de, _ = make_rnaseq(n_genes=400, n_per_group=2, n_de=40, fold=4.0, seed=9)
    install_fake_login(); monkeypatch.chdir(tmp_path)
    d = PageDriver("1_Transcriptomics.py", {"RNA Seq Count Matrix": (df.to_csv(index=False).encode(), "tiny.csv")})
    d.run(); d.click("Apply Filter")
    samples = [s for s in d.at.multiselect[0].options]
    d.set_multiselect("Group A (baseline)", [s for s in samples if s.startswith("Ctrl")])
    d.set_multiselect("Group B (comparison)", [s for s in samples if s.startswith("Treat")])
    assert not d.exceptions(), d.exceptions()
    assert "only 2 samples" in d.text()
