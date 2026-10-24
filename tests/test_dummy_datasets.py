"""
Locks in the results documented in BioAI_dummy_datasets/README_DATASETS.md.
If one of these fails, either a bug was introduced or the documented numbers changed.
"""
import os

import numpy as np
import pandas as pd
import pytest

from modules.figure_studio.data_loader import load_file
from modules.proteomics.missing_value_handler import presence_absence_table
from modules.proteomics.normalization import ProteomicsNormalizer
from modules.transcriptomics.differential_expression import DifferentialExpression
from modules.transcriptomics.feature_builder import FeatureBuilder
from modules.transcriptomics.ml_classifier import MLClassifier
from modules.transcriptomics.multi_group_de import MultiGroupDE
from modules.transcriptomics.qc_diagnostics import QCDiagnostics
from modules.transcriptomics.validator import detect_sample_columns
from tests.page_helpers import Upload

D = os.path.join(os.path.dirname(__file__), "data", "dummy")
KEY = pd.read_csv(os.path.join(D, "ANSWER_KEY_planted_genes_and_proteins.csv"))


def planted(name):
    return set(KEY[KEY.dataset == name].gene_or_protein)


def groups_of(df, first_col="Gene"):
    cols = [c for c in df.columns if c != first_col and pd.api.types.is_numeric_dtype(df[c])]
    return cols, {c: c.split("_")[0] for c in cols}


def test_rnaseq_1_two_groups():
    f = "RNAseq_1_two_groups_counts.csv"
    df = pd.read_csv(os.path.join(D, f)); cols, g = groups_of(df)
    a = [c for c in cols if g[c] == "Control"]; b = [c for c in cols if g[c] == "Treated"]
    r = DifferentialExpression(df, a, b).run()
    sig = set(r[r.Significant].Gene); p = planted(f)
    assert len(sig & p) / len(p) >= 0.55 and len(sig - p) <= 10


def test_rnaseq_2_three_groups_anova():
    f = "RNAseq_2_three_groups_counts.csv"
    df = pd.read_csv(os.path.join(D, f)); cols, g = groups_of(df)
    r = MultiGroupDE(df, g, "Gene").run()
    sig = set(r[r.Significant].Gene); p = planted(f)
    assert len(sig & p) / len(p) >= 0.6 and len(sig - p) <= 25


def test_rnaseq_3_messy_file():
    f = "RNAseq_3_messy_edge_cases_counts.csv"
    df = pd.read_csv(os.path.join(D, f))
    assert "Description" not in detect_sample_columns(df)
    counts = df.drop(columns=["Description"])
    _, flagged = QCDiagnostics(counts).flag_outlier_samples(threshold=2.0)
    assert flagged == ["Treated_6"]
    assert (counts.drop(columns="Gene").sum(axis=1) == 0).sum() == 20       # the all-zero genes exist
    assert counts.Gene.duplicated().sum() == 1


def test_rnaseq_4_pure_noise_has_no_hits_and_chance_level_ml():
    df = pd.read_csv(os.path.join(D, "RNAseq_4_pure_noise_counts.csv")); cols, g = groups_of(df)
    a = [c for c in cols if g[c] == "Control"]; b = [c for c in cols if g[c] == "Treated"]
    r = DifferentialExpression(df, a, b).run()
    assert r.Significant.sum() <= 2
    X, y, _, _ = FeatureBuilder(df, "Gene").build(g)
    s, _ = MLClassifier(X, y).evaluate(["Logistic Regression", "Random Forest"], n_folds=5, select_k=50)
    assert s["Accuracy_Mean"].max() < 0.75


def test_proteomics_1_two_groups_and_switched_on_proteins():
    f = "Proteomics_1_two_groups_log2_intensity.csv"
    df = pd.read_csv(os.path.join(D, f)); cols, g = groups_of(df)
    a = [c for c in cols if g[c] == "Normal"]; b = [c for c in cols if g[c] == "Tumor"]
    from modules.proteomics.missing_value_handler import ProteomicsFilter
    filt = ProteomicsFilter(df, "Gene", cols).filter_by_group_detection(g, 0.7)
    norm = ProteomicsNormalizer(filt, cols).median_normalization()
    r = DifferentialExpression(norm, a, b, already_normalized=True).run()
    sig = set(r[r.Significant].Gene); p = planted(f); only = {x for x in p if "TUMORONLY" in x}
    assert len(sig & (p - only)) / len(p - only) >= 0.7 and len(sig - p) <= 3
    t = presence_absence_table(norm, "Gene", a, b)
    assert set(t.Gene) == only and set(t.Present_in) == {"Group B only"}


def test_proteomics_2_three_groups_anova():
    f = "Proteomics_2_three_groups_log2_intensity.csv"
    df = pd.read_csv(os.path.join(D, f)); cols, g = groups_of(df)
    norm = ProteomicsNormalizer(df, cols).median_normalization()
    r = MultiGroupDE(norm, g, "Gene", already_normalized=True).run()
    sig = set(r[r.Significant].Gene); p = planted(f)
    assert len(sig & p) / len(p) >= 0.9 and len(sig - p) <= 20


def test_proteomics_3_log_ratios_with_negative_numbers():
    f = "Proteomics_3_log_ratios_with_negatives.csv"
    df = pd.read_csv(os.path.join(D, f)); cols, g = groups_of(df)
    assert np.nanmin(df[cols].values) < 0
    a = [c for c in cols if g[c] == "Normal"]; b = [c for c in cols if g[c] == "Tumor"]
    r = DifferentialExpression(df, a, b, already_normalized=True).run()
    sig = set(r[r.Significant].Gene); p = planted(f)
    assert len(sig & p) / len(p) >= 0.9 and len(sig - p) <= 3


def test_proteomics_4_european_format_file_loads_cleanly():
    path = os.path.join(D, "Proteomics_4_messy_semicolon_decimal_comma.csv")
    df = load_file(Upload(open(path, "rb").read(), "Proteomics_4_messy_semicolon_decimal_comma.csv"))
    assert df.shape == (300, 18)
    assert len(detect_sample_columns(df)) == 16 and df["Description"].dtype == object
