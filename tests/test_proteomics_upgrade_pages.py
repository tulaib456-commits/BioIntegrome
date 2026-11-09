"""Drives the upgraded Proteomics upload/DE tabs and the new Discovery page. Network is mocked."""
import os

import numpy as np
import pandas as pd
import pytest

from modules.proteomics import annotation as ann
from tests.page_helpers import PageDriver, install_fake_login
from tests.test_proteomics_upgrade import (
    STRING_JSON, UNIPROT_ENTRY, build_backbone_pdb, diann_table, maxquant_table, REMARKS,
)

DUMMY = os.path.join(os.path.dirname(__file__), "data", "dummy")


def ok(d):
    assert not d.exceptions(), d.exceptions()


def tsv(df):
    return df.to_csv(index=False, sep="\t").encode()


@pytest.fixture()
def env(tmp_path, monkeypatch):
    install_fake_login()
    monkeypatch.chdir(tmp_path)


def upload_driver(data, name):
    d = PageDriver("2_Proteomics.py", {"Upload dataset": (data, name)})
    d.run()
    return d


# ------------------------------------------------------------------ upload tab: readers
def test_maxquant_file_is_recognised_on_the_page(env):
    d = upload_driver(tsv(maxquant_table()), "proteinGroups.txt"); ok(d)
    text = d.text()
    assert "Recognised: MaxQuant" in text and "1 reverse hits removed" in text and "Uploaded: 5 proteins, 4 samples" in text


def test_diann_file_is_recognised_on_the_page(env):
    d = upload_driver(tsv(diann_table()), "report.pg_matrix.tsv"); ok(d)
    assert "Recognised: DIA-NN" in d.text() and "Uploaded: 3 proteins, 4 samples" in d.text()


def test_raw_intensity_table_gets_a_warning_and_is_log_transformed_by_default(env):
    rng = np.random.default_rng(0)
    raw = pd.DataFrame(rng.lognormal(16, 1, (50, 4)) * 10, columns=["A_1", "A_2", "B_1", "B_2"])
    raw.insert(0, "Gene", [f"G{i}" for i in range(50)])
    d = upload_driver(raw.to_csv(index=False).encode(), "raw.csv"); ok(d)
    assert "look like raw intensities" in d.text()
    stored = d.at.session_state["proteomics_dataset"]
    assert stored[["A_1", "A_2", "B_1", "B_2"]].median().median() < 100          # now log2 scale


def test_every_dummy_proteomics_file_uploads_on_the_page(env):
    for f, expected in (("Proteomics_1_two_groups_log2_intensity.csv", "2000 proteins, 16 samples"),
                        ("Proteomics_3_log_ratios_with_negatives.csv", "1200 proteins, 20 samples"),
                        ("Proteomics_4_messy_semicolon_decimal_comma.csv", "300 proteins, 16 samples")):
        d = upload_driver(open(os.path.join(DUMMY, f), "rb").read(), f); ok(d)
        assert expected in d.text(), (f, d.text()[:300])


# ------------------------------------------------------------------ DE tab: ANOVA results persist
def test_anova_results_stay_visible_when_a_dropdown_below_is_changed(env):
    f = "Proteomics_2_three_groups_log2_intensity.csv"
    d = upload_driver(open(os.path.join(DUMMY, f), "rb").read(), f)
    d.assign_groups("proteomics_mv", {"Healthy_": "Healthy", "Mild_": "Mild", "Severe_": "Severe"})
    d.set_radio("How should the remaining missing values be handled?", "Use the values each protein has (no imputation)")
    d.click("Filter and Handle Missing Values"); d.click("Apply Normalization")
    d.set_radio_key("prot_de_mode", "Multi-Group (ANOVA, 3+ conditions)")
    d.assign_groups("prot_anova", {"Healthy_": "Healthy", "Mild_": "Mild", "Severe_": "Severe"})
    d.click_key("prot_anova_run"); ok(d)
    assert "significant proteins" in d.text()
    d.at.selectbox(key="prot_top_n").set_value(50); d._patched_run(); ok(d)            # touching ANY widget re-runs the page
    assert "significant proteins" in d.text() and d.at.selectbox(key="prot_anova_select")


# ------------------------------------------------------------------ Discovery page
def de_state():
    de = pd.DataFrame({
        "Gene": ["EGFR", "TP53", "ATM", "MKI67", "FLAT"], "Mean_GroupA": 1.0, "Mean_GroupB": 2.0,
        "Log2FC": [2.5, -1.8, 1.4, 3.0, 0.1], "P_Value": [1e-6, 1e-5, 1e-3, 1e-7, 0.8],
        "Adj_P_Value": [1e-5, 1e-4, 1e-2, 1e-6, 0.9], "N_GroupA": 4, "N_GroupB": 4,
        "Significant": [True, True, True, True, False]})
    cols = ["A_1", "A_2", "B_1", "B_2"]
    rng = np.random.default_rng(0)
    norm = pd.DataFrame(rng.normal(25, 1, (6, 4)), columns=cols)
    norm.insert(0, "Gene", ["EGFR", "TP53", "ATM", "MKI67", "FLAT", "ONLYB"])
    norm.loc[5, ["A_1", "A_2"]] = np.nan
    dataset = norm.copy(); dataset.insert(1, "UniProt_ID", ["P00533", "P04637", "Q13315", "P46013", "P99999", "Q11111"])
    return de, dataset, norm, (cols[:2], cols[2:])


def discovery(with_results=True, uploads=None):
    d = PageDriver("2_Proteomics_Discovery.py", uploads or {})
    if with_results:
        de, dataset, norm, groups = de_state()
        d.at.session_state["proteomics_de_results"] = de
        d.at.session_state["proteomics_dataset"] = dataset
        d.at.session_state["proteomics_normalized"] = norm
        d.at.session_state["proteomics_de_groups"] = groups
        d.at.session_state["proteomics_de_kind"] = "two_group"
    d.run()
    return d


def fake_network(monkeypatch, record=None):
    def fake_get(url, params=None, as_json=True, timeout=25):
        if record is not None:
            record.append((url, params))
        if "data.rcsb.org/rest/v1/core/entry/" in url:
            return {"rcsb_entry_container_identifiers": {"polymer_entity_ids": ["1"]}}, None
        if "data.rcsb.org/rest/v1/core/polymer_entity/" in url:
            return {"rcsb_polymer_entity_container_identifiers": {"uniprot_ids": ["P00533"]},
                    "rcsb_polymer_entity": {"pdbx_description": "Epidermal growth factor receptor"}}, None
        if "files.rcsb.org" in url:
            return REMARKS + build_backbone_pdb([(-57.0, -47.0)] * 25), None
        if "/search" in url:
            return {"results": [{"primaryAccession": "P00533"}]}, None
        if url.endswith(".json") and "uniprotkb" in url:
            return UNIPROT_ENTRY, None
        if "string-db.org" in url:
            return STRING_JSON, None
        if "alphafold.ebi.ac.uk/api" in url:
            return [{"entryId": "AF-P00533-F1", "pdbUrl": "https://alphafold.ebi.ac.uk/files/AF-P00533-F1-model_v6.pdb",
                     "globalMetricValue": 75.5, "latestVersion": 6, "gene": "EGFR"}], None
        if url.endswith(".pdb"):
            return build_backbone_pdb([(-57.0, -47.0)] * 20), None
        return None, "Not found in this database."
    monkeypatch.setattr(ann, "_get", fake_get)


def test_discovery_page_opens_without_any_data(env):
    d = discovery(with_results=False); ok(d)
    assert "Run Differential Expression" in d.text()


def test_shortlist_ranks_by_score_and_carries_uniprot_ids(env):
    d = discovery(); ok(d)
    options = list(d.at.selectbox(key="disc_pick").options)
    assert options[:5] == ["(none)", "MKI67", "EGFR", "TP53", "ATM"] and "FLAT" not in options
    assert "ONLYB" in options                                   # detected in group B only


def test_profile_lookup_uses_the_known_uniprot_id_without_searching(env, monkeypatch):
    calls = []
    fake_network(monkeypatch, calls)
    d = discovery()
    d.at.selectbox(key="disc_pick").set_value("EGFR"); d._patched_run()
    d.click_key("disc_profile_btn"); ok(d)
    assert "Epidermal growth factor receptor" in d.text() and "Receptor tyrosine kinase" in d.text()
    assert not any("/search" in url for url, _ in calls)        # UniProt_ID column made the search unnecessary
    assert any(url.endswith("P00533.json") for url, _ in calls)


def test_profile_for_a_typed_gene_symbol_searches_uniprot(env, monkeypatch):
    calls = []
    fake_network(monkeypatch, calls)
    d = discovery(with_results=False)
    d.set_text("disc_typed", "EGFR"); d.click_key("disc_profile_btn"); ok(d)
    assert any("/search" in url for url, _ in calls) and "Epidermal growth factor receptor" in d.text()


def test_interaction_network_marks_hubs_and_proteins_changed_in_the_users_data(env, monkeypatch):
    fake_network(monkeypatch)
    d = discovery()
    d.at.selectbox(key="disc_pick").set_value("TP53"); d._patched_run()
    d.click_key("disc_string_btn"); ok(d)
    text = d.text()
    assert "Hub proteins" in text and "TP53" in text and "also changed in your experiment" in text


def test_alphafold_tab_shows_confidence_and_the_viewer(env, monkeypatch):
    fake_network(monkeypatch)
    d = discovery()
    d.at.selectbox(key="disc_pick").set_value("EGFR"); d._patched_run()
    d.click_key("disc_af_btn"); ok(d)
    assert any(m.label.startswith("Mean confidence") and "75.5" in str(m.value) for m in d.at.metric)


def test_network_failure_shows_a_readable_message_on_every_tab(env, monkeypatch):
    monkeypatch.setattr(ann, "_get", lambda *a, **k: (None, "Could not reach the database (ConnectionError). Check the server's internet access and try again."))
    d = discovery(with_results=False)
    d.set_text("disc_typed", "EGFR")
    for key in ("disc_profile_btn", "disc_string_btn", "disc_af_btn"):
        d.click_key(key); ok(d)
    assert "Could not reach the database" in d.text()


def test_unknown_protein_gives_a_message_not_a_crash(env, monkeypatch):
    monkeypatch.setattr(ann, "_get", lambda *a, **k: ({"results": []}, None))
    d = discovery(with_results=False)
    d.set_text("disc_typed", "NOTAGENE"); d.click_key("disc_profile_btn"); ok(d)
    assert "No reviewed UniProt entry" in d.text()


def test_own_structure_upload_reports_quality_and_angles(env):
    pdb = REMARKS + build_backbone_pdb([(-57.0, -47.0)] * 30)
    d = discovery(with_results=False, uploads={"PDB file": (pdb.encode(), "model.pdb")}); ok(d)
    values = {m.label: str(m.value) for m in d.at.metric}
    assert values["Resolution"] == "1.8 Å" and values["R-work / R-free"] == "0.193 / 0.262" and values["Favoured (approx.)"] == "100.0%"
    assert "over-fitting" in d.text()


def test_non_pdb_upload_is_refused_politely(env):
    d = discovery(with_results=False, uploads={"PDB file": (b"data_1ABC\n_cell.length_a 10\n", "x.cif")}); ok(d)
    assert "No atoms found" in d.text()


def test_dose_response_example_fits_two_compounds(env):
    d = discovery(with_results=False)
    d.at.checkbox(key="disc_dr_example").set_value(True); d._patched_run(); ok(d)
    assert d.at.dataframe
    table = [t for t in d.at.dataframe if "Compound" in t.value.columns][-1].value
    assert set(table["Compound"]) == {"Compound A", "Compound B"}


def test_dose_response_with_too_few_concentrations_warns_instead_of_crashing(env):
    csv = "Concentration,Response\n1,90\n1,88\n10,50\n10,52\n100,10\n100,12\n"
    d = discovery(with_results=False, uploads={"Dose-response table (CSV / TSV / Excel)": (csv.encode(), "dr.csv")}); ok(d)
    assert "at least 4" in d.text()


def test_typing_a_pdb_code_finds_the_protein_instead_of_failing(env, monkeypatch):
    """The exact thing that went wrong on the live site: the user typed 7cr5."""
    calls = []
    fake_network(monkeypatch, calls)
    d = discovery(with_results=False)
    d.set_text("disc_typed", "7cr5"); ok(d)
    assert "PDB entry 7CR5 → UniProt P00533" in d.text()
    d.click_key("disc_profile_btn"); ok(d)
    assert "Epidermal growth factor receptor" in d.text()
    assert not any("/search" in url for url, _ in calls)


def test_a_pdb_code_with_several_proteins_lets_the_user_choose(env, monkeypatch):
    from tests.test_proteomics_upgrade import rcsb_fake
    base = rcsb_fake()
    def fake(url, params=None, as_json=True, timeout=25):
        if url.endswith(".json") and "uniprotkb" in url:
            return UNIPROT_ENTRY, None
        return base(url, params, as_json, timeout)
    monkeypatch.setattr(ann, "_get", fake)
    d = discovery(with_results=False)
    d.set_text("disc_typed", "7cr5"); ok(d)
    options = list(d.at.selectbox(key="disc_pdb_pick_7cr5").options)
    assert len(options) == 2 and "Spike glycoprotein" in options[0] and "P0DTC2" in options[0]


def test_when_the_pdb_lookup_fails_the_page_shows_the_reason_instead_of_a_gene_search_message(env, monkeypatch):
    """The live failure: 'No reviewed UniProt entry found for 7CR5' hid the real cause."""
    from tests.test_proteomics_upgrade import rcsb_fake
    monkeypatch.setattr(ann, "_get", rcsb_fake(entry_ok=False, pdbe=(None, "The database answered with an error (HTTP 410): gone")))
    d = discovery(with_results=False)
    d.set_text("disc_typed", "7CR5"); ok(d)
    assert "Could not find the protein of PDB entry 7CR5" in d.text() and "HTTP 503" in d.text()
    d.click_key("disc_profile_btn"); ok(d)
    assert "No reviewed UniProt entry" not in d.text() and "HTTP 410" in d.text()


def test_a_server_error_is_shown_with_its_http_code(env, monkeypatch):
    monkeypatch.setattr(ann, "_get", lambda *a, **k: (None, "The database answered with an error (HTTP 503): maintenance"))
    d = discovery(with_results=False)
    d.set_text("disc_typed", "EGFR"); d.click_key("disc_profile_btn"); ok(d)
    assert "HTTP 503" in d.text() and "maintenance" in d.text()


def test_structure_can_be_fetched_by_pdb_code_and_analysed(env, monkeypatch):
    fake_network(monkeypatch)
    d = discovery(with_results=False)
    d.set_text("disc_pdb_code", "1m17"); d.click_key("disc_pdb_fetch"); ok(d)
    values = {m.label: str(m.value) for m in d.at.metric}
    assert values["Resolution"] == "1.8 Å" and values["Favoured (approx.)"] == "100.0%"
    assert "Showing PDB entry 1M17" in d.text()


def test_structure_fetch_failure_is_explained_not_crashed(env, monkeypatch):
    monkeypatch.setattr(ann, "_get", lambda *a, **k: (None, "Not found in this database."))
    d = discovery(with_results=False)
    d.set_text("disc_pdb_code", "7cr5"); d.click_key("disc_pdb_fetch"); ok(d)
    assert "mmCIF" in d.text()


def test_interaction_network_looks_up_the_gene_symbol_when_only_a_uniprot_id_is_known(env, monkeypatch):
    seen = []
    fake_network(monkeypatch, seen)
    d = discovery(with_results=False)
    d.set_text("disc_typed", "P00533"); d.click_key("disc_string_btn"); ok(d)
    string_calls = [params for url, params in seen if "string-db.org" in url]
    assert string_calls and string_calls[0]["identifiers"].split("\r")[0] == "EGFR"
