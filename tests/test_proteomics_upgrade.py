"""
Tests for: file readers (MaxQuant / DIA-NN), UniProt / STRING / AlphaFold parsing,
structure analysis, dose-response fitting, candidate shortlist.

Network is never used: modules.proteomics.annotation._get is replaced by recorded-format fixtures.
"""
import numpy as np
import pandas as pd
import pytest

from modules.proteomics import annotation as ann
from modules.proteomics import candidates, dose_response as dr, parsers, structure as st
from tests.page_helpers import Upload


# ====================================================================== readers
def maxquant_table():
    rows = [
        # Protein IDs, Majority IDs, Gene names, Reverse, Contaminant, OnlySite, LFQ A_1, A_2, B_1, B_2
        ["P00533;Q9XXXX", "P00533", "EGFR;ERBB1", "", "", "", 1.0e8, 1.2e8, 4.0e8, 4.4e8],
        ["sp|P04637|P53_HUMAN", "sp|P04637|P53_HUMAN", "TP53", "", "", "", 2.0e7, 0, 2.2e7, 2.1e7],
        ["REV__P99999", "REV__P99999", "", "+", "", "", 5e6, 5e6, 5e6, 5e6],
        ["CON__P02768", "CON__P02768", "ALB", "", "+", "", 9e9, 9e9, 9e9, 9e9],
        ["P11111", "P11111", "", "", "", "+", 1e6, 1e6, 1e6, 1e6],
        ["P12345", "P12345", "DUP", "", "", "", 3e6, 3e6, 3e6, 3e6],
        ["P67890", "P67890", "DUP", "", "", "", 4e6, 4e6, 4e6, 4e6],
        ["Q00000", "Q00000", "", "", "", "", 0, 0, 0, 0],          # never measured
        ["Q11111", "Q11111", "", "", "", "", 7e6, 7e6, 7e6, 7e6],  # no gene name
    ]
    cols = ["Protein IDs", "Majority protein IDs", "Gene names", "Reverse", "Potential contaminant",
            "Only identified by site", "LFQ intensity A_1", "LFQ intensity A_2", "LFQ intensity B_1", "LFQ intensity B_2"]
    df = pd.DataFrame(rows, columns=cols)
    for c in ("Intensity A_1", "Intensity A_2", "Intensity B_1", "Intensity B_2"):
        df[c] = 1.0
    return df


def as_upload(df, name, sep="\t"):
    return Upload(df.to_csv(index=False, sep=sep).encode(), name)


def test_maxquant_file_is_cleaned_and_log2_transformed():
    table, info = parsers.load_proteomics_file(as_upload(maxquant_table(), "proteinGroups.txt"))
    assert info["format"].startswith("MaxQuant") and info["intensity_used"] == "LFQ intensity"
    assert info["removed_reverse_hits"] == 1 and info["removed_potential_contaminants"] == 1
    assert info["removed_only_identified_by_site"] == 1 and info["removed_never_measured"] == 1
    assert list(table.columns)[:2] == ["Gene", "UniProt_ID"] and list(table.columns)[2:] == ["A_1", "A_2", "B_1", "B_2"]
    egfr = table[table.Gene == "EGFR"].iloc[0]
    assert egfr.UniProt_ID == "P00533" and egfr.A_1 == pytest.approx(np.log2(1.0e8))
    tp53 = table[table.Gene == "TP53"].iloc[0]
    assert tp53.UniProt_ID == "P04637" and np.isnan(tp53.A_2)          # a 0 means 'not measured'
    assert pd.api.types.is_float_dtype(table["A_1"])


def test_maxquant_duplicate_and_missing_gene_names_are_made_unique_not_lost():
    table, _ = parsers.load_proteomics_file(as_upload(maxquant_table(), "proteinGroups.txt"))
    genes = list(table.Gene)
    assert len(genes) == len(set(genes))
    assert "DUP (P12345)" in genes and "DUP (P67890)" in genes and "Q11111" in genes


def test_maxquant_without_lfq_uses_intensity_and_says_so():
    df = maxquant_table().drop(columns=[c for c in maxquant_table().columns if c.startswith("LFQ")])
    df[["Intensity A_1", "Intensity A_2", "Intensity B_1", "Intensity B_2"]] = 5e6
    _, info = parsers.load_proteomics_file(as_upload(df, "proteinGroups.txt"))
    assert info["intensity_used"] == "Intensity"


def diann_table():
    return pd.DataFrame({
        "Protein.Group": ["P00533", "P04637;Q9ZZZZ", "P12345"],
        "Protein.Names": ["EGFR_HUMAN", "P53_HUMAN", "X_HUMAN"],
        "Genes": ["EGFR", "TP53", ""],
        "First.Protein.Description": ["a", "b", "c"],
        "D:\\data\\ctrl_1.raw": [1e6, 2e6, 0.0],
        "D:\\data\\ctrl_2.raw": [1.1e6, np.nan, 5e5],
        "D:\\data\\trt_1.raw": [4e6, 2e6, 5e5],
        "D:\\data\\trt_2.raw": [4.2e6, 2.2e6, 6e5],
    })


def test_diann_matrix_is_read_with_clean_run_names():
    table, info = parsers.load_proteomics_file(as_upload(diann_table(), "report.pg_matrix.tsv"))
    assert info["format"].startswith("DIA-NN")
    assert list(table.columns) == ["Gene", "UniProt_ID", "ctrl_1", "ctrl_2", "trt_1", "trt_2"]
    assert table.loc[table.Gene == "EGFR", "trt_1"].iloc[0] == pytest.approx(np.log2(4e6))
    assert "P12345" in list(table.Gene)                      # no gene name -> accession
    assert np.isnan(table.loc[table.Gene == "P12345", "ctrl_1"].iloc[0])


def test_generic_table_is_returned_unchanged_and_raw_intensities_are_detected():
    raw = pd.DataFrame({"Gene": list("abc"), "S1": [2e7, 3e7, 1e7], "S2": [2.1e7, 3.1e7, 1.1e7]})
    table, info = parsers.load_proteomics_file(as_upload(raw, "my.csv", sep=","))
    assert info["format"] == "generic table" and table["S1"].tolist() == raw["S1"].tolist()
    assert parsers.looks_unlogged(table, ["S1", "S2"])
    logged = parsers.to_log2(table, ["S1", "S2"])
    assert not parsers.looks_unlogged(logged, ["S1", "S2"])
    assert parsers.to_log2(pd.DataFrame({"S": [0.0, -1.0, 8.0]}), ["S"])["S"].isna().tolist() == [True, True, False]


def test_log2_data_is_not_flagged_as_raw():
    logged = pd.DataFrame({"S1": [22.1, 25.3, 28.0], "S2": [22.4, 25.0, 27.7]})
    assert not parsers.looks_unlogged(logged, ["S1", "S2"])


def test_first_accession_handles_uniprot_header_styles():
    assert parsers.first_accession("sp|P00533|EGFR_HUMAN") == "P00533"
    assert parsers.first_accession("Q9XXXX-2;P00533") in ("Q9XXXX", "P00533")
    assert parsers.first_accession("CON__ALBU") is None
    assert parsers.first_accession(np.nan) is None


# ====================================================================== annotation (mocked network)
UNIPROT_ENTRY = {   # structure follows the real UniProt REST JSON (fields verified from public examples)
    "entryType": "UniProtKB reviewed (Swiss-Prot)", "primaryAccession": "P00533", "uniProtkbId": "EGFR_HUMAN",
    "organism": {"scientificName": "Homo sapiens"},
    "proteinDescription": {"recommendedName": {"fullName": {"value": "Epidermal growth factor receptor"}}},
    "genes": [{"geneName": {"value": "EGFR"}}],
    "comments": [
        {"commentType": "FUNCTION", "texts": [{"value": "Receptor tyrosine kinase binding ligands of the EGF family."}]},
        {"commentType": "SUBCELLULAR LOCATION", "subcellularLocations": [{"location": {"value": "Cell membrane"}}, {"location": {"value": "Endosome"}}]},
        {"commentType": "DISEASE", "disease": {"diseaseId": "Lung cancer"}},
    ],
    "features": [
        {"type": "Domain", "description": "Protein kinase", "location": {"start": {"value": 712}, "end": {"value": 979}}},
        {"type": "Modified residue", "description": "Phosphotyrosine", "location": {}},
        {"type": "Modified residue", "description": "Phosphoserine", "location": {}},
        {"type": "Glycosylation", "description": "N-linked", "location": {}},
    ],
    "uniProtKBCrossReferences": [
        {"database": "PDB", "id": "1M17", "properties": [{"key": "Method", "value": "X-ray"}, {"key": "Resolution", "value": "2.60 A"}, {"key": "Chains", "value": "A=696-1022"}]},
        {"database": "PDB", "id": "5CNO", "properties": [{"key": "Method", "value": "X-ray"}, {"key": "Resolution", "value": "2.00 A"}, {"key": "Chains", "value": "A=696-1022"}]},
        {"database": "ChEMBL", "id": "CHEMBL203"}, {"database": "DrugBank", "id": "DB00317"},
    ],
    "keywords": [{"name": "Kinase"}, {"name": "Receptor"}, {"name": "Membrane"}],
    "sequence": {"length": 1210, "molWeight": 134277},
}


def test_uniprot_entry_is_parsed():
    p = ann.parse_uniprot(UNIPROT_ENTRY)
    assert (p["accession"], p["gene"], p["protein_name"], p["organism"]) == ("P00533", "EGFR", "Epidermal growth factor receptor", "Homo sapiens")
    assert p["reviewed"] is True and p["length"] == 1210 and p["mass_da"] == 134277
    assert "tyrosine kinase" in p["function"] and p["locations"] == ["Cell membrane", "Endosome"]
    assert p["diseases"] == ["Lung cancer"]
    assert p["domains"] == [{"Type": "Domain", "Name": "Protein kinase", "Start": 712, "End": 979}]
    assert p["ptm_counts"] == {"Modified residue": 2, "Glycosylation": 1}
    assert [s["PDB_ID"] for s in p["structures"]] == ["1M17", "5CNO"]
    assert p["drug_links"] == {"ChEMBL": ["CHEMBL203"], "DrugBank": ["DB00317"]}


def test_druggability_indicators_report_facts_including_best_resolution():
    rows = dict(ann.druggability_indicators(ann.parse_uniprot(UNIPROT_ENTRY)))
    assert "2 (2 X-ray, best resolution 2.0" in rows["Experimental 3D structures (PDB)"]
    assert rows["ChEMBL entry"] == "CHEMBL203" and rows["DrugCentral entry"] == "none listed"


def test_uniprot_entry_with_missing_sections_does_not_crash():
    p = ann.parse_uniprot({"primaryAccession": "Q00000"})
    assert p["accession"] == "Q00000" and p["function"] == "" and p["structures"] == []
    ann.druggability_indicators(p)


def test_resolve_accession_needs_no_network_and_gene_symbol_uses_search(monkeypatch):
    monkeypatch.setattr(ann, "_get", lambda *a, **k: (_ for _ in ()).throw(AssertionError("network used")))
    assert ann.resolve_uniprot("P00533") == ("P00533", None)
    monkeypatch.setattr(ann, "_get", lambda url, params=None, **k: ({"results": [{"primaryAccession": "P00533"}]}, None))
    assert ann.resolve_uniprot("EGFR") == ("P00533", None)
    monkeypatch.setattr(ann, "_get", lambda *a, **k: ({"results": []}, None))
    acc, err = ann.resolve_uniprot("NOTAGENE")
    assert acc is None and "No reviewed UniProt entry" in err


def test_network_failure_gives_a_message_not_an_exception(monkeypatch):
    import requests
    monkeypatch.setattr(requests, "get", lambda *a, **k: (_ for _ in ()).throw(requests.ConnectionError("offline")))
    data, err = ann._get("https://rest.uniprot.org/uniprotkb/P00533.json")
    assert data is None and "Could not reach the database" in err


STRING_JSON = [   # real STRING JSON field names (TP53 / ATM / BRCA1 / BRCA2 / ATR example)
    {"stringId_A": "9606.ENSP00000269305", "stringId_B": "9606.ENSP00000369497", "preferredName_A": "TP53", "preferredName_B": "BRCA2", "score": 0.995},
    {"stringId_A": "9606.ENSP00000269305", "stringId_B": "9606.ENSP00000343741", "preferredName_A": "TP53", "preferredName_B": "ATR", "score": 0.996},
    {"stringId_A": "9606.ENSP00000269305", "stringId_B": "9606.ENSP00000278616", "preferredName_A": "TP53", "preferredName_B": "ATM", "score": 0.999},
    {"stringId_A": "9606.ENSP00000269305", "stringId_B": "9606.ENSP00000418960", "preferredName_A": "TP53", "preferredName_B": "BRCA1", "score": 0.999},
    {"stringId_A": "9606.ENSP00000278616", "stringId_B": "9606.ENSP00000418960", "preferredName_A": "ATM", "preferredName_B": "BRCA1", "score": 0.999},
    {"stringId_A": "9606.ENSP00000418960", "stringId_B": "9606.ENSP00000278616", "preferredName_A": "BRCA1", "preferredName_B": "ATM", "score": 0.999},   # reverse duplicate
    {"stringId_A": "x", "stringId_B": "y", "preferredName_A": "TP53", "preferredName_B": "TP53", "score": 1.0},                                       # self loop
]


def test_string_edges_are_deduplicated_and_metrics_find_the_hub():
    edges = ann.parse_string_edges(STRING_JSON)
    assert len(edges) == 5 and set(edges.columns) >= {"Protein_A", "Protein_B", "Score"}
    m = ann.network_metrics(edges, own_proteins=["ATM", "BRCA1"]).set_index("Protein")
    assert m.loc["TP53", "Partners"] == 4 and m.loc["TP53", "Hub"]
    assert m.loc["ATM", "In_your_data"] and not m.loc["ATR", "In_your_data"]


def test_empty_string_response_is_handled(monkeypatch):
    monkeypatch.setattr(ann, "_get", lambda *a, **k: ([], None))
    edges, msg = ann.fetch_string_network(["TP53"])
    assert edges.empty and "Lower the minimum score" in msg


def test_spring_layout_is_finite_deterministic_and_figure_builds():
    edges = ann.parse_string_edges(STRING_JSON)
    metrics = ann.network_metrics(edges)
    nodes = list(metrics.Protein)
    pos1 = ann.spring_layout(nodes, list(zip(edges.Protein_A, edges.Protein_B)))
    pos2 = ann.spring_layout(nodes, list(zip(edges.Protein_A, edges.Protein_B)))
    assert pos1 == pos2 and all(np.isfinite(v).all() for v in pos1.values())
    fig = ann.network_figure(edges, metrics, focus="TP53", log2fc={"ATM": 1.5, "BRCA1": -2.0})
    assert len(fig.data) == 2


def test_alphafold_response_and_not_found(monkeypatch):
    item = [{"entryId": "AF-P00533-F1", "pdbUrl": "https://alphafold.ebi.ac.uk/files/AF-P00533-F1-model_v6.pdb",
             "cifUrl": "c", "globalMetricValue": 75.5, "latestVersion": 6, "gene": "EGFR"}]
    monkeypatch.setattr(ann, "_get", lambda *a, **k: (item, None))
    info, err = ann.fetch_alphafold("P00533")
    assert err is None and info["pdb_url"].endswith("model_v6.pdb") and info["mean_plddt"] == 75.5
    monkeypatch.setattr(ann, "_get", lambda *a, **k: (None, "Not found in this database."))
    assert ann.fetch_alphafold("P00000")[1] == "No AlphaFold model found for this protein."


def test_oversized_structure_file_is_refused_not_embedded(monkeypatch):
    monkeypatch.setattr(ann, "_get", lambda *a, **k: ("A" * 100, None))
    assert ann.download_structure_text("u", max_bytes=10)[1] is not None
    assert ann.download_structure_text("u", max_bytes=1000)[0] == "A" * 100


# ====================================================================== structure
def _place(a, b, c, length, angle_deg, dihedral_deg):
    """NeRF: position of the next atom from three previous atoms and internal coordinates."""
    angle, dih = np.radians(angle_deg), np.radians(dihedral_deg)
    bc = c - b; bc /= np.linalg.norm(bc)
    n = np.cross(b - a, bc); n /= np.linalg.norm(n)
    m = np.cross(n, bc)
    d = np.array([-length * np.cos(angle), length * np.sin(angle) * np.cos(dih), length * np.sin(angle) * np.sin(dih)])
    return c + d[0] * bc + d[1] * m + d[2] * n


def build_backbone_pdb(angles, resnames=None, header=""):
    """Backbone N/CA/C with KNOWN phi/psi for each residue (omega = 180). angles = [(phi, psi), ...]."""
    n_res = len(angles)
    resnames = resnames or ["ALA"] * n_res
    n0, ca0, c0 = np.array([0.0, 0, 0]), np.array([1.458, 0, 0]), None
    c0 = ca0 + 1.525 * np.array([np.cos(np.radians(180 - 111.2)), np.sin(np.radians(180 - 111.2)), 0])
    atoms = [[n0, ca0, c0]]
    for i in range(1, n_res):
        pn, pca, pc = atoms[-1]
        n = _place(pn, pca, pc, 1.329, 116.2, angles[i - 1][1])            # psi(i-1)
        ca = _place(pca, pc, n, 1.458, 121.7, 180.0)                       # omega
        c = _place(pc, n, ca, 1.525, 111.2, angles[i][0])                  # phi(i)
        atoms.append([n, ca, c])
    lines = [header] if header else []
    serial = 1
    for i, (trio, resname) in enumerate(zip(atoms, resnames), start=1):
        for name, xyz in zip(("N", "CA", "C"), trio):
            lines.append(f"ATOM  {serial:5d}  {name:<3s} {resname:3s} A{i:4d}    {xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00{20 + i:6.2f}           {name[0]}")
            serial += 1
    return "\n".join(lines) + "\nEND\n"


def test_dihedral_angles_recover_the_known_phi_psi():
    rng = np.random.default_rng(3)
    angles = [(float(rng.uniform(-170, 170)), float(rng.uniform(-170, 170))) for _ in range(30)]
    table = st.dihedral_table(st.parse_pdb_text(build_backbone_pdb(angles)))
    assert len(table) == 28                                              # first and last residue have no phi/psi
    for k, row in enumerate(table.itertuples(), start=1):
        for got, want in ((row.phi, angles[k][0]), (row.psi, angles[k][1])):
            diff = abs((got - want + 180) % 360 - 180)
            assert diff < 0.5, (k, got, want)


def test_ideal_helix_and_strand_are_favoured_and_a_bad_angle_is_an_outlier():
    helix = build_backbone_pdb([(-57.0, -47.0)] * 12)
    strand = build_backbone_pdb([(-120.0, 130.0)] * 12)
    for pdb, expected in ((helix, "Favoured"), (strand, "Favoured")):
        cls = st.classify_ramachandran(st.dihedral_table(st.parse_pdb_text(pdb)))
        assert (cls.region == expected).all()
    bad = st.classify_ramachandran(st.dihedral_table(st.parse_pdb_text(build_backbone_pdb([(60.0, -120.0)] * 6))))
    assert (bad.region == "Outlier").all()
    summary = st.ramachandran_summary(st.classify_ramachandran(st.dihedral_table(st.parse_pdb_text(helix))))
    assert summary["Favoured_pct"] == 100.0


def test_glycine_is_not_judged():
    pdb = build_backbone_pdb([(60.0, -120.0)] * 6, resnames=["GLY"] * 6)
    cls = st.classify_ramachandran(st.dihedral_table(st.parse_pdb_text(pdb)))
    assert (cls.region == "Glycine (not judged)").all() and st.ramachandran_summary(cls)["residues_judged"] == 0


def test_chain_break_gives_no_angles_across_the_gap():
    atoms = st.parse_pdb_text(build_backbone_pdb([(-57.0, -47.0)] * 8))
    atoms.loc[atoms.resseq > 4, "x"] += 20.0                            # tear the chain between residue 4 and 5
    table = st.dihedral_table(atoms)
    assert set(table.resseq) == {2, 3, 6, 7}


REMARKS = """EXPDTA    X-RAY DIFFRACTION
REMARK   2 RESOLUTION.    1.80 ANGSTROMS.
REMARK   3   R VALUE            (WORKING SET) : 0.193
REMARK   3   FREE R VALUE                     : 0.262
REMARK   3   FREE R VALUE TEST SET SIZE   (%) : 5.000
"""


def test_header_quality_reads_resolution_and_r_values_and_explains_them():
    info = st.header_quality(REMARKS + build_backbone_pdb([(-57.0, -47.0)] * 4))
    assert info == {"method": "X-RAY DIFFRACTION", "resolution": 1.8, "r_work": 0.193, "r_free": 0.262}
    text = " ".join(st.interpret_quality(info))
    assert "high quality" in text and "over-fitting" in text            # gap 0.069 > 0.05
    assert "AlphaFold" in " ".join(st.interpret_quality({"resolution": None, "r_work": None, "r_free": None, "method": None}))


def test_bfactors_composition_and_bad_file():
    def hetatm(serial, name, resname, resseq, xyz, b):
        return f"HETATM{serial:5d}  {name:<3s} {resname:3s} A{resseq:4d}    {xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00{b:6.2f}           {name[0]}\n"
    hetero = hetatm(9000, "O", "HOH", 100, (1, 1, 1), 30.0) + hetatm(9001, "C1", "LIG", 200, (2, 2, 2), 40.0)
    pdb = build_backbone_pdb([(-57.0, -47.0)] * 5) + hetero
    atoms = st.parse_pdb_text(pdb)
    b = st.bfactor_table(atoms)
    assert len(b) == 5 and b.Mean_B.tolist() == [21, 22, 23, 24, 25]
    comp = st.composition(atoms)
    assert comp["residues"] == 5 and comp["waters"] == 1 and comp["ligands"] == ["LIG"]
    with pytest.raises(ValueError, match="No atoms"):
        st.parse_pdb_text("data_1ABC\n_cell.length_a 10\n")


def test_figures_build():
    atoms = st.parse_pdb_text(build_backbone_pdb([(-57.0, -47.0)] * 10))
    st.bfactor_figure(st.bfactor_table(atoms))
    st.ramachandran_figure(st.classify_ramachandran(st.dihedral_table(atoms)))


def test_viewer_html_embeds_the_structure_safely_and_pins_the_library_version():
    html = st.viewer_html('ATOM </script><script>alert(1)</script>', color_by="plddt")
    assert html.count("<script>") == 1 and html.count("</script>") == 1
    assert "3dmol@2.5.5" in html and "plddt" in html


# ====================================================================== dose response
def simulate(ic50=2.5, hill=1.3, bottom=5.0, top=100.0, sd=3.0, reps=3, seed=0, decreasing=True):
    rng = np.random.default_rng(seed)
    conc = np.repeat(np.logspace(-2, 2, 8), reps)
    if decreasing:
        y = dr.four_pl(conc, bottom, top, np.log10(ic50), hill)
    else:
        y = dr.four_pl(conc, top, bottom, np.log10(ic50), hill)     # rises with dose
    return conc, y + rng.normal(0, sd, len(conc))


def test_ic50_is_recovered_without_systematic_bias_and_inside_its_own_interval():
    estimates = []
    for seed in range(60):
        conc, y = simulate(seed=seed)
        fit = dr.fit_four_parameter(conc, y)
        estimates.append(fit["IC50"])
    assert np.median(estimates) == pytest.approx(2.5, rel=0.06)                 # no systematic bias
    fit = dr.fit_four_parameter(*simulate(seed=0))
    assert fit["IC50_low"] < fit["IC50"] < fit["IC50_high"] and fit["R_squared"] > 0.97
    assert 0.9 < fit["Hill"] < 1.8


def test_confidence_interval_has_honest_coverage():
    """A nominal 95% interval must contain the true IC50 in (about) 95% of repeat experiments."""
    hits = 0
    runs = 300
    for seed in range(runs):
        conc, y = simulate(seed=seed + 100)
        fit = dr.fit_four_parameter(conc, y)
        hits += fit["IC50_low"] <= 2.5 <= fit["IC50_high"]
    assert 0.90 <= hits / runs <= 0.99, hits / runs


def test_increasing_curve_activator_ec50():
    conc, y = simulate(ic50=0.8, decreasing=False, seed=4)
    fit = dr.fit_four_parameter(conc, y)
    assert fit["IC50"] == pytest.approx(0.8, rel=0.2) and fit["Top"] < fit["Bottom"] or fit["Top"] > fit["Bottom"]
    assert fit["IC50"] == pytest.approx(0.8, rel=0.2)


def test_bad_inputs_are_refused_with_clear_messages():
    with pytest.raises(ValueError, match="at least 4"):
        dr.fit_four_parameter([1, 1, 10, 10, 100, 100], [90, 88, 50, 52, 10, 12])
    with pytest.raises(ValueError, match="identical"):
        dr.fit_four_parameter(np.logspace(-2, 2, 8), [50] * 8)
    with pytest.raises(ValueError, match="at least 5"):
        dr.fit_four_parameter([0.1, 1, 10, 100], [90, 60, 30, 10])


def test_zero_concentration_is_left_out_of_the_fit_and_used_to_normalise():
    conc, y = simulate(seed=2)
    df = pd.DataFrame({"conc": np.r_[[0.0, 0.0, 0.0], conc], "resp": np.r_[[200.0, 202.0, 198.0], y * 2]})
    norm = dr.normalise_to_control(df, "conc", "resp")
    assert norm.loc[norm.conc == 0, "resp"].mean() == pytest.approx(100.0)
    assert dr.fit_four_parameter(norm.conc, norm.resp)["IC50"] == pytest.approx(2.5, rel=0.2)
    with pytest.raises(ValueError, match="control"):
        dr.normalise_to_control(df[df.conc > 0], "conc", "resp")


def test_extrapolated_ic50_is_flagged():
    conc = np.repeat(np.logspace(-2, 0, 6), 3)                         # tested up to 1, true IC50 = 50
    rng = np.random.default_rng(1)
    y = dr.four_pl(conc, 0, 100, np.log10(50), 1.0) + rng.normal(0, 0.5, len(conc))
    fit = dr.fit_four_parameter(conc, y)
    assert any("extrapolated" in w or "Poor fit" in w or "confidence interval" in w for w in fit["warnings"])


def test_several_compounds_are_fitted_separately_and_a_bad_one_does_not_stop_the_rest():
    c1, y1 = simulate(ic50=1.0, seed=7)
    c2, y2 = simulate(ic50=10.0, seed=8)
    df = pd.concat([
        pd.DataFrame({"conc": c1, "resp": y1, "drug": "A"}),
        pd.DataFrame({"conc": c2, "resp": y2, "drug": "B"}),
        pd.DataFrame({"conc": [1.0, 10.0], "resp": [50.0, 40.0], "drug": "bad"}),
    ])
    table, fits, errors = dr.fit_table(df, "conc", "resp", "drug")
    assert set(table.Compound) == {"A", "B"} and "bad" in errors
    ic = table.set_index("Compound").IC50
    assert ic["A"] == pytest.approx(1.0, rel=0.2) and ic["B"] == pytest.approx(10.0, rel=0.2)
    dr.dose_response_figure(df, "conc", "resp", fits, "drug")


# ====================================================================== candidate shortlist
def test_candidates_are_ranked_by_evidence_times_effect_and_one_group_proteins_are_appended():
    de = pd.DataFrame({"Gene": ["A", "B", "C", "D"], "Log2FC": [3.0, -1.2, 5.0, 0.2],
                       "Adj_P_Value": [1e-6, 1e-8, 0.2, 0.001], "Significant": [True, True, False, False]})
    only = pd.DataFrame({"Gene": ["X"], "Present_in": ["Group B only"]})
    t = candidates.rank_candidates(de, only, "Normal", "Tumor")
    assert t.Protein.tolist() == ["A", "B", "X"] and t.Rank.tolist() == ["1", "2", "—"]
    assert t.loc[t.Protein == "A", "Evidence"].iloc[0] == "Higher in Tumor"
    assert t.loc[t.Protein == "B", "Evidence"].iloc[0] == "Higher in Normal"
    assert t.loc[t.Protein == "X", "Evidence"].iloc[0] == "Detected in Group B only"


def test_candidates_work_for_anova_results_and_uniprot_hand_off():
    anova = pd.DataFrame({"Gene": ["P1", "P2"], "F_Statistic": [9.0, 4.0], "P_Value": [1e-4, 1e-3],
                          "Adj_P_Value": [1e-4, 1e-3], "Max_Mean_Diff": [2.0, 1.0], "Significant": [True, True]})
    t = candidates.rank_candidates(anova)
    assert t.Protein.tolist() == ["P1", "P2"]
    dataset = pd.DataFrame({"Gene": ["P1", "P2"], "UniProt_ID": ["P00533", None]})
    withid = candidates.add_uniprot_ids(t, dataset, "Gene")
    out = candidates.target_list_for_docking(withid)
    assert out.UniProt_ID.tolist()[0] == "P00533" and list(out.columns)[:2] == ["Protein", "UniProt_ID"]
    assert candidates.add_uniprot_ids(t, pd.DataFrame({"Gene": ["P1"]}), "Gene")["UniProt_ID"].isna().all()


def test_no_significant_proteins_gives_an_empty_but_valid_table():
    de = pd.DataFrame({"Gene": ["A"], "Log2FC": [0.1], "Adj_P_Value": [0.9], "Significant": [False]})
    assert candidates.rank_candidates(de).empty
