"""
Protein annotation from public databases (needs internet on the server):

  * UniProt   -- what the protein is, does, where it sits, known structures/drugs
  * STRING    -- which proteins it interacts with (confidence 0-1)
  * AlphaFold -- predicted 3D structure + confidence

Design: every function returns (result, error_message). Network failure,
unknown protein or a changed response never crash the page; they give a
readable message. All HTTP goes through _get() so tests can replace it.
"""

import re

import numpy as np
import pandas as pd

from modules.proteomics.parsers import first_accession

UNIPROT = "https://rest.uniprot.org/uniprotkb"
STRING = "https://string-db.org/api/json"
ALPHAFOLD = "https://alphafold.ebi.ac.uk/api/prediction"
PDBE = "https://www.ebi.ac.uk/pdbe/api"
RCSB_FILES = "https://files.rcsb.org/download"
RCSB_DATA = "https://data.rcsb.org/rest/v1/core"
HEADERS = {"User-Agent": "BioAI/1.0 (research tool)", "Accept": "application/json"}


# ------------------------------------------------------------------ HTTP
def _get(url, params=None, as_json=True, timeout=25):
    """Returns (data, error). The ONLY place that touches the network."""
    try:
        import requests

        response = requests.get(url, params=params, headers=HEADERS, timeout=timeout)
    except Exception as error:                              # noqa: BLE001
        return None, (
            f"Could not reach the database ({type(error).__name__}). "
            "Check the server's internet access and try again."
        )

    if response.status_code == 404:
        return None, "Not found in this database."
    if response.status_code == 429:
        return None, "The database is limiting requests (HTTP 429). Wait a minute and try again."
    if response.status_code >= 400:
        snippet = " ".join((response.text or "").split())[:160]
        return None, f"The database answered with an error (HTTP {response.status_code}): {snippet}"

    if not as_json:
        return response.text, None
    try:
        return response.json(), None
    except ValueError:
        return None, "The database returned a reply that could not be read."


# ------------------------------------------------------------------ UniProt
def looks_like_pdb_id(text):
    """PDB codes are 4 characters: a digit then 3 letters/digits (e.g. 7CR5, 1M17)."""
    return bool(re.fullmatch(r"[0-9][A-Za-z0-9]{3}", (text or "").strip()))


def uniprot_from_pdb(pdb_id):
    """PDB code -> list of UniProt accessions of the proteins in that structure (via PDBe/SIFTS)."""
    pdb_id = (pdb_id or "").strip().lower()                  # PDBe only accepts lowercase codes
    data, error = _get(f"{PDBE}/mappings/uniprot/{pdb_id}")
    if error:
        return [], error
    mapping = (data or {}).get(pdb_id, {}).get("UniProt", {})
    if not mapping:
        return [], "This PDB entry has no UniProt-mapped protein (for example it contains only DNA/RNA)."
    return list(mapping.keys()), None


def fetch_pdb_text(pdb_id, max_bytes=30_000_000):
    """Classic PDB-format file of an experimental structure from RCSB."""
    if not looks_like_pdb_id(pdb_id):
        return None, "A PDB code has 4 characters, for example 7CR5 or 1M17."
    text, error = _get(f"{RCSB_FILES}/{pdb_id.strip().upper()}.pdb", as_json=False, timeout=60)
    if error and "Not found" in error:
        return None, (
            "RCSB has no classic PDB-format file for this code. Very large structures are only "
            "distributed as mmCIF, which this tool cannot read yet."
        )
    if error:
        return None, error
    if len(text) > max_bytes:
        return None, "This structure file is too large to analyse here."
    return text, None


def resolve_uniprot(query, organism=9606):
    """UniProt accession, PDB code or gene symbol -> UniProt accession (reviewed human entry first)."""
    query = (query or "").strip()
    if not query:
        return None, "Enter a UniProt ID, PDB code or gene symbol."

    accession = first_accession(query)
    if accession and accession == query.split("-")[0]:
        return accession, None

    if looks_like_pdb_id(query):
        accessions, _ = uniprot_from_pdb(query)
        if accessions:
            return accessions[0], None

    data, error = _get(
        f"{UNIPROT}/search",
        {"query": f"(gene_exact:{query.upper()}) AND (organism_id:{organism}) AND (reviewed:true)",
         "format": "json", "size": 1, "fields": "accession"},
    )
    if error and "HTTP 400" in error:
        return None, (
            f"UniProt could not use '{query}' as a search. Type a gene symbol (for example EGFR), "
            "a UniProt ID (for example P00533) or a 4-character PDB code (for example 7CR5)."
        )
    if error:
        return None, error
    results = (data or {}).get("results", [])
    if not results:
        return None, f"No reviewed UniProt entry found for '{query}' (organism {organism})."
    return results[0]["primaryAccession"], None


def _comment_texts(entry, comment_type):
    out = []
    for comment in entry.get("comments", []):
        if comment.get("commentType") == comment_type:
            out += [t.get("value", "") for t in comment.get("texts", []) if t.get("value")]
    return out


def parse_uniprot(entry):
    """UniProt JSON entry -> flat dictionary the page can show."""
    description = entry.get("proteinDescription", {})
    name = (
        description.get("recommendedName", {}).get("fullName", {}).get("value")
        or description.get("submissionNames", [{}])[0].get("fullName", {}).get("value", "")
    )
    genes = entry.get("genes", [])
    gene = genes[0].get("geneName", {}).get("value", "") if genes else ""

    locations = []
    for comment in entry.get("comments", []):
        if comment.get("commentType") == "SUBCELLULAR LOCATION":
            for loc in comment.get("subcellularLocations", []):
                value = loc.get("location", {}).get("value")
                if value and value not in locations:
                    locations.append(value)

    diseases = []
    for comment in entry.get("comments", []):
        if comment.get("commentType") == "DISEASE":
            disease = comment.get("disease", {})
            if disease.get("diseaseId"):
                diseases.append(disease["diseaseId"])

    domains, ptm = [], {}
    for feature in entry.get("features", []):
        kind = feature.get("type", "")
        if kind in ("Domain", "Zinc finger", "DNA binding", "Motif"):
            loc = feature.get("location", {})
            domains.append({
                "Type": kind, "Name": feature.get("description", ""),
                "Start": loc.get("start", {}).get("value"), "End": loc.get("end", {}).get("value"),
            })
        if kind in ("Modified residue", "Glycosylation", "Disulfide bond", "Cross-link", "Lipidation"):
            ptm[kind] = ptm.get(kind, 0) + 1

    structures, drug_links = [], {}
    for ref in entry.get("uniProtKBCrossReferences", []):
        database = ref.get("database")
        if database == "PDB":
            props = {p.get("key"): p.get("value") for p in ref.get("properties", [])}
            structures.append({"PDB_ID": ref.get("id"), "Method": props.get("Method", ""),
                               "Resolution": props.get("Resolution", ""), "Chains": props.get("Chains", "")})
        elif database in ("ChEMBL", "DrugBank", "DrugCentral", "GuidetoPHARMACOLOGY", "BindingDB"):
            drug_links.setdefault(database, []).append(ref.get("id"))

    sequence = entry.get("sequence", {})
    return {
        "accession": entry.get("primaryAccession", ""),
        "entry_name": entry.get("uniProtkbId", ""),
        "reviewed": "reviewed" in str(entry.get("entryType", "")).lower() and "unreviewed" not in str(entry.get("entryType", "")).lower(),
        "protein_name": name,
        "gene": gene,
        "organism": entry.get("organism", {}).get("scientificName", ""),
        "length": sequence.get("length"),
        "mass_da": sequence.get("molWeight"),
        "function": " ".join(_comment_texts(entry, "FUNCTION")),
        "locations": locations,
        "diseases": diseases,
        "domains": domains,
        "ptm_counts": ptm,
        "structures": structures,
        "drug_links": drug_links,
        "keywords": [k.get("name", "") for k in entry.get("keywords", [])],
    }


def fetch_uniprot_profile(accession):
    entry, error = _get(f"{UNIPROT}/{accession}.json")
    if error:
        return None, error
    try:
        return parse_uniprot(entry), None
    except Exception:                                       # noqa: BLE001
        return None, "UniProt returned an entry in an unexpected format."


def druggability_indicators(profile):
    """
    Plain facts that matter when choosing a target. These are INDICATORS from
    database cross-references, not a prediction of druggability.
    """
    rows = []
    n_struct = len(profile["structures"])
    xray = [s for s in profile["structures"] if "x-ray" in s["Method"].lower()]
    best = None
    for s in profile["structures"]:
        match = re.search(r"([\d.]+)", s.get("Resolution", "") or "")
        if match:
            value = float(match.group(1))
            best = value if best is None else min(best, value)
    rows.append(("Experimental 3D structures (PDB)", f"{n_struct}" + (f" ({len(xray)} X-ray, best resolution {best} Å)" if best else "")))
    for database in ("ChEMBL", "DrugBank", "DrugCentral", "GuidetoPHARMACOLOGY", "BindingDB"):
        ids = profile["drug_links"].get(database)
        rows.append((f"{database} entry", ", ".join(ids[:3]) + (" …" if len(ids) > 3 else "") if ids else "none listed"))
    keywords = {k.lower() for k in profile["keywords"]}
    for label, word in (("Kinase", "kinase"), ("Receptor", "receptor"), ("Protease", "protease"),
                        ("Membrane protein", "membrane"), ("Secreted", "secreted"), ("Transcription factor", "transcription regulation")):
        if any(word in k for k in keywords):
            rows.append(("Protein class (keyword)", label))
    rows.append(("Annotated disease links", ", ".join(profile["diseases"][:4]) if profile["diseases"] else "none listed"))
    return rows


# ------------------------------------------------------------------ STRING
def fetch_string_network(genes, species=9606, min_score=700, add_nodes=0):
    """Interactions among (and around) the given gene symbols. score 0-1."""
    genes = [g for g in dict.fromkeys(genes) if g]
    if not genes:
        return None, "No proteins given."
    params = {"identifiers": "\r".join(genes), "species": species,
              "required_score": int(min_score), "caller_identity": "BioAI"}
    if add_nodes:
        params["add_nodes"] = int(add_nodes)
    data, error = _get(f"{STRING}/network", params)
    if error:
        return None, error
    edges = parse_string_edges(data)
    if edges.empty:
        return edges, "No interactions at this confidence. Lower the minimum score."
    return edges, None


def parse_string_edges(data):
    rows = []
    for item in data or []:
        a, b = item.get("preferredName_A"), item.get("preferredName_B")
        if a and b and a != b:
            rows.append({"Protein_A": a, "Protein_B": b, "Score": float(item.get("score", 0)),
                         "Experimental": float(item.get("escore", np.nan)) if "escore" in item else np.nan,
                         "Database": float(item.get("dscore", np.nan)) if "dscore" in item else np.nan,
                         "Textmining": float(item.get("tscore", np.nan)) if "tscore" in item else np.nan})
    edges = pd.DataFrame(rows, columns=["Protein_A", "Protein_B", "Score", "Experimental", "Database", "Textmining"])
    if edges.empty:
        return edges
    key = edges.apply(lambda r: tuple(sorted((r.Protein_A, r.Protein_B))), axis=1)
    return edges.assign(_k=key).drop_duplicates("_k").drop(columns="_k").reset_index(drop=True)


def network_metrics(edges, own_proteins=()):
    """
    Per protein: number of partners (degree), summed confidence, and a hub flag
    (degree clearly above the average). `own_proteins` marks those that are in the user's data.
    """
    if edges.empty:
        return pd.DataFrame(columns=["Protein", "Partners", "Summed_score", "Hub", "In_your_data"])
    long = pd.concat([
        edges[["Protein_A", "Score"]].rename(columns={"Protein_A": "Protein"}),
        edges[["Protein_B", "Score"]].rename(columns={"Protein_B": "Protein"}),
    ])
    out = long.groupby("Protein").agg(Partners=("Score", "size"), Summed_score=("Score", "sum")).reset_index()
    cutoff = out["Partners"].mean() + out["Partners"].std(ddof=0)
    out["Hub"] = (out["Partners"] >= max(cutoff, 3))
    own = set(own_proteins)
    out["In_your_data"] = out["Protein"].isin(own)
    out["Summed_score"] = out["Summed_score"].round(2)
    return out.sort_values(["Partners", "Summed_score"], ascending=False).reset_index(drop=True)


def spring_layout(nodes, edges, iterations=200, seed=0):
    """Small Fruchterman-Reingold layout (no extra library). Returns {node: (x, y)}."""
    rng = np.random.default_rng(seed)
    n = len(nodes)
    if n == 0:
        return {}
    index = {name: i for i, name in enumerate(nodes)}
    pos = rng.uniform(-1, 1, (n, 2))
    k = np.sqrt(1.0 / n)
    pairs = [(index[a], index[b]) for a, b in edges if a in index and b in index]
    temperature = 0.1
    for _ in range(iterations):
        delta = pos[:, None, :] - pos[None, :, :]
        dist = np.linalg.norm(delta, axis=2) + 1e-9
        force = (k * k / dist ** 2)[:, :, None] * delta
        disp = force.sum(axis=1)
        for i, j in pairs:
            d = pos[i] - pos[j]
            length = np.linalg.norm(d) + 1e-9
            pull = d * (length / k)
            disp[i] -= pull
            disp[j] += pull
        length = np.linalg.norm(disp, axis=1)[:, None] + 1e-9
        pos += disp / length * np.minimum(length, temperature)
        temperature *= 0.98
    pos -= pos.mean(axis=0)
    scale = np.abs(pos).max() or 1.0
    pos /= scale
    return {name: (float(pos[i, 0]), float(pos[i, 1])) for name, i in index.items()}


def network_figure(edges, metrics, focus=None, log2fc=None):
    """Plotly network. Node size = partners; red outline = in your data; colour = log2FC if given."""
    import plotly.graph_objects as go

    nodes = list(metrics["Protein"])
    pos = spring_layout(nodes, list(zip(edges.Protein_A, edges.Protein_B)))
    fig = go.Figure()

    xs, ys = [], []
    for a, b in zip(edges.Protein_A, edges.Protein_B):
        xs += [pos[a][0], pos[b][0], None]
        ys += [pos[a][1], pos[b][1], None]
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", hoverinfo="none",
                             line=dict(width=1, color="rgba(120,120,120,0.45)"), showlegend=False))

    sizes = 12 + 4 * metrics["Partners"].clip(upper=15)
    colors = [(log2fc or {}).get(p, np.nan) for p in nodes]
    have_color = any(np.isfinite(c) for c in colors)
    fig.add_trace(go.Scatter(
        x=[pos[p][0] for p in nodes], y=[pos[p][1] for p in nodes],
        mode="markers+text", text=nodes, textposition="top center",
        marker=dict(size=sizes,
                    color=[c if np.isfinite(c) else 0 for c in colors] if have_color else "#0072B2",
                    colorscale="RdBu_r", cmid=0, showscale=have_color,
                    colorbar=dict(title="log2FC") if have_color else None,
                    line=dict(width=[3 if (p == focus or d) else 0.5 for p, d in zip(nodes, metrics["In_your_data"])],
                              color=["black" if p == focus else "crimson" for p in nodes])),
        hovertext=[f"{p}: {n} partners" for p, n in zip(nodes, metrics["Partners"])], hoverinfo="text",
        showlegend=False))
    fig.update_layout(xaxis=dict(visible=False), yaxis=dict(visible=False), plot_bgcolor="white",
                      height=620, margin=dict(l=10, r=10, t=30, b=10))
    return fig


# ------------------------------------------------------------------ AlphaFold
def fetch_alphafold(accession):
    data, error = _get(f"{ALPHAFOLD}/{accession}")
    if error:
        return None, "No AlphaFold model found for this protein." if "Not found" in error else error
    if not data:
        return None, "No AlphaFold model found for this protein."
    item = data[0]
    return {
        "entry_id": item.get("entryId"), "pdb_url": item.get("pdbUrl"), "cif_url": item.get("cifUrl"),
        "mean_plddt": item.get("globalMetricValue"), "version": item.get("latestVersion"),
        "gene": item.get("gene"), "pae_image": item.get("paeImageUrl"),
    }, None


def download_structure_text(url, max_bytes=6_000_000):
    """PDB text for the viewer. Fetched by the server (avoids browser cross-site limits)."""
    text, error = _get(url, as_json=False, timeout=40)
    if error:
        return None, error
    if len(text) > max_bytes:
        return None, "This structure file is too large to display here; download it from the link instead."
    return text, None

# END OF FILE: modules/proteomics/annotation.py