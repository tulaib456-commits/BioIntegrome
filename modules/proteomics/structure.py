"""
Structure tools for researchers who have a 3D structure (X-ray, cryo-EM, NMR or AlphaFold):

  * header_quality()   resolution, R-work, R-free from the PDB file header
  * bfactor_table()    mean B-factor per residue (flexibility / disorder)
  * dihedral_table()   backbone phi/psi angles, computed with numpy
  * ramachandran_summary()  APPROXIMATE favoured / allowed / outlier split
  * viewer_html()      interactive 3D viewer (3Dmol.js) as one HTML string

Only the classic PDB text format is read (not mmCIF). The Ramachandran regions
are simple boxes meant for a quick screen -- use MolProbity or PROCHECK for
publication-grade validation.
"""

import json
import re

import numpy as np
import pandas as pd

THREEDMOL_VERSION = "2.5.5"
THREEDMOL_PRIMARY = f"https://cdn.jsdelivr.net/npm/3dmol@{THREEDMOL_VERSION}/build/3Dmol-min.js"
THREEDMOL_BACKUP = "https://3Dmol.org/build/3Dmol-min.js"

WATER = {"HOH", "WAT", "H2O", "DOD"}


# ------------------------------------------------------------------ reading
def parse_pdb_text(text):
    """ATOM/HETATM records -> DataFrame (first model only)."""
    rows = []
    for line in text.splitlines():
        record = line[:6].strip()
        if record == "ENDMDL":
            break
        if record not in ("ATOM", "HETATM") or len(line) < 54:
            continue
        try:
            rows.append({
                "record": record, "name": line[12:16].strip(), "altloc": line[16].strip(),
                "resname": line[17:20].strip(), "chain": line[21].strip() or "A",
                "resseq": int(line[22:26]), "icode": line[26].strip(),
                "x": float(line[30:38]), "y": float(line[38:46]), "z": float(line[46:54]),
                "occupancy": float(line[54:60]) if line[54:60].strip() else 1.0,
                "bfactor": float(line[60:66]) if line[60:66].strip() else np.nan,
            })
        except ValueError:
            continue
    atoms = pd.DataFrame(rows)
    if atoms.empty:
        raise ValueError("No atoms found. Upload a PDB-format file (.pdb / .ent). mmCIF is not supported here.")
    # keep only the first alternate location
    first_alt = atoms["altloc"].replace("", np.nan).dropna()
    if not first_alt.empty:
        atoms = atoms[(atoms["altloc"] == "") | (atoms["altloc"] == first_alt.iloc[0])]
    return atoms.reset_index(drop=True)


def header_quality(text):
    """Resolution, R-work, R-free, method from REMARK/EXPDTA lines (None when absent)."""
    info = {"method": None, "resolution": None, "r_work": None, "r_free": None}
    for line in text.splitlines():
        if line.startswith("EXPDTA") and info["method"] is None:
            info["method"] = line[6:].strip()
        elif line.startswith("REMARK   2 RESOLUTION") and info["resolution"] is None:
            match = re.search(r"RESOLUTION\.\s+([\d.]+)\s+ANGSTROM", line)
            if match:
                info["resolution"] = float(match.group(1))
        elif line.startswith("REMARK   3") and "R VALUE" in line and "WORKING SET" in line and info["r_work"] is None:
            match = re.search(r":\s*([\d.]+)\s*$", line)
            if match:
                info["r_work"] = float(match.group(1))
        elif line.startswith("REMARK   3") and "FREE R VALUE" in line and "TEST SET" not in line and info["r_free"] is None:
            match = re.search(r":\s*([\d.]+)\s*$", line)
            if match:
                info["r_free"] = float(match.group(1))
    return info


def interpret_quality(info):
    """Plain-language reading of the header numbers (common rules of thumb)."""
    notes = []
    res = info.get("resolution")
    if res is not None:
        if res <= 2.0:
            notes.append(f"Resolution {res} Å: high quality; side chains and waters are usually well defined (good for docking).")
        elif res <= 3.0:
            notes.append(f"Resolution {res} Å: medium quality; use side-chain/water positions with care.")
        else:
            notes.append(f"Resolution {res} Å: low; fine details are uncertain, so docking results need extra caution.")
    rw, rf = info.get("r_work"), info.get("r_free")
    if rf is not None:
        notes.append(
            f"R-free {rf:.3f}: " + ("typical for a well-refined structure." if rf <= 0.25 else
                                      "acceptable, but check how well the model fits." if rf <= 0.30 else
                                      "high; the model may not fit the data well.")
        )
    if rw is not None and rf is not None:
        gap = rf - rw
        notes.append(
            f"R-free minus R-work = {gap:.3f}: " + ("small, no sign of over-fitting." if gap <= 0.05 else
                                                      "somewhat large; a common rule of thumb is that a gap above ~0.05-0.07 can mean over-fitting.")
        )
    if not notes:
        notes.append("No resolution or R-values found in the header (normal for AlphaFold models, NMR and many cryo-EM files).")
    return notes


def composition(atoms):
    protein = atoms[atoms.record == "ATOM"]
    hetero = atoms[(atoms.record == "HETATM")]
    ligands = sorted(set(hetero.resname) - WATER)
    residues = protein.drop_duplicates(["chain", "resseq", "icode"])
    return {
        "chains": sorted(set(protein.chain)),
        "residues": int(len(residues)),
        "waters": int(hetero[hetero.resname.isin(WATER)].drop_duplicates(["chain", "resseq"]).shape[0]),
        "ligands": ligands,
    }


# ------------------------------------------------------------------ B-factors
def bfactor_table(atoms):
    protein = atoms[atoms.record == "ATOM"]
    grouped = protein.groupby(["chain", "resseq", "icode", "resname"], sort=False)["bfactor"].mean().reset_index()
    grouped = grouped.rename(columns={"bfactor": "Mean_B"})
    grouped["Residue_number"] = range(1, len(grouped) + 1)
    return grouped


def bfactor_figure(table):
    import plotly.graph_objects as go

    fig = go.Figure()
    for chain, sub in table.groupby("chain", sort=False):
        fig.add_trace(go.Scatter(x=sub["resseq"], y=sub["Mean_B"], mode="lines", name=f"Chain {chain}"))
    fig.update_layout(xaxis_title="Residue number", yaxis_title="Mean B-factor (Å²)", height=380,
                      plot_bgcolor="white", margin=dict(l=10, r=10, t=30, b=10))
    return fig


# ------------------------------------------------------------------ dihedral angles
def _dihedral(p0, p1, p2, p3):
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1 = b1 / np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    return float(np.degrees(np.arctan2(y, x)))


def dihedral_table(atoms):
    """phi/psi per residue (degrees) where the chain is continuous."""
    protein = atoms[atoms.record == "ATOM"]
    rows = []
    for chain, sub in protein.groupby("chain", sort=False):
        residues = []
        for (resseq, icode), res in sub.groupby(["resseq", "icode"], sort=False):
            coords = {r.name: np.array([r.x, r.y, r.z]) for r in res.itertuples()}
            if all(k in coords for k in ("N", "CA", "C")):
                residues.append((resseq, icode, res.resname.iloc[0], coords))
        for i in range(1, len(residues) - 1):
            prev, cur, nxt = residues[i - 1], residues[i], residues[i + 1]
            joined_before = np.linalg.norm(prev[3]["C"] - cur[3]["N"]) < 2.0
            joined_after = np.linalg.norm(cur[3]["C"] - nxt[3]["N"]) < 2.0
            if not (joined_before and joined_after):
                continue
            phi = _dihedral(prev[3]["C"], cur[3]["N"], cur[3]["CA"], cur[3]["C"])
            psi = _dihedral(cur[3]["N"], cur[3]["CA"], cur[3]["C"], nxt[3]["N"])
            rows.append({"chain": chain, "resseq": cur[0], "resname": cur[2], "phi": phi, "psi": psi})
    return pd.DataFrame(rows, columns=["chain", "resseq", "resname", "phi", "psi"])


def _in_box(phi, psi, boxes):
    return any(p0 <= phi <= p1 and s0 <= psi <= s1 for p0, p1, s0, s1 in boxes)


# (phi_min, phi_max, psi_min, psi_max)  -- coarse, general-case regions
FAVOURED = [(-100, -35, -70, -5),          # right-handed alpha helix
            (-180, -45, 100, 180),         # beta strand / PPII
            (-180, -45, -180, -170)]       # beta strand (wrap-around)
ALLOWED = [(-180, -20, -110, 50),
           (-180, -40, 80, 180),
           (-180, -40, -180, -150),
           (30, 100, -20, 100)]            # left-handed helix


def classify_ramachandran(table):
    """Adds a 'region' column: Favoured / Allowed / Outlier (glycine and proline are not judged)."""
    table = table.copy()
    labels = []
    for r in table.itertuples():
        if r.resname == "GLY":
            labels.append("Glycine (not judged)")
        elif _in_box(r.phi, r.psi, FAVOURED):
            labels.append("Favoured")
        elif _in_box(r.phi, r.psi, ALLOWED):
            labels.append("Allowed")
        else:
            labels.append("Outlier")
    table["region"] = labels
    return table


def ramachandran_summary(classified):
    judged = classified[classified.region != "Glycine (not judged)"]
    n = len(judged)
    out = {"residues_judged": n}
    for label in ("Favoured", "Allowed", "Outlier"):
        count = int((judged.region == label).sum())
        out[label] = count
        out[label + "_pct"] = round(100 * count / n, 1) if n else None
    return out


def ramachandran_figure(classified):
    import plotly.express as px

    colors = {"Favoured": "#2e7d32", "Allowed": "#f9a825", "Outlier": "#c62828", "Glycine (not judged)": "#9e9e9e"}
    fig = px.scatter(classified, x="phi", y="psi", color="region", color_discrete_map=colors,
                     hover_data=["chain", "resseq", "resname"])
    fig.update_traces(marker=dict(size=6, opacity=0.8))
    for x0, x1, y0, y1 in FAVOURED:
        fig.add_shape(type="rect", x0=x0, x1=x1, y0=y0, y1=y1, line=dict(color="rgba(46,125,50,0.5)", width=1),
                      fillcolor="rgba(46,125,50,0.08)", layer="below")
    fig.update_layout(xaxis=dict(range=[-180, 180], title="phi (°)", dtick=60),
                      yaxis=dict(range=[-180, 180], title="psi (°)", dtick=60, scaleanchor="x"),
                      height=520, plot_bgcolor="white", margin=dict(l=10, r=10, t=30, b=10))
    return fig


# ------------------------------------------------------------------ 3D viewer
_VIEWER = """<!DOCTYPE html><html><head><meta charset="utf-8">
<style>html,body{margin:0;padding:0;font-family:Arial,sans-serif}
#v{width:100%;height:__HEIGHT__px;position:relative;border:1px solid #ccc}
#err{display:none;padding:12px;border:1px solid #d32f2f;color:#b71c1c;background:#fdecea;border-radius:6px;font-size:14px}</style>
</head><body>
<div id="err">The 3D viewer library could not be loaded (internet blocked or slow). Download the structure file and open it in PyMOL, ChimeraX or Mol* instead.</div>
<div id="v"></div>
<script>
var PDB = __PDB__; var COLOR_BY = __COLOR__; var PRIMARY = __PRIMARY__; var BACKUP = __BACKUP__;
function plddt(a){var b=a.b; return b>90?"#0053D6":b>70?"#65CBF3":b>50?"#FFDB13":"#FF7D45";}
function start(){
  var viewer = $3Dmol.createViewer("v", {backgroundColor:"white"});
  viewer.addModel(PDB, "pdb");
  var cartoon = {};
  if (COLOR_BY === "plddt") { cartoon = {colorfunc: plddt}; }
  else if (COLOR_BY === "bfactor") { cartoon = {colorscheme: {prop:"b", gradient:"roygb", min:100, max:10}}; }
  else if (COLOR_BY === "chain") { cartoon = {colorscheme: "chain"}; }
  else { cartoon = {color: "spectrum"}; }
  viewer.setStyle({}, {cartoon: cartoon});
  viewer.setStyle({hetflag:true, not:{resn:["HOH","WAT"]}}, {stick:{colorscheme:"greenCarbon", radius:0.2}});
  viewer.zoomTo(); viewer.render();
}
function load(src, ok, fail){var s=document.createElement("script"); s.src=src; s.onload=ok; s.onerror=fail; document.head.appendChild(s);}
load(PRIMARY, start, function(){ load(BACKUP, start, function(){
  document.getElementById("err").style.display="block"; document.getElementById("v").style.display="none"; }); });
</script></body></html>"""


def viewer_html(pdb_text, color_by="spectrum", height=520):
    """color_by: 'plddt' (AlphaFold confidence), 'bfactor', 'chain' or 'spectrum'."""
    safe = json.dumps(pdb_text).replace("<", "\\u003c")
    return (_VIEWER.replace("__PDB__", safe)
            .replace("__COLOR__", json.dumps(color_by))
            .replace("__HEIGHT__", str(int(height)))
            .replace("__PRIMARY__", json.dumps(THREEDMOL_PRIMARY))
            .replace("__BACKUP__", json.dumps(THREEDMOL_BACKUP)))

# END OF FILE: modules/proteomics/structure.py