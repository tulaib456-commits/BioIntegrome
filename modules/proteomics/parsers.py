"""
Readers for the files proteomics software really produces.

Supported: MaxQuant proteinGroups.txt, DIA-NN report.pg_matrix.tsv, and any
generic table (proteins as rows, samples as columns).

Every reader returns (table, info):
  table  columns: Gene | UniProt_ID | one numeric column per sample (log2 intensity)
  info   what was done (rows removed, which intensity was used, ...) so the
         user can see it -- nothing is changed silently.
"""

import os
import re

import numpy as np
import pandas as pd

from modules.figure_studio.data_loader import load_file

# official UniProt accession pattern
_ACCESSION = re.compile(
    r"[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2}"
)


def first_accession(text):
    """First UniProt accession inside strings like 'sp|P00533|EGFR_HUMAN;Q9XXXX-2'."""
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return None
    match = _ACCESSION.search(str(text))
    return match.group(0) if match else None


def first_item(text):
    """'EGFR;ERBB1' -> 'EGFR' ; empty/NaN -> None."""
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return None
    first = str(text).split(";")[0].strip()
    return first or None


def detect_format(df):
    columns = set(df.columns)
    if "Protein IDs" in columns and any(
        c.startswith(("LFQ intensity ", "Intensity ", "iBAQ ")) for c in columns
    ):
        return "maxquant"
    if "Protein.Group" in columns or ("Protein.Ids" in columns and "Genes" in columns):
        return "diann"
    return "generic"


def looks_unlogged(df, samples):
    """True when the numbers look like raw intensities (e.g. 2.3e7) rather than log2 (~15-35)."""
    values = df[samples].to_numpy(dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    return bool(values.size and np.median(values) > 100)


def to_log2(df, samples):
    """log2 of raw intensities; 0 and negative values become missing (that is what 0 means here)."""
    out = df.copy()
    values = out[samples].astype(float).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        logged = np.where(values > 0, np.log2(values), np.nan)
    out[samples] = logged
    return out


def _unique_names(names):
    seen, result = {}, []
    for name in names:
        if name in seen:
            seen[name] += 1
            result.append(f"{name}_{seen[name]}")
        else:
            seen[name] = 0
            result.append(name)
    return result


def _identifier_column(gene_series, accession_series):
    """Gene symbol where known, else accession; duplicates made unique with the accession."""
    ids = [g if g else a for g, a in zip(gene_series, accession_series)]
    ids = [i if i else f"unknown_{n + 1}" for n, i in enumerate(ids)]
    counts = pd.Series(ids).value_counts()
    return [
        f"{i} ({a})" if counts[i] > 1 and a else i
        for i, a in zip(ids, accession_series)
    ]


def parse_maxquant(df, intensity="auto"):
    """
    MaxQuant proteinGroups.txt.
    intensity: 'auto' (LFQ if present, else Intensity), 'LFQ intensity', 'iBAQ', 'Intensity'.
    Removes reverse hits, potential contaminants and 'only identified by site'.
    """
    info = {"format": "MaxQuant proteinGroups.txt", "rows_in": len(df)}

    prefixes = ["LFQ intensity ", "iBAQ ", "Intensity "]
    if intensity != "auto":
        prefixes = [intensity.rstrip() + " "]
    chosen = None
    for prefix in prefixes:
        sample_cols = [c for c in df.columns if c.startswith(prefix)]
        # 'Intensity ' also matches 'Intensity L ...' (SILAC) -- keep the plain ones
        if prefix == "Intensity ":
            sample_cols = [c for c in sample_cols if not re.match(r"Intensity [HLM] ", c)]
        if sample_cols:
            chosen = prefix
            break
    if chosen is None:
        raise ValueError(
            "No intensity columns found (looked for 'LFQ intensity ...', 'iBAQ ...', 'Intensity ...')."
        )
    info["intensity_used"] = chosen.strip()

    keep = pd.Series(True, index=df.index)
    for flag, label in (
        ("Reverse", "reverse hits"),
        ("Potential contaminant", "potential contaminants"),
        ("Only identified by site", "only identified by site"),
    ):
        if flag in df.columns:
            marked = df[flag].astype(str).str.strip().eq("+")
            info[f"removed_{label.replace(' ', '_')}"] = int(marked.sum())
            keep &= ~marked
    df = df[keep].copy()

    samples = [c[len(chosen):].strip() for c in sample_cols]
    values = df[sample_cols].apply(pd.to_numeric, errors="coerce")
    values.columns = _unique_names(samples)

    accessions = [first_accession(x) for x in df.get("Majority protein IDs", df["Protein IDs"])]
    genes = [first_item(x) for x in df["Gene names"]] if "Gene names" in df.columns else [None] * len(df)

    table = pd.DataFrame({
        "Gene": _identifier_column(genes, accessions),
        "UniProt_ID": accessions,
    })
    table = pd.concat([table.reset_index(drop=True), values.reset_index(drop=True)], axis=1)
    return _finish(table, list(values.columns), info)


def parse_diann(df):
    """DIA-NN report.pg_matrix.tsv (protein groups x runs)."""
    info = {"format": "DIA-NN protein group matrix", "rows_in": len(df)}

    annotation = {"Protein.Group", "Protein.Ids", "Protein.Names", "Genes", "First.Protein.Description"}
    sample_cols = [c for c in df.columns if c not in annotation and pd.api.types.is_numeric_dtype(df[c])]
    if not sample_cols:
        raise ValueError("No numeric run columns found in this DIA-NN file.")

    nice = []
    for c in sample_cols:
        base = os.path.basename(str(c).replace("\\", "/"))
        nice.append(re.sub(r"\.(raw|d|mzML|wiff|dia)$", "", base, flags=re.I))
    nice = _unique_names(nice)

    values = df[sample_cols].apply(pd.to_numeric, errors="coerce")
    values.columns = nice

    group_col = "Protein.Group" if "Protein.Group" in df.columns else "Protein.Ids"
    accessions = [first_accession(x) for x in df[group_col]]
    genes = [first_item(x) for x in df["Genes"]] if "Genes" in df.columns else [None] * len(df)

    table = pd.DataFrame({
        "Gene": _identifier_column(genes, accessions),
        "UniProt_ID": accessions,
    })
    table = pd.concat([table.reset_index(drop=True), values.reset_index(drop=True)], axis=1)
    return _finish(table, nice, info)


def _finish(table, samples, info):
    """Shared last step: zero -> missing, log2, drop proteins never measured."""
    table = to_log2(table, samples)
    before = len(table)
    table = table[table[samples].notna().any(axis=1)].reset_index(drop=True)
    info["removed_never_measured"] = before - len(table)
    info["proteins"] = len(table)
    info["samples"] = len(samples)
    info["transform"] = "log2 (zeros treated as missing)"
    return table, info


def load_proteomics_file(uploaded_file, sheet_name=None, intensity="auto"):
    """
    One call for the page: read the file (any CSV/TSV/TXT/XLSX), detect which
    software made it, and return (table, info).
    """
    raw = load_file(uploaded_file, sheet_name=sheet_name)
    kind = detect_format(raw)

    if kind == "maxquant":
        table, info = parse_maxquant(raw, intensity=intensity)
    elif kind == "diann":
        table, info = parse_diann(raw)
    else:
        table, info = raw, {"format": "generic table", "proteins": len(raw)}

    info["notes"] = raw.attrs.get("notes", [])
    return table, info

# END OF FILE: modules/proteomics/parsers.py