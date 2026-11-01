"""
Turns statistics into a shortlist a researcher can act on: which proteins are
worth following up as drug targets or biomarkers, and why.
"""

import numpy as np
import pandas as pd


def rank_candidates(de_results, only_one_group=None, group_a_label="Group A", group_b_label="Group B", top=None):
    """
    de_results: output of DifferentialExpression.run() (two groups) or MultiGroupDE.run().
    Only significant proteins are ranked. Score = -log10(adjusted p) x |effect|,
    so a protein needs BOTH strong evidence and a real-sized change to rank high.
    only_one_group: optional presence/absence table; these proteins are appended,
    flagged as 'detected in one group only' (they cannot be t-tested).
    """
    rows = []
    sig = de_results[de_results["Significant"].fillna(False)] if "Significant" in de_results else de_results.iloc[0:0]

    for r in sig.itertuples(index=False):
        d = r._asdict()
        adj = d.get("Adj_P_Value", np.nan)
        evidence = -np.log10(max(adj, 1e-300)) if np.isfinite(adj) else 0.0
        if "Log2FC" in d:
            effect = abs(d["Log2FC"])
            direction = f"Higher in {group_b_label}" if d["Log2FC"] > 0 else f"Higher in {group_a_label}"
            fold = d["Log2FC"]
        else:
            effect = d.get("Max_Mean_Diff", 0.0)
            direction, fold = "Differs between groups", np.nan
        rows.append({"Protein": d["Gene"], "Evidence": direction, "Log2FC": fold,
                     "Adj_P_Value": adj, "Score": round(evidence * effect, 2)})

    table = pd.DataFrame(rows, columns=["Protein", "Evidence", "Log2FC", "Adj_P_Value", "Score"])
    table = table.sort_values("Score", ascending=False).reset_index(drop=True)
    if top:
        table = table.head(top)
    table.insert(0, "Rank", [str(i) for i in range(1, len(table) + 1)])      # text, so "—" can share the column

    if only_one_group is not None and len(only_one_group):
        extra = pd.DataFrame({
            "Rank": "—", "Protein": only_one_group.iloc[:, 0],
            "Evidence": "Detected in " + only_one_group["Present_in"].str.replace(" only", "") + " only",
            "Log2FC": np.nan, "Adj_P_Value": np.nan, "Score": np.nan})
        table = pd.concat([table, extra], ignore_index=True)
    return table


def add_uniprot_ids(table, dataset, protein_column, id_column="UniProt_ID"):
    """Attach UniProt IDs (present when the file came from MaxQuant / DIA-NN)."""
    if id_column not in dataset.columns:
        return table.assign(UniProt_ID=np.nan)
    mapping = dataset.set_index(protein_column)[id_column].to_dict()
    return table.assign(UniProt_ID=table["Protein"].map(mapping))


def target_list_for_docking(table):
    """Minimal hand-off table for the future Drug Discovery (docking) module."""
    keep = table[table["Rank"] != "—"] if "Rank" in table else table
    cols = [c for c in ("Protein", "UniProt_ID", "Evidence", "Log2FC", "Adj_P_Value", "Score") if c in keep.columns]
    return keep[cols].reset_index(drop=True)

# END OF FILE: modules/proteomics/candidates.pys