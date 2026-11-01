import numpy as np
import pandas as pd
import streamlit as st

from modules.auth import require_active_plan
from modules.figure_studio.data_loader import list_sheets, load_file
from modules.figure_studio.export_component import _show_html
from modules.proteomics import annotation as ann
from modules.proteomics import candidates as cand
from modules.proteomics import dose_response as dr
from modules.proteomics import structure as stc
from modules.proteomics.missing_value_handler import presence_absence_table
from modules.proteomics.parsers import first_accession
from modules.transcriptomics.result_display import display_results_table
from modules.transcriptomics.validator import detect_gene_column, detect_sample_columns

require_active_plan()

st.title("🔬 Proteomics Discovery")
st.caption(
    "From a list of changed proteins to a drug-target decision: shortlist → what the protein "
    "is → what it interacts with → its 3D structure → how your compound affects it."
)

# ------------------------------------------------------------------ what the Proteomics page produced
de_results = st.session_state.get("proteomics_de_results")
dataset = st.session_state.get("proteomics_dataset")
normalized = st.session_state.get("proteomics_normalized")
de_groups = st.session_state.get("proteomics_de_groups") if st.session_state.get("proteomics_de_kind") == "two_group" else None

shortlist = None
log2fc = {}
significant_proteins = []

if de_results is not None:
    labels = ("Group A", "Group B")
    only_one = None
    if de_groups and normalized is not None:
        try:
            only_one = presence_absence_table(
                normalized, detect_gene_column(normalized), de_groups[0], de_groups[1]
            )
        except Exception:                                   # noqa: BLE001
            only_one = None
    shortlist = cand.rank_candidates(de_results, only_one, *labels)
    if dataset is not None:
        shortlist = cand.add_uniprot_ids(shortlist, dataset, detect_gene_column(dataset))
    significant_proteins = [p for p in shortlist["Protein"].tolist()]
    if "Log2FC" in de_results.columns:
        log2fc = dict(zip(de_results["Gene"], de_results["Log2FC"]))

# ------------------------------------------------------------------ choose a protein
st.subheader("Protein to investigate")
col_pick, col_type = st.columns([2, 1])
with col_pick:
    picked = st.selectbox(
        "From your shortlist", ["(none)"] + significant_proteins, key="disc_pick",
        help="Filled after you run Differential Expression on the Proteomics page."
    )
with col_type:
    typed = st.text_input("…or type a UniProt ID / gene symbol", key="disc_typed", placeholder="e.g. EGFR or P00533")

query = typed.strip() or ("" if picked == "(none)" else picked)

known_id = None
if query and shortlist is not None and "UniProt_ID" in shortlist.columns:
    hit = shortlist.loc[shortlist["Protein"] == query, "UniProt_ID"].dropna()
    known_id = hit.iloc[0] if len(hit) else None
query_for_lookup = known_id or first_accession(query) or query.split(" (")[0].strip()


def resolve():
    """UniProt accession for the chosen protein (cached per protein)."""
    cache = st.session_state.setdefault("disc_accession", {})
    if query_for_lookup not in cache:
        cache[query_for_lookup] = ann.resolve_uniprot(query_for_lookup)
    return cache[query_for_lookup]


tabs = st.tabs([
    "1. Shortlist", "2. Protein profile", "3. Interactions", "4. 3D structure (AlphaFold)",
    "5. My structure (X-ray / PDB)", "6. Drug response (IC50)"
])

# ================================================================== 1. shortlist
with tabs[0]:
    st.header("Candidate targets from your experiment")
    if shortlist is None:
        st.info("Run Differential Expression on the Proteomics page first. Then your ranked shortlist appears here.")
    else:
        st.caption(
            "Score = −log10(adjusted p) × |log2 fold-change|: a protein ranks high only if it has "
            "BOTH strong statistical evidence AND a real-sized change. Proteins detected in only one "
            "group are listed at the end; they cannot be t-tested, but are often the most interesting."
        )
        top_n = st.slider("Show top", 5, 200, 25, key="disc_top")
        view = shortlist[shortlist["Rank"] == "—"]
        ranked = shortlist[shortlist["Rank"] != "—"].head(top_n)
        display_results_table(pd.concat([ranked, view]), "disc_shortlist", default_index=2)

        handoff = cand.target_list_for_docking(shortlist)
        st.download_button(
            "Download target list (CSV) — for docking / Drug Discovery",
            handoff.to_csv(index=False).encode("utf-8"), file_name="target_list.csv",
            mime="text/csv", key="disc_handoff"
        )
        if "UniProt_ID" in handoff.columns and handoff["UniProt_ID"].isna().all():
            st.caption(
                "UniProt IDs are filled automatically when your file came from MaxQuant or DIA-NN. "
                "Otherwise use tab 2 to look a protein up."
            )
        st.info(
            "The Drug Discovery module (docking, QSAR, ADMET) is planned. This CSV is the hand-off: "
            "protein, UniProt ID, evidence and score."
        )

# ================================================================== 2. protein profile
with tabs[1]:
    st.header("What is this protein?")
    if not query:
        st.info("Choose a protein from your shortlist or type a UniProt ID / gene symbol above.")
    else:
        if st.button(f"Look up {query} in UniProt", key="disc_profile_btn"):
            accession, error = resolve()
            if error:
                st.session_state["disc_profile_result"] = (query, None, error)
            else:
                profile, error = ann.fetch_uniprot_profile(accession)
                st.session_state["disc_profile_result"] = (query, profile, error)

        result = st.session_state.get("disc_profile_result")
        if result and result[0] == query:
            _, profile, error = result
            if error:
                st.error(error)
            else:
                st.subheader(f"{profile['protein_name']} ({profile['gene']})")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("UniProt", profile["accession"])
                c2.metric("Length", f"{profile['length']} aa" if profile["length"] else "n/a")
                c3.metric("Mass", f"{profile['mass_da'] / 1000:.1f} kDa" if profile["mass_da"] else "n/a")
                c4.metric("Entry", "Reviewed" if profile["reviewed"] else "Unreviewed")
                st.caption(f"Organism: {profile['organism']}")

                if profile["function"]:
                    st.markdown("**Function**")
                    st.write(profile["function"])
                if profile["locations"]:
                    st.markdown("**Subcellular location:** " + ", ".join(profile["locations"]))
                if profile["diseases"]:
                    st.markdown("**Disease associations:** " + ", ".join(profile["diseases"]))
                if profile["domains"]:
                    st.markdown("**Domains and regions**")
                    st.dataframe(pd.DataFrame(profile["domains"]), width="stretch", hide_index=True)
                if profile["ptm_counts"]:
                    st.markdown("**Annotated modifications:** " + ", ".join(f"{k}: {v}" for k, v in profile["ptm_counts"].items()))

                st.markdown("**Target indicators** (facts from database cross-references; not a prediction)")
                st.dataframe(
                    pd.DataFrame(ann.druggability_indicators(profile), columns=["Indicator", "Value"]),
                    width="stretch", hide_index=True
                )
                if profile["structures"]:
                    st.markdown("**Experimental structures**")
                    structures = pd.DataFrame(profile["structures"])
                    structures["Link"] = "https://www.rcsb.org/structure/" + structures["PDB_ID"].astype(str)
                    st.dataframe(structures, width="stretch", hide_index=True,
                                 column_config={"Link": st.column_config.LinkColumn("Link")})
                st.caption("Source: UniProt (rest.uniprot.org). Always confirm important facts in the original database.")

# ================================================================== 3. interactions
with tabs[2]:
    st.header("What does it interact with?")
    if not query:
        st.info("Choose a protein above first.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            level = st.selectbox(
                "Minimum confidence", ["Medium (0.4)", "High (0.7)", "Highest (0.9)"], index=1, key="disc_string_level"
            )
        with c2:
            extra = st.slider("Add the most connected extra partners", 0, 20, 5, key="disc_string_extra")
        include_mine = st.checkbox(
            "Also include my other top shortlisted proteins (shows which of them are connected)",
            value=bool(significant_proteins), key="disc_string_mine", disabled=not significant_proteins
        )
        min_score = {"Medium": 400, "High": 700, "Highest": 900}[level.split(" ")[0]]

        if st.button("Build interaction network", key="disc_string_btn"):
            focus = query_for_lookup
            if not first_accession(query_for_lookup) == query_for_lookup:
                focus = query_for_lookup                           # gene symbol is what STRING wants
            else:
                profile_result = st.session_state.get("disc_profile_result")
                if profile_result and profile_result[1]:
                    focus = profile_result[1]["gene"] or focus
            genes = [focus]
            if include_mine:
                genes += [p.split(" (")[0] for p in significant_proteins if p != query][:30]
            edges, message = ann.fetch_string_network(genes, min_score=min_score, add_nodes=extra)
            st.session_state["disc_network"] = (query, focus, edges, message)

        net = st.session_state.get("disc_network")
        if net and net[0] == query:
            _, focus, edges, message = net
            if message:
                st.warning(message)
            if edges is not None and not edges.empty:
                mine = {p.split(" (")[0] for p in significant_proteins}
                metrics = ann.network_metrics(edges, own_proteins=mine)
                st.plotly_chart(
                    ann.network_figure(edges, metrics, focus=focus,
                                       log2fc={k.split(" (")[0]: v for k, v in log2fc.items()}),
                    width="stretch", key="disc_network_fig"
                )
                st.caption(
                    "Node size = number of partners. Red outline = in your shortlist. "
                    "Colour = log2 fold-change in your data (when available). Edge = STRING confidence ≥ chosen level."
                )
                hubs = metrics[metrics["Hub"]]["Protein"].tolist()
                if hubs:
                    st.success("Hub proteins (many partners): " + ", ".join(hubs[:10]))
                shared = metrics[metrics["In_your_data"] & (metrics["Protein"] != focus)]["Protein"].tolist()
                if shared:
                    st.info("Connected proteins that are also changed in your experiment: " + ", ".join(shared[:15]))
                display_results_table(metrics, "disc_net_metrics", default_index=1)
                st.download_button("Download interactions (CSV)", edges.to_csv(index=False).encode("utf-8"),
                                   file_name="interactions.csv", mime="text/csv", key="disc_net_dl")
                st.caption("Source: STRING (string-db.org). Scores combine experiments, databases, co-expression and text mining; "
                           "they show functional association, not proof of physical binding.")

# ================================================================== 4. AlphaFold
with tabs[3]:
    st.header("Predicted 3D structure (AlphaFold)")
    if not query:
        st.info("Choose a protein above first.")
    else:
        if st.button("Load AlphaFold model", key="disc_af_btn"):
            accession, error = resolve()
            if error:
                st.session_state["disc_af"] = (query, None, None, error)
            else:
                info, error = ann.fetch_alphafold(accession)
                text = None
                if info and not error:
                    text, error = ann.download_structure_text(info["pdb_url"])
                st.session_state["disc_af"] = (query, info, text, error)

        af = st.session_state.get("disc_af")
        if af and af[0] == query:
            _, info, text, error = af
            if error:
                st.warning(error)
            if info:
                plddt = info.get("mean_plddt")
                if plddt is not None:
                    reading = ("very high" if plddt > 90 else "confident" if plddt > 70 else "low" if plddt > 50 else "very low")
                    st.metric("Mean confidence (pLDDT)", f"{plddt:.1f}", reading)
                st.markdown(
                    f"Model **{info['entry_id']}** · [open in AlphaFold DB]"
                    f"(https://alphafold.ebi.ac.uk/entry/{info['entry_id'].split('-')[1] if info.get('entry_id') else ''}) · "
                    f"[download PDB]({info['pdb_url']})"
                )
            if text:
                _show_html(stc.viewer_html(text, color_by="plddt", height=520), 560)
                st.caption(
                    "Colour = AlphaFold confidence per residue: blue > 90 (very high), cyan 70-90 (confident), "
                    "yellow 50-70 (low), orange < 50 (very low; often flexible/disordered, do not trust the shape). "
                    "This is a PREDICTION, not an experimental structure. Drag to rotate, scroll to zoom."
                )

# ================================================================== 5. own structure
with tabs[4]:
    st.header("Analyse my structure (X-ray / cryo-EM / NMR / model)")
    st.caption("Upload a PDB-format file (.pdb / .ent). Quality numbers come from the file header; "
               "angles and B-factors are calculated here.")
    pdb_file = st.file_uploader("PDB file", type=["pdb", "ent", "txt"], key="disc_pdb_upload")
    if pdb_file is not None:
        try:
            text = pdb_file.getvalue().decode("utf-8", errors="ignore")
            atoms = stc.parse_pdb_text(text)
        except ValueError as error:
            st.error(str(error))
            atoms = None
        if atoms is not None:
            header = stc.header_quality(text)
            comp = stc.composition(atoms)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Method", (header["method"] or "n/a").title())
            c2.metric("Resolution", f"{header['resolution']} Å" if header["resolution"] else "n/a")
            c3.metric("R-work / R-free", f"{header['r_work'] or 'n/a'} / {header['r_free'] or 'n/a'}")
            c4.metric("Residues", comp["residues"])
            for note in stc.interpret_quality(header):
                st.write("• " + note)
            st.caption(
                f"Chains: {', '.join(comp['chains'])} · waters: {comp['waters']} · "
                f"ligands: {', '.join(comp['ligands']) if comp['ligands'] else 'none'}"
            )

            st.subheader("Flexibility (B-factors)")
            bt = stc.bfactor_table(atoms)
            st.plotly_chart(stc.bfactor_figure(bt), width="stretch", key="disc_bfac")
            st.caption("High B-factor = mobile or poorly defined region. Loops with very high values are often unreliable for docking.")

            st.subheader("Backbone angles (Ramachandran)")
            angles = stc.dihedral_table(atoms)
            if angles.empty:
                st.info("No continuous backbone found, so angles cannot be calculated.")
            else:
                classified = stc.classify_ramachandran(angles)
                summary = stc.ramachandran_summary(classified)
                m1, m2, m3 = st.columns(3)
                m1.metric("Favoured (approx.)", f"{summary['Favoured_pct']}%")
                m2.metric("Allowed", f"{summary['Allowed_pct']}%")
                m3.metric("Outliers", f"{summary['Outlier_pct']}%")
                st.plotly_chart(stc.ramachandran_figure(classified), width="stretch", key="disc_rama")
                outliers = classified[classified["region"] == "Outlier"]
                if len(outliers):
                    with st.expander(f"{len(outliers)} outlier residue(s)"):
                        st.dataframe(outliers[["chain", "resseq", "resname", "phi", "psi"]].round(1), hide_index=True)
                st.caption(
                    "APPROXIMATE screen using coarse regions (glycine not judged). Use MolProbity or PROCHECK "
                    "for publication-grade validation. Well-refined structures usually have > 95% favoured."
                )

            st.subheader("3D view")
            color_by = st.selectbox("Colour by", ["B-factor", "Chain", "Rainbow (N → C)"], key="disc_color")
            if len(text) > 6_000_000:
                st.info("This file is too large for the viewer; open it in PyMOL, ChimeraX or Mol*.")
            else:
                mode = {"B-factor": "bfactor", "Chain": "chain", "Rainbow (N → C)": "spectrum"}[color_by]
                _show_html(stc.viewer_html(text, color_by=mode, height=520), 560)

# ================================================================== 6. dose response
with tabs[5]:
    st.header("How strongly does my compound act? (IC50 / EC50)")
    st.caption(
        "Upload concentration and response measurements. A four-parameter logistic curve is fitted "
        "(the field standard) and the IC50 is reported with a 95% confidence interval."
    )
    use_example = st.checkbox("Use simulated example data (2 compounds)", key="disc_dr_example")

    table = None
    if use_example:
        rng = np.random.default_rng(1)
        conc = np.repeat(np.logspace(-2, 2, 8), 3)
        frames = []
        for name, ic50 in (("Compound A", 0.8), ("Compound B", 12.0)):
            frames.append(pd.DataFrame({
                "Concentration_uM": conc,
                "Response_percent": dr.four_pl(conc, 4.0, 100.0, np.log10(ic50), 1.2) + rng.normal(0, 3, len(conc)),
                "Compound": name}))
        table = pd.concat(frames, ignore_index=True)
    else:
        upload = st.file_uploader("Dose-response table (CSV / TSV / Excel)", type=["csv", "tsv", "txt", "xlsx"], key="disc_dr_upload")
        if upload is not None:
            try:
                sheets = list_sheets(upload)
                sheet = st.selectbox("Sheet", sheets, key="disc_dr_sheet") if len(sheets) > 1 else None
                table = load_file(upload, sheet_name=sheet)
            except Exception as error:                      # noqa: BLE001
                st.error(f"Could not read this file: {error}")

    if table is not None:
        numeric = [c for c in table.columns if pd.api.types.is_numeric_dtype(table[c])]
        if len(numeric) < 2:
            st.error("The table needs at least two numeric columns: concentration and response.")
        else:
            display_results_table(table, "disc_dr_table", default_index=1)
            c1, c2, c3, c4 = st.columns(4)
            conc_col = c1.selectbox("Concentration column", numeric, key="disc_dr_conc")
            resp_col = c2.selectbox("Response column", [c for c in numeric if c != conc_col], key="disc_dr_resp")
            others = [c for c in table.columns if c not in (conc_col, resp_col)]
            default_group = next(
                (c for c in others if not pd.api.types.is_numeric_dtype(table[c]) and 1 < table[c].nunique() <= 30), None
            )
            group_options = [None] + others
            group_col = c3.selectbox("Compound column (optional)", group_options,
                                     index=group_options.index(default_group), key="disc_dr_group",
                                     format_func=lambda c: "(single compound)" if c is None else str(c))
            unit = c4.text_input("Concentration unit", "µM", key="disc_dr_unit")

            has_control = bool((table[conc_col] == 0).any())
            normalise = st.checkbox(
                "Normalise to untreated control (concentration 0 = 100%)", value=False, key="disc_dr_norm",
                disabled=not has_control,
                help="Needs rows with concentration 0. Skip this if your response is already a percentage."
            )

            working = table[[conc_col, resp_col] + ([group_col] if group_col else [])].dropna()
            try:
                if normalise:
                    working = dr.normalise_to_control(working, conc_col, resp_col, group_col)
                summary, fits, errors = dr.fit_table(working, conc_col, resp_col, group_col)
            except ValueError as error:
                st.error(str(error))
                summary, fits, errors = pd.DataFrame(), {}, {}

            for name, message in errors.items():
                st.warning(f"{name}: {message}")
            if not summary.empty:
                st.plotly_chart(
                    dr.dose_response_figure(working, conc_col, resp_col, fits, group_col, unit=unit,
                                            y_label="Response (% of control)" if normalise else str(resp_col)),
                    width="stretch", key="disc_dr_fig"
                )
                shown = summary.copy()
                for col in ("IC50", "IC50_95%_low", "IC50_95%_high", "Hill", "Bottom", "Top"):
                    shown[col] = shown[col].map(lambda v: f"{v:.3g}")
                shown["R_squared"] = shown["R_squared"].map(lambda v: f"{v:.3f}")
                st.dataframe(shown.rename(columns={"IC50": f"IC50 ({unit})"}), width="stretch", hide_index=True)
                for _, row in summary.iterrows():
                    if row["Warnings"]:
                        st.warning(f"{row['Compound']}: {row['Warnings']}")
                st.download_button("Download fit results (CSV)", summary.to_csv(index=False).encode("utf-8"),
                                   file_name="dose_response_fits.csv", mime="text/csv", key="disc_dr_dl")
                st.caption(
                    "Use at least 6-8 concentrations spanning the whole curve (the IC50 should lie inside the tested range) "
                    "and 2-3 replicates. IC50 values from different assays are only comparable under the same conditions."
                )

# END OF FILE: pages/2_Proteomics_Discovery.py