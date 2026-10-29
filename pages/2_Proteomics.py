import streamlit as st
import pandas as pd
import numpy as np

from modules.auth import require_active_plan
from modules.proteomics.missing_value_handler import (
    MissingValueAnalyzer, ProteomicsFilter, Imputer, presence_absence_table
)
from modules.proteomics.normalization import ProteomicsNormalizer

# Reused directly from the Transcriptomics engine -- same
# statistical logic applies once data is protein x sample shaped
from modules.transcriptomics.validator import detect_gene_column, detect_sample_columns
from modules.transcriptomics.gene_id_mapper import detect_id_type, GeneIDConverter
from modules.transcriptomics.differential_expression import DifferentialExpression, suggest_groups
from modules.transcriptomics.multi_group_de import MultiGroupDE, suggest_multi_groups
from modules.transcriptomics.group_assignment import group_assignment_widget
from modules.transcriptomics.enrichment import EnrichmentAnalysis, ENRICHR_LIBRARIES, SUPPORTED_ORGANISMS
from modules.transcriptomics.feature_builder import FeatureBuilder
from modules.transcriptomics.ml_classifier import MLClassifier
from modules.figure_studio.plot_editor import render_editable_plot
from modules.transcriptomics.dnn_classifier import DNNClassifier
from modules.transcriptomics.model_interpretation import ROCAnalysis, ModelInterpreter
from modules.transcriptomics.plots import PlotFactory
from modules.transcriptomics.result_display import display_results_table
from modules.figure_studio.data_loader import load_file
from modules.transcriptomics.pdf_report import generate_pdf_report

require_active_plan()

st.title("🧪 Proteomics")

if "proteomics_dataset" not in st.session_state:
    st.session_state.proteomics_dataset = None
    st.session_state.proteomics_filtered = None
    st.session_state.proteomics_normalized = None

tabs = st.tabs([
    "Upload", "Missing Values", "Normalization",
    "Differential Expression", "Enrichment Analysis", "Machine Learning", "Export"
])

with tabs[0]:

    from modules.figure_studio.data_loader import list_sheets
    from modules.proteomics.parsers import load_proteomics_file, looks_unlogged, to_log2

    st.header("Upload Protein Intensity Matrix")
    st.caption(
        "Accepted: MaxQuant proteinGroups.txt, DIA-NN report.pg_matrix.tsv, or any table "
        "with proteins as rows and one numeric column per sample (CSV, TSV, TXT, Excel). "
        "MaxQuant and DIA-NN files are cleaned and log2-transformed automatically."
    )

    uploaded = st.file_uploader("Upload dataset", type=["csv", "tsv", "txt", "xlsx"])

    if uploaded is not None:

        data = None
        info = {}

        try:
            sheets = list_sheets(uploaded)
            sheet = st.selectbox("Which Excel sheet?", sheets, key="prot_sheet") if len(sheets) > 1 else None
            data, info = load_proteomics_file(uploaded, sheet_name=sheet)
        except Exception as error:
            st.error(f"Could not read this file: {error}")

        if data is not None:

            for note in info.get("notes", []):
                st.caption(f"ℹ️ {note}")

            if info.get("format") != "generic table":
                st.success(f"Recognised: {info['format']}")
                details = [f"intensity used: {info.get('intensity_used', 'run columns')}"]
                for key, label in (
                    ("removed_reverse_hits", "reverse hits"),
                    ("removed_potential_contaminants", "contaminants"),
                    ("removed_only_identified_by_site", "'only identified by site'"),
                    ("removed_never_measured", "proteins never measured"),
                ):
                    if info.get(key):
                        details.append(f"{info[key]} {label} removed")
                details.append("values converted to log2 (a 0 means 'not measured' and became a missing value)")
                st.caption(" · ".join(details))

            n_samples_found = len(detect_sample_columns(data))

            if n_samples_found < 4:
                st.error(
                    f"Only {n_samples_found} numeric sample column(s) were found. "
                    "Proteomics needs proteins as rows and at least 4 numeric sample "
                    "columns (two groups with replicates)."
                )
            else:
                sample_columns = detect_sample_columns(data)

                if info.get("format") == "generic table" and looks_unlogged(data, sample_columns):
                    st.warning(
                        "These values look like raw intensities (very large numbers). "
                        "Statistical tests on raw intensities are misleading; log2 transformation is standard."
                    )
                    if st.checkbox("Apply log2 transformation (recommended)", value=True, key="prot_log2"):
                        data = to_log2(data, sample_columns)

                st.session_state.proteomics_dataset = data
                st.success(f"Uploaded: {len(data)} proteins, {n_samples_found} samples")
                display_results_table(data, "prot_upload", default_index=4)

with tabs[1]:

    if st.session_state.proteomics_dataset is not None:

        st.header("Missing Value Analysis & Imputation")

        protein_column = detect_gene_column(st.session_state.proteomics_dataset)
        samples = detect_sample_columns(st.session_state.proteomics_dataset)

        analyzer = MissingValueAnalyzer(
            st.session_state.proteomics_dataset, protein_column, samples
        )

        summary = analyzer.missingness_summary()

        st.metric(
            "Overall missing values",
            f"{summary['Missing_Fraction'].mean() * 100:.1f}%"
        )

        st.caption(
            "Assign groups below to filter/impute correctly -- a "
            "protein detected in one group but not another is "
            "meaningful signal, not noise."
        )

        group_map = group_assignment_widget(samples, key_prefix="proteomics_mv")

        if len(set(group_map.values())) >= 2:

            min_valid = st.slider(
                "Minimum valid-value fraction (in at least one group)",
                0.3, 1.0, 0.7
            )

            mv_method = st.radio(
                "How should the remaining missing values be handled?",
                [
                    "Use the values each protein has (no imputation)",
                    "Perseus-style imputation (replace with low values)"
                ],
                key="prot_mv_method",
                help=(
                    "No imputation: each protein is tested on the samples where it "
                    "was measured (needs 2+ per group). Best when values are missing "
                    "more or less at random, or only a few are missing. "
                    "Perseus-style: use when a missing value means the protein was "
                    "BELOW the detection limit (typical for abundant-vs-absent "
                    "proteins); it fills gaps with low values drawn from each "
                    "sample's own intensity distribution."
                )
            )

            if st.button("Filter and Handle Missing Values"):

                try:
                    prot_filter = ProteomicsFilter(
                        st.session_state.proteomics_dataset, protein_column, samples
                    )
                    filtered = prot_filter.filter_by_group_detection(group_map, min_valid)

                    if mv_method.startswith("Perseus"):
                        handled = Imputer().impute(filtered, samples)
                        action = "imputed (Perseus-style)"
                    else:
                        handled = filtered
                        action = "kept with their missing values (no imputation)"

                    st.session_state.proteomics_filtered = handled

                    st.success(
                        f"{len(filtered)} of {len(st.session_state.proteomics_dataset)} "
                        f"proteins retained and {action}."
                    )
                except ValueError as error:
                    st.error(f"This step could not be run: {error}")

        if st.session_state.proteomics_filtered is not None:
            display_results_table(st.session_state.proteomics_filtered, "prot_filtered", default_index=4)

    else:
        st.info("Upload a dataset first.")

with tabs[2]:

    if st.session_state.proteomics_filtered is not None:

        st.header("Normalization")

        method = st.selectbox(
            "Normalization Method",
            ["Median Normalization (recommended)", "Quantile Normalization"]
        )

        samples = detect_sample_columns(st.session_state.proteomics_filtered)
        normalizer = ProteomicsNormalizer(st.session_state.proteomics_filtered, samples)

        if st.button("Apply Normalization"):

            if method.startswith("Median"):
                st.session_state.proteomics_normalized = normalizer.median_normalization()
            else:
                st.session_state.proteomics_normalized = normalizer.quantile_normalization()

        if st.session_state.proteomics_normalized is not None:
            display_results_table(st.session_state.proteomics_normalized, "prot_normalized", default_index=4)

    else:
        st.info("Complete Missing Value handling first.")

with tabs[3]:

    if st.session_state.proteomics_normalized is not None:

        st.header("Differential Expression")
        st.caption(
            "Same engine as Transcriptomics -- proven, tested logic, "
            "applied to protein abundance instead of gene expression."
        )

        data = st.session_state.proteomics_normalized
        protein_column = detect_gene_column(data)
        samples = detect_sample_columns(data)

        de_mode = st.radio(
            "Comparison Type",
            ["Two-Group (t-test)", "Multi-Group (ANOVA, 3+ conditions)"],
            horizontal=True, key="prot_de_mode"
        )

        if de_mode == "Two-Group (t-test)":

            suggested_a, suggested_b = suggest_groups(samples)

            col1, col2 = st.columns(2)
            with col1:
                group_a = st.multiselect("Group A", samples, default=suggested_a or [], key="prot_group_a")
            with col2:
                group_b = st.multiselect(
                    "Group B", [s for s in samples if s not in group_a],
                    default=[s for s in (suggested_b or []) if s not in group_a],
                    key="prot_group_b"
                )

            if group_a and group_b:

                de = None
                de_results = None

                if len(group_a) < 2 or len(group_b) < 2:
                    st.warning(
                        f"Group A has {len(group_a)} sample(s) and Group B has "
                        f"{len(group_b)}. Each group needs at least 2 samples "
                        "(replicates) for a statistical test."
                    )
                else:
                    try:
                        de = DifferentialExpression(data, group_a, group_b, already_normalized=True)
                        de_results = de.run()
                    except ValueError as error:
                        st.error(f"This comparison cannot be run: {error}")
                        de = None

                if de is not None and de_results is not None:
                    st.session_state.proteomics_de_results = de_results
                    st.session_state.proteomics_de_groups = (group_a, group_b)
                    st.session_state.proteomics_de_kind = "two_group"

                    st.success(f"{int(de_results['Significant'].sum())} significant proteins")
                    display_results_table(de_results, "prot_de")

                    if getattr(de, "n_untestable", 0):
                        st.caption(
                            f"{de.n_untestable} protein(s) could not be tested (fewer than 2 "
                            "valid values in a group, or no variation) and were left out of "
                            "the FDR correction."
                        )

                    only_one_group = presence_absence_table(
                        data, protein_column, group_a, group_b
                    )
                    with st.expander(
                        f"Proteins detected in only ONE group ({len(only_one_group)}) — "
                        "cannot be t-tested, often the most interesting",
                        expanded=len(only_one_group) > 0
                    ):
                        if only_one_group.empty:
                            st.info("No protein is detected in one group and absent from the other.")
                        else:
                            st.caption(
                                "Detected in at least 70% of one group and in none of the "
                                "other. These are 'switched on/off' or below-detection-limit "
                                "candidates; confirm them with a targeted assay."
                            )
                            display_results_table(only_one_group, "prot_only_one_group", default_index=1)
                            st.download_button(
                                "Download this table (CSV)",
                                only_one_group.to_csv(index=False).encode("utf-8"),
                                file_name="proteins_detected_in_one_group.csv",
                                mime="text/csv", key="prot_only_one_dl"
                            )

                    st.info(
                        "Next step: open **Proteomics Discovery** in the sidebar to turn these "
                        "results into a ranked target shortlist, protein profiles, interaction "
                        "networks, 3D structures and dose-response analysis."
                    )

                    plot_factory = PlotFactory(de.df)
                    render_editable_plot(plot_factory.volcano_plot(de_results), "tx_volcano", default_title="Volcano plot")

                    st.subheader("PCA")
                    pca_group_map = {s: "Group A" for s in group_a}
                    pca_group_map.update({s: "Group B" for s in group_b})
                    render_editable_plot(plot_factory.pca_plot(pca_group_map), "prot_pca", default_title="PCA plot")

        else:

            suggested_map = suggest_multi_groups(samples)
            valid_map = group_assignment_widget(samples, key_prefix="prot_anova", suggested_map=suggested_map)

            if len(set(valid_map.values())) >= 2 and st.button("Run ANOVA", key="prot_anova_run"):

                try:
                    new_engine = MultiGroupDE(data, valid_map, protein_column, already_normalized=True)
                    new_results = new_engine.run()

                    st.session_state.proteomics_de_results = new_results
                    st.session_state.proteomics_multi_de_engine = new_engine
                    st.session_state.proteomics_de_kind = "anova"
                except ValueError as error:
                    st.error(f"ANOVA cannot be run: {error}")

            # results live in session_state, so they stay visible when you touch any dropdown below
            mgde = st.session_state.get("proteomics_multi_de_engine")

            if st.session_state.get("proteomics_de_kind") == "anova" and mgde is not None:

                anova_results = st.session_state.proteomics_de_results

                st.success(f"{int(anova_results['Significant'].sum())} significant proteins")
                display_results_table(anova_results, "prot_anova")

                st.subheader("Inspect a Protein")

                n_top = st.selectbox("How many proteins to list", [10, 30, 50, 100, 500], index=1, key="prot_top_n")
                top_proteins = anova_results.head(n_top)[anova_results.columns[0]].tolist()

                selected_protein = st.selectbox(
                    "Choose a significant protein to inspect",
                    top_proteins, key="prot_anova_select"
                )

                plot_factory = PlotFactory(data)

                st.subheader("Compare Multiple Proteins")

                multi_proteins = st.multiselect(
                    "Select proteins to compare (heatmap)",
                    top_proteins, default=top_proteins[:10], key="prot_anova_multi"
                )

                if multi_proteins:

                    expr_rows = []
                    for protein in multi_proteins:
                        row = mgde.df[mgde.df[protein_column] == protein]
                        if not row.empty:
                            expr_rows.append(row[mgde.samples].iloc[0])

                    if expr_rows:
                        expr_matrix = pd.DataFrame(expr_rows, index=multi_proteins)
                        render_editable_plot(
                            plot_factory.top_genes_heatmap(expr_matrix),
                            "prot_anova_heatmap",
                            default_title="Heatmap of Top Proteins"
                        )

                if selected_protein:

                    expression_table = mgde.gene_expression_table(selected_protein)

                    render_editable_plot(
                        plot_factory.gene_boxplot(expression_table, selected_protein),
                        "prot_anova_box",
                        default_title="Boxplot of Protein Abundance"
                    )

                    st.caption("Tukey HSD pairwise comparison:")
                    posthoc_table = mgde.posthoc(selected_protein)
                    if posthoc_table is None:
                        st.info("Not enough valid values for this protein to run Tukey HSD.")
                    else:
                        st.dataframe(posthoc_table)

                st.info(
                    "Next step: open **Proteomics Discovery** in the sidebar for the ranked "
                    "shortlist, protein profiles, interactions and structures."
                )
    else:
        st.info("Complete Normalization first.")

with tabs[4]:

    if st.session_state.get("proteomics_de_results") is not None:

        st.header("Enrichment Analysis")

        de_results = st.session_state.proteomics_de_results
        significant = de_results.loc[de_results["Significant"], de_results.columns[0]].tolist()

        st.caption(f"{len(significant)} significant proteins available.")

        library_label = st.selectbox("Gene Set Library", list(ENRICHR_LIBRARIES.keys()), key="prot_enrich_lib")
        organism = st.selectbox("Organism", SUPPORTED_ORGANISMS, key="prot_enrich_org")

        if st.button("Run Enrichment Analysis", key="prot_enrich_run"):

            enrichment = EnrichmentAnalysis(significant)
            results, error = enrichment.run(library_label, organism=organism)

            if error:
                st.warning(error)
            else:
                st.session_state.proteomics_enrichment = results
                display_results_table(results, "prot_enrichment", long_text_columns=["Matched_Genes", "Pathway"])

                plot_factory = PlotFactory(st.session_state.proteomics_normalized)
                render_editable_plot(plot_factory.enrichment_bar_plot(results), "prot_enrich_bar", default_title="Enrichment Bar Plot")

    else:
        st.info("Run Differential Expression first.")

with tabs[5]:

    if st.session_state.get("proteomics_de_results") is not None:

        st.header("Machine Learning")
        st.caption("Same cross-validated engine as Transcriptomics.")

        data = st.session_state.proteomics_normalized
        protein_column = detect_gene_column(data)
        samples = detect_sample_columns(data)

        de_results = st.session_state.proteomics_de_results

        label_source = st.radio(
            "Sample Labels From",
            ["Two-Group DE assignment", "Multi-Group ANOVA assignment", "Manual assignment"],
            horizontal=True, key="prot_ml_label_source"
        )

        group_map = None

        if label_source == "Two-Group DE assignment" and st.session_state.get("proteomics_de_groups"):
            group_a, group_b = st.session_state.proteomics_de_groups
            group_map = {s: "Group A" for s in group_a}
            group_map.update({s: "Group B" for s in group_b})

        elif label_source == "Multi-Group ANOVA assignment" and st.session_state.get("proteomics_multi_de_engine"):
            group_map = st.session_state.proteomics_multi_de_engine.group_map

        else:
            st.info("No assignment found from that step yet — assign manually below.")
            group_map = group_assignment_widget(samples, key_prefix="prot_ml_manual")

        if len(set(group_map.values())) >= 2:

            st.info(
                "Proteins are chosen INSIDE each cross-validation fold, using only "
                "the training samples. Choosing proteins first with all samples "
                "(for example 'significant proteins from DE') leaks the answer into "
                "the test and reports inflated, even perfect, accuracy on pure noise."
            )

            select_k = st.slider(
                "Number of top proteins to auto-select (per fold)", 10, 200, 50,
                key="prot_select_k"
            )

            fb = FeatureBuilder(data, protein_column)

            X = None
            try:
                X, y, features, nan_count = fb.build(group_map, gene_list=None)
            except ValueError as build_error:
                st.error(str(build_error))

            if X is not None:

                if nan_count > 0:
                    st.warning(
                        f"{nan_count} missing values found and filled with 0 "
                        "for the machine-learning step only."
                    )

                ml = MLClassifier(X, y)
                min_class = int(y.value_counts().min())

                if min_class < 2:
                    st.error(
                        "At least one class has fewer than 2 samples — "
                        "cannot run cross-validation."
                    )
                else:

                    if min_class > 2:
                        n_folds = st.slider(
                            "Cross-Validation Folds", 2, min_class,
                            min(int(ml.suggest_folds()), min_class), key="prot_n_folds"
                        )
                    else:
                        n_folds = 2
                        st.caption(
                            "The smallest group has only 2 samples, so 2-fold "
                            "cross-validation is used."
                        )

                    model_options = list(ml.models.keys())
                    selected_models = st.multiselect(
                        "Models to Evaluate", model_options,
                        default=model_options, key="prot_ml_models"
                    )

                    if selected_models and st.button("Run Cross-Validation", key="prot_ml_run"):

                        with st.spinner("Running stratified cross-validation..."):
                            summary, detail = ml.evaluate(
                                selected_models, n_folds=n_folds, select_k=select_k
                            )

                        st.session_state.prot_ml_summary = summary
                        st.session_state.prot_ml_select_k = select_k

                    if st.session_state.get("prot_ml_summary") is not None:

                        st.success("Cross-validation complete.")
                        st.dataframe(st.session_state.prot_ml_summary)

                        if X.shape[0] < 30:
                            st.info(
                                "With this few samples, even correctly implemented "
                                "cross-validation can show very high or perfect scores — "
                                "treat it as preliminary until confirmed on an "
                                "independent, larger dataset."
                            )

                        best_model = st.session_state.prot_ml_summary.iloc[0]["Model"]
                        k_used = st.session_state.get("prot_ml_select_k", select_k)

                        cm = ml.confusion_matrix_cv(best_model, n_folds=n_folds, select_k=k_used)
                        plot_factory = PlotFactory(data)
                        render_editable_plot(plot_factory.confusion_matrix_heatmap(cm), "prot_cm", default_title="Confusion Matrix")

                        roc_analysis = ROCAnalysis(ml, best_model, n_folds=n_folds, select_k=k_used)
                        curves = roc_analysis.curves()

                        col_roc, col_pr = st.columns(2)
                        with col_roc:
                            render_editable_plot(plot_factory.roc_curve_plot(curves), "prot_roc", default_title="ROC Curve")
                        with col_pr:
                            render_editable_plot(plot_factory.pr_curve_plot(curves), "prot_pr", default_title="Precision-Recall Curve")

                        if st.button("Compute SHAP Importance", key="prot_shap_btn"):
                            final_pipeline = ml.fit_final_model(best_model, select_k=k_used)
                            interpreter = ModelInterpreter(final_pipeline, ml.X)
                            shap_values, shap_error = interpreter.compute()

                            if shap_error:
                                st.warning(shap_error)
                            else:
                                importance = interpreter.importance_table(shap_values)
                                display_results_table(importance, "prot_shap_importance", default_index=1)
                                render_editable_plot(plot_factory.shap_importance_plot(importance), "prot_shap", default_title="SHAP Importance")

    else:
        st.info("Run Differential Expression first.")


with tabs[6]:

    st.header("Export")

    if st.session_state.proteomics_dataset is not None:
        st.download_button(
            "Download Raw Dataset (CSV)",
            st.session_state.proteomics_dataset.to_csv(index=False),
            file_name="proteomics_raw.csv", mime="text/csv"
        )

    if st.session_state.proteomics_filtered is not None:
        st.download_button(
            "Download Filtered/Imputed Dataset (CSV)",
            st.session_state.proteomics_filtered.to_csv(index=False),
            file_name="proteomics_filtered.csv", mime="text/csv"
        )

    if st.session_state.proteomics_normalized is not None:
        st.download_button(
            "Download Normalized Dataset (CSV)",
            st.session_state.proteomics_normalized.to_csv(index=False),
            file_name="proteomics_normalized.csv", mime="text/csv"
        )

    if st.session_state.get("proteomics_de_results") is not None:
        st.download_button(
            "Download Differential Expression Results (CSV)",
            st.session_state.proteomics_de_results.to_csv(index=False),
            file_name="proteomics_de_results.csv", mime="text/csv"
        )

    if st.session_state.get("proteomics_enrichment") is not None:
        st.download_button(
            "Download Enrichment Results (CSV)",
            st.session_state.proteomics_enrichment.to_csv(index=False),
            file_name="proteomics_enrichment.csv", mime="text/csv"
        )

        st.divider()
        st.subheader("Full PDF Report")

        if st.button("Generate PDF Report", key="prot_pdf_btn"):

            figures = {}
            if st.session_state.get("proteomics_enrichment") is not None:
                pf = PlotFactory(st.session_state.proteomics_normalized)
                figures["Top Enriched Pathways"] = pf.enrichment_bar_plot(st.session_state.proteomics_enrichment)

            pdf_bytes = generate_pdf_report(
                dataset_summary_dict={"Proteins": len(st.session_state.proteomics_dataset), "Samples": len(detect_sample_columns(st.session_state.proteomics_dataset))},
                qc_dict={"Missing values handled": "Yes" if st.session_state.proteomics_filtered is not None else "No"},
                filtering_info=f"{len(st.session_state.proteomics_filtered)} proteins after missing-value filtering." if st.session_state.proteomics_filtered is not None else "Not yet run.",
                normalization_method="Applied" if st.session_state.proteomics_normalized is not None else "Not yet run",
                de_results=st.session_state.get("proteomics_de_results"),
                enrichment_results=st.session_state.get("proteomics_enrichment"),
                ml_summary=st.session_state.get("prot_ml_summary"),
                figures=figures if figures else None
            )

            st.session_state.prot_pdf_bytes = pdf_bytes

        if st.session_state.get("prot_pdf_bytes") is not None:
            st.download_button("Download PDF Report", st.session_state.prot_pdf_bytes, file_name="Proteomics_Report.pdf", mime="application/pdf", key="prot_pdf_download")