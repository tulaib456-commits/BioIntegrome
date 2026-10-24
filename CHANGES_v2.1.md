# BioAI — change list (package v2.1)

## A. Figure Studio (pages/3_Figure_Studio.py + modules/figure_studio/*)
1. **Crash fixed:** files without a text column (e.g. ID | Before | After) crashed the whole page.
2. **Data layout step:** "Long" or "Wide"; wide tables (Before | After) are reshaped automatically; numeric group codes (0/1/2) can be treated as categories.
3. **Robust file loading:** Excel sheet selection; comma/semicolon/tab/pipe detection; UTF-8/Latin-1; decimal commas and thousands separators; numbers stored as text; empty rows/columns removed; duplicate column names renamed; leading zeros (007) kept; every change is reported to the user.
4. **Downloads:** PNG and SVG buttons under the chart + camera (PNG) and floppy-disk (SVG) icons in the chart toolbar, created in the browser (no server Chrome). "Share chart" (uploads to Plotly Cloud) removed for privacy. PDF kept as an optional server-side extra.
5. **Full data preview** (all rows, choose how many) instead of 5 rows.
6. **Bar chart was wrong:** it plotted the SUM of all rows. Now mean ± SD / SEM, mean, median or sum — your choice.
7. **Four plot types that did nothing now work:** Kaplan-Meier (with log-rank), Forest, Venn (2-set), Paired dot plot (time order selectable). Clear messages for bad input (e.g. event column not 0/1, lower CI above estimate, same column chosen twice).
8. **Grouped Comparison:** brackets were positioned by alphabetical order but boxes by data order (wrong stars on wrong boxes) — fixed; multiple-comparison correction added (Holm default, Bonferroni, Benjamini-Hochberg, none) and Tukey HSD; test choice (auto / Welch / Mann-Whitney / Tukey); statistics table with raw and adjusted p-values shown under the figure.
9. **Statistics box:** `pd` import bug (chi-square crashed); paired tests align pairs BY SUBJECT ID (a missing value no longer shifts all following pairs); paired effect size is d_z; correlation drops missing values pairwise; chi-square switches to Fisher's exact when counts are small and reports Cramér's V; clear messages instead of crashes for tiny/constant groups; post-hoc tables after ANOVA/Kruskal-Wallis.
10. **Log scale** is applied only when all values are positive (otherwise points silently vanished) and the user is told.
11. Deprecated `use_container_width` replaced (Streamlit announced removal).

## B. Transcriptomics (pages/1_Transcriptomics.py + engine)
12. **Missing values no longer silently become p = 1.0.** Handled gene by gene; untestable genes get p = NaN and are excluded from the FDR correction (and counted for the user).
13. **Input checks:** group with < 2 samples, negative values in raw-count mode, samples with zero reads, text columns inside the sample list — each refused with a plain-language message. The page no longer crashes while you are still choosing samples.
14. **Median-of-ratios normalization** no longer returns all-NaN on sparse data (DESeq2 "poscounts" fallback).
15. **Annotation text columns** (e.g. "Description") are no longer treated as samples.
16. **ANOVA:** rewritten to handle missing values per gene (vectorized, verified against scipy on 1 489 proteins); Tukey post-hoc cannot crash on missing data.
17. **Machine learning:** the option "Use significant genes from DE/ANOVA" was REMOVED. It selected genes using all samples before cross-validation and reported 100 % accuracy on pure noise (honest: ~41 %). Genes are now always selected inside each fold.
18. **Folds slider crash** (smallest group with exactly 2 samples) fixed.
19. PCA works with missing values (display only).

## C. Proteomics (pages/2_Proteomics.py + engine)
20. **Machine learning had no honest option at all** (always used pre-selected significant proteins) → now in-fold selection, same as above.
21. **Imputation now follows Perseus** (per-sample distribution, mean − 1.8 SD, width 0.3 SD). It used each protein's own spread before.
22. **New choice on the Missing Values tab:** "use the values each protein has (no imputation)" or Perseus-style imputation, with plain-language guidance (measured: 60–90 % of real changes found vs 16–68 % with imputation when values are missing at random).
23. **Median normalization** estimates shifts from proteins measured in every sample (falls back to all values if fewer than 50) — roughly halved false positives in simulation.
24. **New table "Proteins detected in only ONE group":** switched on/off proteins cannot be t-tested; they used to vanish silently.
25. Upload uses the robust loader (semicolon/decimal-comma files, text annotation columns) and refuses files with fewer than 4 numeric sample columns with a clear message.
26. Same crash protection for DE/ANOVA/ML as Transcriptomics.

## D. Quality system (new)
27. **150 automated tests** (tests/): known answers from R, planted-truth simulations, a dataset zoo, the three real pages clicked through, and the dummy datasets with their documented results.
28. **GitHub Actions** workflow (.github/workflows/tests.yml) runs the tests on every push. `TESTING.md` explains everything for a non-programmer.
29. `requirements.txt`: streamlit pinned to 1.66.0 and plotly to 7.1.0 (the tested versions).

## Not changed / not verified
Login and the paid-plan gate, Render/Streamlit-Cloud hosting, Enrichr (needs internet), PDF report content, deep-learning (DNN) training, Safari/Firefox download behaviour, Python 3.11 (tests ran on 3.12).
