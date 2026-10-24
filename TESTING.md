# BioAI — How We Test Everything (read this before selling anything)

## The one command

Open a terminal in your project folder and run:

```
pip install pytest
python -m pytest -q
```

It takes about 2–3 minutes. The last line tells you the result:

* `138 passed` (green) → every check passed. Safe to deploy.
* `2 failed, 136 passed` (red) → **do not deploy.** Copy the lines that start with
  `FAILED` and paste them to Claude as text. Do not paste screenshots.

You can also let GitHub do this for you: the file `.github/workflows/tests.yml`
runs the same tests automatically on every `git push`. On GitHub open the
**Actions** tab: a green tick means all tests passed, a red cross means something broke.

## What the tests cover (and what they cannot)

| Layer | What it checks | Files |
|---|---|---|
| 1. Known answers | Our statistics reproduce **published R / SciPy results** (paired t-test, Welch, ANOVA, Kruskal-Wallis, Tukey HSD, Fisher exact, Holm/Bonferroni/BH correction, Kaplan-Meier + log-rank vs `lifelines`) | `test_figure_studio_stats.py` |
| 2. Planted truth | Fake RNA-seq / proteomics data where we **know** which genes are really different: does the pipeline find them? does it invent others? does ML leak labels (it must score ~50 % on pure noise)? | `test_transcriptomics_engine.py`, `test_proteomics_engine.py`, `sim_data.py` |
| 3. Dataset zoo | Awkward real-world files: wide layout, semicolon/decimal-comma CSV, text-stored numbers, empty rows, duplicate columns, missing values, tiny groups, constant columns, numeric group codes, 100 000 rows, unicode names | `test_figure_studio_data_and_plots.py` |
| 4. Real pages | The actual Streamlit pages are opened and clicked through: upload → every tab → every button, including the dangerous moments (one sample in a group, 2 samples per group, empty dropdowns) | `test_*_page.py` |

**Not covered (be honest with customers about this):** the real Supabase login and
paid-plan gate, Render/Streamlit Cloud hosting itself, Enrichr (needs internet),
browser-specific behaviour (Safari/Firefox) of the PNG/SVG download buttons, and
huge real datasets. Test those by hand (see the checklist below).

## Every customer bug becomes a test (this is how quality grows)

1. Customer reports a problem → ask for the **file** (public data only) and the
   exact **error text**.
2. Send both to Claude. Claude reproduces it, writes a test that fails, fixes the
   code, and the test now passes.
3. That test stays forever, so the same bug can never come back silently.

## Manual checklist before each release (15 minutes)

Do these on the **live Render and Streamlit Cloud sites**, not on your laptop.

1. Sign up with a fresh e-mail → confirmation mail arrives and links to the right site.
2. Log in on the free plan → Figure Studio works, Transcriptomics shows the upgrade gate.
3. Figure Studio: upload `tests/data/Hemoglobin.xlsx` → layout detected as "Wide" →
   Bar Chart → click **Download PNG** and **Download SVG** → open both files.
4. Figure Studio statistics: Compare 2 Groups on the Hemoglobin file →
   result must be **paired t-test, t = −3.9129, p = 0.000281**.
5. Transcriptomics: upload a count matrix → Filter → Normalize → DE → volcano plot → PDF report.
6. Proteomics: upload a protein matrix → assign groups → both missing-value options → DE → ML.
7. Log out → the protected pages must not open.

## Reference results (a result that changes = a bug until proven otherwise)

| Dataset | Analysis | Expected |
|---|---|---|
| `tests/data/Hemoglobin.xlsx` (50 patients) | Paired t-test Before vs After | t = −3.9129, p = 0.000281, d_z = −0.553 |
| R `sleep` (in `test_figure_studio_stats.py`) | Paired t-test | t = −4.0621, p = 0.002833 |
| R `PlantGrowth` | ANOVA / Tukey | F = 4.846, p = 0.01591 / trt2−trt1 p = 0.0120 |

Add a row for every public dataset you validate (GEO accession, CPTAC cohort, …).

## Free ways to get more testing before you have revenue

* **Founding testers:** 3–5 researchers/students get a free Pro plan for one month in
  exchange for a written list of what confused or broke. Give each a specific job
  (one tests Figure Studio, one Proteomics, one Transcriptomics).
* **Validation note:** a short write-up "BioAI vs R on N public datasets" is both a
  quality check and a marketing document. Offer a university colleague an
  acknowledgment or co-authorship for repeating your analyses in R/SPSS.
* **Pay only for results, only after revenue:** a small fixed fee per *verified* bug
  once customers are paying.
* Do **not** promise equity or revenue share without a lawyer.

## Rules that keep the numbers honest (do not weaken these)

* Every multi-test analysis is corrected for multiple testing (Holm / BH / Tukey).
* Feature selection for machine learning happens **inside** each cross-validation
  fold. Choosing genes first with all samples gives ~100 % accuracy even on random data.
* A missing value is never silently turned into p = 1.0; untestable genes are
  reported as untestable and left out of the FDR correction.
* Bad input is refused with a plain-language message, never a red traceback.
