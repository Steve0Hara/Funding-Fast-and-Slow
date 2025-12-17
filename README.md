# Thesis: Funding, Fast and Slow. 
## Data & Analysis workspace. Author: Stefan Wilneder.

This contains the **data pulls/exports** (Crunchbase, Companies House, patents, uni rankings) and the **notebooks/scripts** used to clean, enrich, engineer features, and run the empirical analyses for the thesis.

## Where to look for the results (HTML reports)

To make the core analysis easy to review without running any code, I’m including **HTML exports** (that can run in any browser) of the key notebooks where the **models, statistical tests, tables, and diagrams** live:

- `final_models.html` — main modeling results (tables + model outputs)
- `eda_and_kruskal_wallis.html` — EDA + Kruskal–Wallis analyses and related tables/plots
- `sankey.html` — interactive Sankey diagrams (funding-flow visualisations)

All other code files are primarily “pipeline” code (data processing, merging, enrichment, utilities). They are included for completeness and reproducibility, but they’re not converted to HTML because they’re mainly useful for **running the project** rather than reviewing results.

---

## Code files (every code/config file in v4)

### Companies House (scripts + checks)
- `v4/Companies House Data Scripts/fetch_companies_house_api.py`  
  Matches organizations in a CSV to Companies House records (fuzzy name + optional founder/officer matching), then writes:
  - enriched output CSV
  - a “candidates” CSV (all candidate matches per org for auditing)
  - a “search errors” CSV for rows that failed due to request/HTTP errors  
  Configuration lives in the in-file `CONFIG` object; requires `COMPANIES_HOUSE_API_KEY` (or `CONFIG.api_key`).

- `v4/Companies House Data Scripts/fetch_companies_house_pdfs.py`  
  Downloads Companies House filing PDFs (CS01/SH01/AR01/IN01) for matched companies and logs success/failures.  

- `v4/Companies House Data/manual_match_check.ipynb`  
  Small notebook to sample ~200 match pairs for manual inspection.

### Crunchbase (merge)
- `v4/Crunchbase Data/Merging CB Datasets/merge_cb_data.ipynb`  
  Merges Crunchbase bulk export tables into a single organization-level dataset using a YAML-driven join spec.

- `v4/Crunchbase Data/Merging CB Datasets/org_big_config.yaml`  
  YAML config describing the base table, selected columns, renames, and join steps used by `merge_cb_data.ipynb`. (Uses absolute paths.)

### Study pipeline (preprocess → features → models)
- `v4/Study/cb data pre-processing/cb_data_pre_processing.ipynb`  
  Takes the merged Crunchbase table and produces cleaned analysis-ready subsets (global/UK/USA).

- `v4/Study/constructing features/constructing_features.ipynb`  
  Feature engineering notebook (creates covariates/outcomes, round-specific subsets, “apples-to-apples” filters, freeze date logic).

- `v4/Study/Models/final_models.ipynb`  
  Main modeling notebook; loads engineered feature CSVs and prepares the analysis frames used downstream.

- `v4/Study/Models/eda_and_kruskal_wallis.ipynb`  
  EDA blocks comparing funding speed / outcomes across geographies, stages, and sectors; expects inputs under `exports/` and sector splits.

- `v4/Study/Models/logit_old.ipynb`  
  Older logistic modeling exploration (legacy).

- `v4/Study/Models/sankey.ipynb`  
  Minimal notebook for Sankey-related visuals.

### Serial founders
- `v4/Serial Founders Data/serial_founders_analysis.ipynb`  
  Identifies “serial founders” from Crunchbase bulk export roles by looking for prior founding roles (based on founded dates / role start dates).

---

## Data folders (CSV placeholders only)

- `v4/Crunchbase Data/bulk_export/*.csv`  
  Raw Crunchbase bulk export tables (organizations, jobs, people, funding rounds, etc.).

- `v4/Crunchbase Data/Merging CB Datasets/*.csv`  
  Merged organization-level output(s) produced by the merge notebook.

- `v4/Study/cb data pre-processing/*.csv`  
  Cleaned analysis-ready subsets (global / UK / USA splits).

- `v4/Study/constructing features/*.csv`  
  Feature-engineered datasets and round-specific subsets.

- `v4/Study/Models/exports/*.csv`  
  Model/EDA input slices (including time-windowed or currency-normalized exports).

- `v4/Study/Models/export_sectors/*.csv`  
  Sector-specific splits/exports used in sector analyses.

- `v4/Companies House Data/*.csv`  
  Companies House matching/enrichment outputs used in the study.

- `v4/Serial Founders Data/*.csv`  
  Serial-founder output table(s).

- `v4/Patent Data/*.csv`  
  Patent-related exports from Google BigQuery (query code not included, just data) (combined/split files used for analysis/joins).

### Non-CSV data
- `v4/Companies House Data/funded_pdfs/<company_number>_<company_name>/<form_type>/*.pdf`  
  Downloaded Companies House filings grouped by company and form type (e.g., `AR01`, `CS01`, `SH01`, `IN01`).

- `v4/Uni Rankings/Top_Universities_THE.xlsx`  
- `v4/Uni Rankings/FT_uni_ranking.xlsx`

---

## Typical run order

1. **Crunchbase bulk export in** → `v4/Crunchbase Data/bulk_export/*.csv`
2. **Merge Crunchbase tables** → `v4/Crunchbase Data/Merging CB Datasets/merge_cb_data.ipynb` (+ `org_big_config.yaml`)
3. **Pre-process for study splits** → `v4/Study/cb data pre-processing/cb_data_pre_processing.ipynb`
4. **Engineer features / define cohorts** → `v4/Study/constructing features/constructing_features.ipynb`
5. **(Companies House enrichment) already done**
   - match/enrich via API → `v4/Companies House Data Scripts/fetch_companies_house_api.py`
   - optionally download filings PDFs → `v4/Companies House Data Scripts/fetch_companies_house_pdfs.py`
6. **Model + EDA** → notebooks in `v4/Study/Models/`

---

## Notes

- The Companies House scripts are designed to run by editing the in-file `CONFIG` (no CLI args) and require a Companies House API key via `COMPANIES_HOUSE_API_KEY` or `CONFIG.api_key`.
- Several configs use **absolute paths** (notably `org_big_config.yaml` and `fetch_companies_house_pdfs.py`), so moving the folder or renaming parent directories will break runs unless updated.
- Some notebooks expect being run from their own directory so relative paths like `exports/...` resolve.
