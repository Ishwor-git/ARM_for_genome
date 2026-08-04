# Preprocessing Documentation

Pipeline steps, commands, and outputs for reproducibly building the working
dataset from the raw GEO files. Run steps in order. All scripts are idempotent
and overwrite their outputs.

## Environment

- Python 3.13
- `pandas`, `numpy`, `scipy`, `scikit-learn` (`pandas`, `sklearn` `feature_selection`, `scipy.stats.ttest_ind`)
- No hand-written data pipeline code beyond `src/preprocessing/`; uses standard library functions only.

## Files

| Step | Script | Run |
|---|---|---|
| 1. Parse raw GEO files | `src/preprocessing/dataview.py` | `python src/preprocessing/dataview.py` |
| 2. Remove non-item columns | `src/preprocessing/clean_features.py` | `python src/preprocessing/clean_features.py` |
| 3. Feature selection + discretization | `src/preprocessing/feature_selection.py` | see below |

Inputs (raw, must exist):
- `data/GSE2034_series_matrix.txt`
- `data/GSE2034_clinical.txt`

---

## Step 1 — `dataview.py`

Reads the raw GEO series-matrix and clinical files, extracts the expression
table, transposes it to samples-as-rows, merges with clinical data, and writes
the working tables under `data/final/`.

```bash
python src/preprocessing/dataview.py
```

Behavior:
- Finds the expression table header line (starts with `"ID_REF"`) and reads the
  TAB-delimited matrix below it (`286` sample columns × `22283` probe rows,
  plus a trailing `!series_matrix_table_end` row).
- Reads the clinical table after the `#`-comment block (header row starts with
  `PID`).
- Renames `GEO_asscession_number` → `GEO_accession_number` (source typo),
  drops `PID`.
- Sets `ID_REF` as index, transposes so rows = samples, drops `!series_matrix_table_end` from columns, merges clinical on `GEO_accession_number`.

Outputs (all written to `data/final/`):

| File | Shape | Content |
|---|---|---|
| `expression_matrix.csv` | 286 × 22,284 | Expression intensities, one column per probe (includes 68 `AFFX-*` control probes and the `!series_matrix_table_end` column; sample IDs not written) |
| `clinical_data.csv` | 286 × 6 | `GEO_accession_number`, `lymph_node_status`, `time_to_relapse`, `relapse`, `ER_Status`, `Brain_relapses` |
| `final_data.csv` | 286 × 22,290 | Full merge: 22,284 expression columns + 6 clinical columns |

Key stdout: expression start line, shapes, missing counts (expect 0/0),
sample ID overlap (expect 286/286/286).

---

## Step 2 — `clean_features.py`

Builds the ARM-ready feature table by removing columns that are not valid
association-rule items.

```bash
python src/preprocessing/clean_features.py
```

Input: `data/final/final_data.csv`

Removed columns:
- All `AFFX-*` Affymetrix QC/control probes (68 columns) — spike-in and
  hybridization controls, not tumor biology.
- Leakage-risk / non-feature columns: `time_to_relapse`, `Brain_relapses`,
  `GEO_accession_number` (incl. typo variant).

Kept: 22,215 expression probes + clinical covariates `lymph_node_status`,
`ER_Status` + target `relapse`.

Output: `data/final/cleaned_data.csv` (286 × 22,219)

Stdout: `Cleaned 22290 -> 22219 columns`

---

## Step 3 — `feature_selection.py`

Class-aware filter feature selection (scored against the `relapse` target),
per-feature median-split discretization of expression probes, and export of
top-k binarized subsets. One run per method:

```bash
python src/preprocessing/feature_selection.py --method mutual_information
python src/preprocessing/feature_selection.py --method chi_square
python src/preprocessing/feature_selection.py --method t_test
```

`--method` choices: `mutual_information` (default), `chi_square`, `t_test`.

Scoring (higher = more relevant):
- `mutual_information`: `mutual_info_classif` on median-split values, `random_state=42`
- `chi_square`: chi² statistic on median-split values
- `t_test`: `|t|` from `ttest_ind` (relapse vs no-relapse) on continuous values

Discretization: per-probe median split `(value > median) → 1/0`, applied to
expression probes only (not clinical covariates, not target). Note: forces
~50/50 high/low per feature (see AGENTS.md §4 for limitations).

Input: `data/final/cleaned_data.csv`

Outputs (per method, under `data/final/feature_selection/<method>/`):

| File | Content |
|---|---|
| `gene_<method>_rankings.csv` | One row per gene: `gene_id`, score column (`mi_score` / `chi2_statistic` / `abs_t_statistic`), sorted descending |
| `top{k}_genes_discretized.csv` | Top-k selected probes (0/1) + `relapse` column appended; one file per k |

Top-k values produced: `k = 100, 150, 200, 250, 300, 350, 400`.

Internal handling: drops the `!series_matrix_table_end` column from
`cleaned_data.csv` if present (series-matrix artifact carried through Steps 1–2).

Stdout (per run): cleaned shape, expression probe count (22,215), target
distribution (179/107), top-10 genes with scores.

---

## Resulting Pipeline Dependency Order

```
raw GEO files
  └─ dataview.py        → final_data.csv, expression_matrix.csv, clinical_data.csv
       └─ clean_features.py  → cleaned_data.csv
            └─ feature_selection.py  → rankings + top100–top400 discretized subsets
```
