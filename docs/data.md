# Data Documentation

## 1. Overview

This project studies **breast cancer relapse using gene expression microarray
data** as the basis for association rule mining. The dataset is a
high-dimensional expression matrix (tens of thousands of probe-level features)
measured on a small number of tumor samples, paired with per-patient clinical
follow-up information. The class label of interest is **distant relapse
within the follow-up period** (`relapse`).

The defining characteristic of this dataset is its dimensionality: roughly
**22,000+ expression features measured on only 286 patients** (a
feature-to-sample ratio on the order of 78:1). This extreme p ≫ n structure
is what motivates the project's feature-selection and discretization approach.

## 2. Source

| Field | Value |
|---|---|
| GEO accession | **GSE2034** |
| Title | "Breast cancer relapse free survival" |
| Platform | Affymetrix Human Genome U133A (GEO platform **GPL96**) |
| Organism | *Homo sapiens* (taxid 9606) |
| Submission date | Dec 03 2004 |
| Public release | Feb 23 2005 |
| Last update | Aug 10 2018 |
| Pubmed | 15721472 (original Wang et al. publication) |
| BioProject | PRJNA91859 |
| Original submission | Veridex (contact: Tim Jatkoe, San Diego, CA) |
| Supplementary raw files | `GSE2034_RAW.tar` (per-sample `.CEL.gz` files, via GEO FTP) |

The data was obtained directly from the NCBI Gene Expression Omnibus (GEO).

## 3. Raw Files

Two raw files are stored locally under `data/`:

| File | Size | Lines / Shape | Content |
|---|---|---|---|
| `data/GSE2034_series_matrix.txt` | ~37.7 MB | 22,338 lines | GEO series matrix: full expression table (one row per probe, one column per sample) plus series/sample metadata header |
| `data/GSE2034_clinical.txt` | ~9.7 KB | 286 sample rows | Clinical follow-up table, one row per patient |

### 3.1 Series matrix structure

The series matrix file has a metadata block followed by a TAB-delimited
expression table delimited by `!series_matrix_table_begin` /
`!series_matrix_table_end`:

- **Metadata lines (1–53):** `!Series_*` and `!Sample_*` fields describing the
  series and each sample (title, GEO sample ID, tissue source = "Breast",
  molecule = "total RNA", single channel, global scaling to target intensity 600,
  per-sample recurrence/relapse characteristics, etc.).
- **Expression table:** a matrix of **22,283 probe rows × 286 sample columns**,
  keyed by Affymetrix probe ID (`ID_REF`). Cell values are continuous
  signal intensities (real numbers; images globally scaled to target 600).

### 3.2 Clinical table structure

The clinical file is a plain columnar table (the first 7 lines are
`#`-prefixed column descriptions) with **286 rows × 7 columns**:

| Column | Type | Description |
|---|---|---|
| `PID` | integer | Internal patient ID |
| `GEO_asscession_number` | text | GEO sample ID (GSMxxxxx), sample identifier |
| `lymph_node_status` | text | All samples are "negative" (LN-negative cohort by design) |
| `time_to_relapse` | integer | Time to relapse or last follow-up, months |
| `relapse` | binary (0/1) | 1 = distant metastasis/relapse occurred |
| `ER_Status` | text | ER+ / ER- estrogen receptor status |
| `Brain_relapses` | binary (0/1) | 1 = brain relapse observed |

## 4. Dataset Composition

### 4.1 Sample counts

| Subset | Count |
|---|---|
| Total patients | 286 |
| Relapse-free (relapse = 0) | 179 |
| Relapsed (relapse = 1) | 107 |
| ER+ | 209 |
| ER- | 77 |
| Brain relapses (yes) | 10 |
| Lymph-node negative | 286 (all) |

`time_to_relapse` ranges from **2 to 171 months**.

### 4.2 Feature counts

| Feature group | Count |
|---|---|
| Gene expression probes (Affymetrix probe IDs) | **22,215** |
| &nbsp;&nbsp;– plain IDs (`\d+_at`, e.g. `1053_at`) | 11,774 |
| &nbsp;&nbsp;– suffixed IDs (`\d+_{s,x,i,r,g,f}_at`, e.g. `1007_s_at`) | 10,441 |
| Affymetrix QC control probes (`AFFX-*`) | 68 |
| **Total expression-matrix columns** | 22,284 |

### 4.3 Column groups

The data contains distinct column classes that must not be treated
interchangeably:

- **Gene expression probes** — the ~22,215 candidate ARM items. Continuous
  intensities per sample.
- **Affymetrix control probes** (`AFFX-BioB-*`, `AFFX-BioC-*`, `AFFX-CreX-*`,
  `AFFX-DapX-*`, `AFFX-HUMGAPDH*`, `AFFX-HUMISGF3A*`, `AFFX-HUMRGE*`,
  `AFFX-LysX-*`, `AFFX-PheX-*`, `AFFX-ThrX-*`, `AFFX-TrpnX-*`, `AFFX-r2-*`,
  `AFFX-M27830*`, and others) — spike-in / hybridization QC controls, not
  tumor biology. **Excluded from all analysis.**
- **Clinical covariates** — `lymph_node_status`, `ER_Status` (already
  categorical; not discretized).
- **Outcome / leakage-risk columns** — `relapse` (class label),
  `time_to_relapse`, `Brain_relapses` (outcome-adjacent, leakage risk),
  `GEO_accession_number` (sample identifier, not a feature).


## 5. Citation

Data citation:

> Wang Y, Klijn JGM, Zhang Y, Sieuwerts AM, et al. Gene-expression profiles
> to predict distant metastasis of lymph-node-negative primary breast cancer.
> *Lancet* 2005;365(9460):671-679. (GEO series GSE2034.)

GEO record: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE2034
