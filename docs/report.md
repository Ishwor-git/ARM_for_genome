# Association Rule Mining on High-Dimensional Breast Cancer Microarray Data

**Authors:** Ishwor Raj Pokharel, Suprim Koirala
**Course:** Data Mining — 1-month project

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Data](#2-data)
3. [Preprocessing](#3-preprocessing)
4. [Methodology and Implementation](#4-methodology-and-implementation)
5. [Results](#5-results)
6. [Conclusion](#6-conclusion)

---

## 1. Introduction

Association Rule Mining (ARM) is a fundamental technique in Knowledge Discovery in
Databases (KDD) for identifying relationships and co-occurrence patterns among variables
in large datasets. Originally developed for transactional (market-basket) analysis, ARM
finds combinations of features that occur together frequently and derives interpretable
rules of the form `{item_A, item_B, ...} → {item_C}` from them. Algorithms such as Apriori
and FP-Growth made it possible to systematically search the combinatorial space of
possible itemsets and mine association rules at scale. Although traditionally associated
with retail analytics, the ability of ARM to discover multi-feature relationships has
motivated its application to more complex domains, including biomedical research and
genomics, where associations among multiple genes, molecular markers, or phenotypic
characteristics may reveal insights that univariate analyses cannot capture.

Genomic datasets, however, differ substantially from the transactional datasets for which
classical ARM was designed. High-throughput technologies such as microarrays measure the
expression levels of tens of thousands of genes across a relatively small number of
biological samples, producing a *high-dimensional* setting in which the number of features
*p* greatly exceeds the number of observations *n* (p ≫ n). Microarray measurements are
also continuous and must therefore be discretized before conventional ARM algorithms can
be applied. Despite these challenges, biological phenomena often depend on *combinations*
of genes rather than isolated features, so ARM provides a natural framework for
investigating recurring combinations of gene-expression states and their association with
clinical characteristics.

The same characteristics that make genomic data attractive for pattern mining also create
substantial computational and analytical challenges. Classical ARM operates over a
combinatorial search space whose size grows rapidly with the number of items: with tens of
thousands of genes, even a small fraction of them yields an enormous number of possible
itemsets, leading to excessive time and memory consumption. Stringent minimum-support
thresholds eliminate potentially interesting patterns, while relaxing them produces a huge
volume of redundant, weak, or uninterpretable rules. The central challenge is therefore not
simply whether ARM *can* generate rules from genomic data, but whether it can do so within
practical computational limits while producing patterns that remain analytically useful.

This project answers that question empirically on a breast-cancer microarray dataset
(GSE2034, 286 samples × 22,215 probes). A supervised pipeline was executed — cleaning,
class-aware feature selection, discretization, Apriori and FP-Growth mining, base-rate-
adjusted rule filtering, class-relative support thresholding, and a permutation null test —
to discover interpretable gene co-expression rules associated with disease relapse, and to
locate the point at which classical ARM becomes intractable on this data.

---

## 2. Data

### 2.1 Source

| Field | Value |
|---|---|
| GEO accession | **GSE2034** — "Breast cancer relapse free survival" |
| Platform | Affymetrix Human Genome U133A (GEO platform **GPL96**) |
| Organism | *Homo sapiens* |
| Original submission | Veridex (contact: Tim Jatkoe, San Diego, CA) |
| Public release | Feb 23 2005 |
| Pubmed | 15721472 (Wang et al.) |

### 2.2 Dataset statistics

| Property | Value |
|---|---|
| Total samples (*n*) | 286 |
| Raw probes | ~22,289 |
| Probes after cleaning | 22,215 |
| *p*:*n* ratio | ~78:1 |
| Target variable | `relapse` (binary) |
| Class distribution | 179 no-relapse / 107 relapse |
| Class ratio | ~1.67:1 |
| Data format | Continuous log-intensity matrix |

The dataset's defining statistical challenge is the extreme feature-to-sample ratio
(*p* ≈ 22,215, *n* = 286, roughly 78:1) combined with moderate class imbalance
(~1.67:1). Both properties directly motivate the feature-selection and class-relative
mining decisions described below.

### 2.3 Feature types

The raw feature space is heterogeneous, combining platform control probes, gene
expression probes, and clinical covariates:

| Feature type | Count | Retained? |
|---|---|---|
| AFFX control probes | 68 | No — removed |
| Gene expression probes | 22,215 | Yes — MI/χ²/t-test ranked, top-*k* selected |
| Clinical covariates | 2 | Yes — kept as-is |
| Leakage-risk columns | 3 | No — removed |
| Target label | 1 | Yes — one-hot encoded |

- **AFFX control probes** (`AFFX-BioB-*`, `AFFX-CreX-*`, `AFFX-HUMGAPDH*`, etc.) are
  spike-in / hybridization QC controls, not tumor biology. Excluded from all analysis.
- **Gene expression probes** are the primary ARM items.
- **Clinical covariates** (`lymph_node_status`, `ER_Status`) are established prognostic
  factors retained for potential inclusion as items.
- **Leakage-risk columns** (`time_to_relapse`, `Brain_relapses`, `GEO_accession_number`)
  were excluded: they are outcome-adjacent (leak the label) or a sample identifier.

### 2.4 Target variable and leakage rules

`relapse` is the intended class label (1 = distant metastasis/relapse occurred).
`time_to_relapse` and `Brain_relapses` create **leakage risk** if included as predictor
items — a rule using `time_to_relapse` to "predict" `relapse` is close to circular.
`GEO_accession_number` is a sample identifier and must never be treated as a feature.

---

## 3. Preprocessing

> Very short summary; full commands and file layouts are in `preprocessing.md`.

- **Parse & merge** raw GEO files (`dataview.py`): extract the series-matrix expression
  table, transpose to samples-as-rows, merge with clinical data → `final_data.csv`
  (286 × 22,290).
- **Clean** (`clean_features.py`): drop 68 `AFFX-*` control probes and the leakage
  columns `time_to_relapse`, `Brain_relapses`, `GEO_accession_number` → `cleaned_data.csv`
  (286 × 22,219).
- **Feature selection** (`feature_selection.py`): score all 22,215 probes against the
  `relapse` label with three class-aware methods — mutual information, chi-square,
  two-sample *t*-test — and export top-*k* subsets for
  *k* ∈ {100, 150, 200, 250, 300, 350, 400}.
- **Discretize**: per-probe median split `(value > median) → 1/0`, applied to expression
  probes only → binary transaction matrix (286 × *k*).
- **Encode target**: one-hot `relapse_yes` / `relapse_no` items (so both classes are
  mineable as consequents).

**Known limitation (flagged, not fixed):** the per-feature median split forces ~50/50
high/low support for every gene, is unsupervised w.r.t. `relapse`, and discards
expression-magnitude information. It is an acceptable first-pass baseline but must be
read as a limitation (see §6).

---

## 4. Methodology and Implementation

This section documents the association-rule mining methodology executed on the binary
matrices produced in §3, with the important code for each stage. Every stage was applied
independently to the three feature-selection rankings (mutual information, chi-square,
*t*-test) and to both exported subset sizes *k* ∈ {100, 150}, so the effect of
feature-count sensitivity can be separated from the composition of the selected gene set.
Frequent-itemset mining is executed twice — once with **Apriori** and once with
**FP-Growth**, both via `mlxtend.frequent_patterns` (the algorithms are *not*
hand-implemented). Support, confidence, and lift are computed by the same library.

### 4.1 Column cleaning

The first preprocessing stage removes columns that are not valid ARM items:
Affymetrix QC controls and leakage-risk / non-feature columns.

`src/preprocessing/clean_features.py`:

```python
import re
from pathlib import Path
import pandas as pd


LEAKAGE_COLS = {
    "time_to_relapse",
    "Brain_relapses",
    "GEO_accession_number",
    "GEO_asscession_number",      # source typo variant
}

CLINICAL_COLS = {"lymph_node_status", "ER_Status"}
TARGET_COL = "relapse"


def clean_features(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns that are not valid ARM items."""
    result = df.copy()

    known_typos = {"GEO_asscession_number": "GEO_accession_number"}
    result = result.rename(columns=known_typos)

    affx_pattern = re.compile(r"^AFFX-", re.IGNORECASE)
    affx_cols = [c for c in result.columns if affx_pattern.match(str(c))]
    leakage_present = LEAKAGE_COLS & set(result.columns)

    if affx_cols:
        result = result.drop(columns=affx_cols)
    if leakage_present:
        result = result.drop(columns=list(leakage_present))

    return result
```

Removal is explicit and pattern-based (never positional), so it does not silently break
if the raw column order changes. Result: `cleaned_data.csv` (286 × 22,219).

### 4.2 Class-aware feature selection

All 22,215 expression probes are scored against the binary `relapse` label using three
methods. Mutual information and chi-square are computed on the *discretized* values;
the *t*-test uses the *continuous* intensities, avoiding discretization information loss
at the cost of comparing means only.

`src/preprocessing/feature_selection.py`:

```python
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from sklearn.feature_selection import chi2, mutual_info_classif

from clean_features import CLINICAL_COLS, TARGET_COL

SEED = 42
METHOD_CHOICES = ["mutual_information", "chi_square", "t_test"]
TOPK_VALUES = list(range(100, 401, 50))          # k = 100, 150, ..., 400


def compute_scores(method, expr_df, expr_discrete, y):
    """Per-gene selection scores (higher = more relevant)."""
    if method == "mutual_information":
        scores = mutual_info_classif(expr_discrete, y, random_state=SEED)
        score_col = "mi_score"
    elif method == "chi_square":
        chi2_stat, _ = chi2(expr_discrete, y)
        scores = chi2_stat
        score_col = "chi2_statistic"
    elif method == "t_test":
        group_0 = expr_df[y == 0].values
        group_1 = expr_df[y == 1].values
        t_stats, _ = ttest_ind(group_1, group_0, axis=0)
        scores = np.abs(t_stats)
        score_col = "abs_t_statistic"
    else:
        raise ValueError(f"Unknown method: {method}")
    return scores, score_col
```

Observed score ranges: MI 0–0.119 (median ≈ 0.019), χ² up to 12.55, |*t*| up to 4.89.

**Cross-method agreement.** Because the three rankings are computed on the same
discretized matrix, their agreement is directly measurable. At *k* = 100 only 2 of 100
genes are selected by all three methods; the three rankings together nominate 264
distinct genes. The MI ranker shares just 7 genes with the other two methods combined,
while χ² and *t*-test agree on 29; at *k* = 150 the picture is essentially unchanged
(3 common genes, union 396). The choice of ranker therefore materially changes the
feature set handed to the miner.

### 4.3 Discretization (per-probe median split)

Each of the 22,215 expression probes is converted independently into a binary item via a
per-probe median split with strict inequality:

```
x' = 1  if  x > median(x_probe)
      0  otherwise
```

so that values equal to the probe's median fall into the low bin:

```python
# Per-feature median split on expression probes only
expr_discrete = expr_df.copy()
for col in expr_discrete.columns:
    median_val = expr_discrete[col].median()
    expr_discrete[col] = (expr_discrete[col] > median_val).astype(int)
```

This produces an approximately — but not exactly — balanced split per gene (observed
high-bin support in [0.489, 0.500] across selected genes). The split is unsupervised with
respect to `relapse`; it is a deliberate first-pass baseline rather than a final
discretization method.

### 4.4 One-hot target re-encoding

A critical detail: `mlxtend` turns a bare binary column into a *single* item `relapse`
that only ever represents relapse = yes, leaving no item for relapse = no. Before mining,
the target is re-encoded into two mutually exclusive one-hot items. This step is shared
verbatim by the Apriori and FP-Growth runners:

```python
TARGET_COL = "relapse"


def one_hot_encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """Replace the bare binary target column with two one-hot columns."""
    if TARGET_COL not in df.columns:
        return df

    encoded = df.copy()
    target = encoded.pop(TARGET_COL).astype(int)
    encoded[f"{TARGET_COL}_yes"] = (target == 1).astype(bool)
    encoded[f"{TARGET_COL}_no"] = (target == 0).astype(bool)
    return encoded
```

Input shape grows from 286 × 101 (top100 + bare `relapse`) to 286 × 102
(top100 + `relapse_yes` + `relapse_no`).

### 4.5 Apriori mining

`src/ARM/apriori/run_apriori.py` — the mining core:

```python
import argparse
import json
import time
from pathlib import Path

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# ... argparse for --input, --output-dir, --min-support, --max-len,
#     --min-confidence (defaults: 0.1, 3, 0.5) ...

# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------
df = pd.read_csv(input_path)

# ---------------------------------------------------------
# Re-encode target into one-hot relapse_yes / relapse_no items
# ---------------------------------------------------------
if TARGET_COL in df.columns:
    df = one_hot_encode_target(df)

# ---------------------------------------------------------
# Run Apriori
# ---------------------------------------------------------
start_time = time.time()
frequent_itemsets = apriori(
    df,
    min_support=args.min_support,
    use_colnames=True,
    max_len=args.max_len,
)
apriori_time = time.time() - start_time

# ---------------------------------------------------------
# Generate association rules
# ---------------------------------------------------------
rule_start_time = time.time()
if len(frequent_itemsets) > 0:
    rules = association_rules(
        frequent_itemsets,
        metric="confidence",
        min_threshold=args.min_confidence,
    )
else:
    rules = pd.DataFrame()
rule_time = time.time() - rule_start_time

# ---------------------------------------------------------
# Save results + metadata
# ---------------------------------------------------------
frequent_itemsets.to_csv(output_dir / "frequent_itemsets.csv", index=False)
rules.to_csv(output_dir / "association_rules.csv", index=False)
```

Run command and parameters:

```bash
python src/ARM/apriori/run_apriori.py \
  --input "data/final/feature_selection/<method>/top100_genes_discretized.csv" \
  --output-dir "data/final/ARM/apriori_v2/<method>/top100" \
  --min-support 0.1 \
  --max-len 3 \
  --min-confidence 0.5
```

These thresholds deliberately favour recall over precision: at *p* ≫ *n* the primary
failure mode of classical ARM is rule *volume*, and the filtering stages in §4.7–§4.10
are what separate signal from noise.

### 4.6 FP-Growth mining

`src/ARM/fpgrowth/run_fpgrowth.py` — structurally identical to the Apriori runner, with
only the algorithm call changed:

```python
import pandas as pd
from mlxtend.frequent_patterns import association_rules, fpgrowth

# ... identical data loading and one-hot target re-encoding ...

start_time = time.time()
frequent_itemsets = fpgrowth(
    df,
    min_support=args.min_support,
    use_colnames=True,
    max_len=args.max_len,
)
fpgrowth_time = time.time() - start_time

rule_start_time = time.time()
if len(frequent_itemsets) > 0:
    rules = association_rules(
        frequent_itemsets,
        metric="confidence",
        min_threshold=args.min_confidence,
    )
else:
    rules = pd.DataFrame()
rule_time = time.time() - rule_start_time

frequent_itemsets.to_csv(output_dir / "frequent_itemsets.csv", index=False)
rules.to_csv(output_dir / "association_rules.csv", index=False)
```

Because both algorithms enumerate the *exact* set of frequent itemsets, their outputs are
identical by construction. The two runs are therefore reported once (§5.1) and compared
only on runtime (§5.2). Run all experiments in one shot:

```bash
./run_fpgrowth_experiment.sh            # uses `python`
PY=python3 ./run_fpgrowth_experiment.sh # override interpreter
```

### 4.7 Class-association rule extraction

A class-association rule (CAR) is defined as any rule whose consequent is a *single* item
in {`relapse_yes`, `relapse_no`}; rules with multi-item or non-class consequents are
discarded. `mlxtend` writes itemsets as `frozenset({...})` strings, which are parsed back
into sets:

`src/ARM/fpgrowth/filter_class_rules_v2.py`:

```python
import re

_FROZENSET_RE = re.compile(r"frozenset\(\{(.*)\}\)", re.DOTALL)
_ITEM_RE = re.compile(r"'([^']*)'")


def parse_itemsets(value):
    m = _FROZENSET_RE.fullmatch(str(value))
    if not m:
        return None
    inner = m.group(1).strip()
    if not inner:
        return frozenset()
    return frozenset(_ITEM_RE.findall(inner))


def assign_class(consequent):
    if consequent in (RELAPSE_YES_ITEMS, RELAPSE_BARE_ITEMS):
        return "relapse_yes"
    if consequent == RELAPSE_NO_ITEMS:
        return "relapse_no"
    return None


def contains_relapse(consequent):
    if not isinstance(consequent, frozenset):
        return False
    return any(item in RELAPSE_ITEM_NAMES for item in consequent)
```

Three quality metrics are used to rank retained CARs:

- **Raw statistics**: support, confidence, lift. Antecedent support must be ≥ 0.07
  (~20 of 286 samples) so sample-rare antecedents cannot manufacture high-confidence
  rules.
- **Certainty factor (CF)**: a base-rate-adjusted confidence (see §4.8).
- **Leverage**: `support(A ∪ C) − support(A)·support(C)` — a co-occurrence check.
  Negative leverage flags rules whose joint occurrence is rarer than independence would
  predict (not sample-backed).

### 4.8 Certainty factor (base-rate-adjusted confidence)

A rule's confidence must be judged against the *base rate* of its consequent's class.
Letting *s* denote the consequent's marginal support (the class base rate), the
certainty factor is:

```
CF = (conf − s) / (1 − s)   when conf ≥ s   (beating base rate)
CF = (conf − s) / s         when conf <  s   (worse than base rate)
```

CF = 0 means confidence equals the class base rate (no signal beyond chance);
CF = 1 means perfect confidence on a rare class. Because CF is normalized against each
consequent's own rate, `relapse_yes` and `relapse_no` rules are directly comparable
despite the 107:179 class imbalance.

`src/ARM/fpgrowth/filter_class_rules_v2.py`:

```python
def certainty_factor(row):
    """Base-rate-adjusted confidence (CF), in [-1, 1]."""
    confidence = row["confidence"]
    consequent_support = row["consequent support"]
    if confidence >= consequent_support:
        denominator = 1.0 - consequent_support
    else:
        denominator = consequent_support
    if denominator == 0:
        return 0.0
    return (confidence - consequent_support) / denominator


# Annotate every class rule with its CF, then summarize per class
df = df.copy()
df["CF"] = df.apply(certainty_factor, axis=1)
```

Rules that do not beat the base rate (CF ≤ 0) are dropped in a post-hoc filter — without
re-mining:

```python
def process_cf_positive_filter(method, rules_path, out_dir):
    """Drop class rules with CF <= 0 (worse-than-baseline noise)."""
    df = pd.read_csv(rules_path)
    positive = df[df["CF"] > 0].copy()
    # ... per-class dropped fractions and CF-bin tallies ...
```

Run:

```bash
python src/ARM/fpgrowth/filter_class_rules_v2.py \
  --method mutual_information chi_square t_test \
  --base-dir data/final/ARM/fpgrowth_v2

# post-hoc CF > 0 filter (no re-mining)
python src/ARM/fpgrowth/filter_class_rules_v2.py \
  --cf-positive \
  --base-dir data/final/ARM/fpgrowth_v2
```

### 4.9 Class-relative support thresholding

A single global support threshold applied to the full 286-sample dataset systematically
disadvantages the minority class: itemsets confined to the 107 relapse-yes patients must
clear a bar defined against the majority base rate. This is the multiple-minimum-support
problem of imbalanced ARM. The methodology therefore mines each class subset
*independently* at a matched *absolute* support — a floor of **20 samples** applied to
both classes and converted to a per-subset fraction,
`min_support = 20 / n_class` (0.187 for the 107 relapse-yes patients, 0.112 for the 179
relapse-no patients):

`src/ARM/fpgrowth/run_class_relative_fpgrowth.py`:

```python
from mlxtend.frequent_patterns import fpgrowth

MIN_ABS_SUPPORT = 20
MAX_LEN = 3


def split_by_class(df):
    """Split transactions into relapse_yes / relapse_no gene-only subsets."""
    cols = list(df.columns)
    class_col = None
    if CLASS_YES in cols and CLASS_NO in cols:
        yes_mask = df[CLASS_YES].astype(bool)
        no_mask = df[CLASS_NO].astype(bool)
        class_cols = [CLASS_YES, CLASS_NO]
    elif TARGET_COL in cols:
        yes_mask = df[TARGET_COL].astype(bool)
        no_mask = ~yes_mask
        class_cols = [TARGET_COL]
    else:
        raise ValueError("No class column found in input columns.")
    gene_cols = [c for c in cols if c not in class_cols]
    return df.loc[yes_mask, gene_cols], df.loc[no_mask, gene_cols]


def mine_subset(subset_df, min_support, max_len):
    if subset_df.empty:
        return pd.DataFrame(columns=["support", "itemsets"])
    return fpgrowth(
        subset_df,
        min_support=min_support,
        use_colnames=True,
        max_len=max_len,
    )
```

From the two per-class itemset sets, an itemset is called *class-exclusive* if it is
frequent in one class and occurs in fewer than 20 samples in the other; itemsets frequent
in both are *shared*. Exclusive itemsets are the candidates for class-characteristic
signal:

```python
def find_exclusive(needle, haystack_counts):
    """Itemsets frequent in `needle` but below the abs threshold in the other."""
    exclusive = []
    for itemset, info in needle.items():
        count_here = info["count"]
        count_other = haystack_counts.get(itemset, {}).get("count", 0)
        if count_other < MIN_ABS_SUPPORT:
            exclusive.append((itemset, count_here, count_other))
    return exclusive
```

The per-class mining loop converts each itemset's support fraction back to an exact
absolute count (`count = round(support * n_class)`), since `min_support = 20/n` is
exactly integral:

```python
for class_name, subset_df in ((CLASS_YES, yes_df), (CLASS_NO, no_df)):
    n = n_class[class_name]
    min_support = min_abs / n
    fi = mine_subset(subset_df, min_support, max_len)

    itemsets = {}
    for _, row in fi.iterrows():
        itemset = parse_itemset(row["itemsets"])
        if itemset is None:
            continue
        count = round(row["support"] * n)
        itemsets[itemset] = {"count": count, "support": float(row["support"])}
```

Run:

```bash
python src/ARM/fpgrowth/run_class_relative_fpgrowth.py \
  --method mutual_information chi_square t_test \
  --top-k top100 \
  --base-dir data/final/ARM/fpgrowth_v2
```

### 4.10 Permutation null test for rule significance

Exclusivity counts grow mechanically with itemset volume, so raw exclusive counts are not
evidence of real class differences. A permutation null test quantifies how much
class-exclusivity would appear *by chance alone*: the relapse labels are shuffled 20
times, preserving the exact 107:179 class sizes, and the identical class-relative mining
procedure is re-run on each shuffled matrix. A fixed master seed (20260806) keeps the
shuffle sequence reproducible.

`src/ARM/fpgrowth/permutation_test_class_relative.py`:

```python
import numpy as np

MASTER_SEED = 20260806
N_SHUFFLES = 20


def shuffled_labels(rng, n_total, n_yes):
    """Permutation-preserving assignment: n_yes ones / rest zeros."""
    labels = np.zeros(n_total, dtype=int)
    labels[:n_yes] = 1
    rng.shuffle(labels)
    return labels


def class_relative_metrics(df, min_abs, max_len):
    """Run the identical class-relative FP-Growth procedure on `df`.

    Mirrors run_method()'s subset mining + exclusivity comparison on the
    given dataframe (real or label-shuffled).
    """
    yes_df, no_df = split_by_class(df)
    subsets = {CLASS_YES: (yes_df, len(yes_df)),
               CLASS_NO: (no_df, len(no_df))}

    found = {}
    for class_name, (subset_df, n) in subsets.items():
        min_support = min_abs / n
        fi = mine_subset(subset_df, min_support, max_len)
        found[class_name] = _fsrows(fi, n)

    yes_excl = find_exclusive(found[CLASS_YES], found[CLASS_NO])
    no_excl = find_exclusive(found[CLASS_NO], found[CLASS_YES])

    return {
        "total_yes": len(found[CLASS_YES]),
        "total_no": len(found[CLASS_NO]),
        "yes_excl": len(yes_excl),
        "no_excl": len(no_excl),
    }
```

The empirical *p*-value for an observed exclusive count is the fraction of the 20 null
runs whose count reaches or exceeds it (0.05 granularity):

```python
rng = np.random.default_rng(MASTER_SEED)
real = class_relative_metrics(df_real, MIN_ABS_SUPPORT, MAX_LEN)

shuffled_counts = []
for _ in range(N_SHUFFLES):
    y_shuffled = shuffled_labels(rng, n_total, n_yes)
    df_shuffled = df_real.copy()
    # overwrite class columns with shuffled labels
    df_shuffled[CLASS_YES] = y_shuffled
    df_shuffled[CLASS_NO] = 1 - y_shuffled
    shuffled_counts.append(class_relative_metrics(df_shuffled, MIN_ABS_SUPPORT, MAX_LEN))

p_yes = sum(s["yes_excl"] >= real["yes_excl"] for s in shuffled_counts) / N_SHUFFLES
p_no  = sum(s["no_excl"]  >= real["no_excl"]  for s in shuffled_counts) / N_SHUFFLES
```

Run:

```bash
python src/ARM/fpgrowth/permutation_test_class_relative.py \
  --method mutual_information \
  --top-k top100 \
  --base-dir data/final/ARM/fpgrowth_v2
```

### 4.11 Characterization of the validated signal

The class-relative output is dominated by three-item itemsets, which are hard to
interpret. To characterize the signal the permutation test validates, the
relapse-no-exclusive *two-item* itemsets are extracted; for each pair the exact support
inside each class is recomputed by direct counting, and the genes participating in many
pairs are tallied.

`src/ARM/fpgrowth/extract_relapse_no_2itemsets.py`:

```python
def absolute_count(df, itemset):
    """Exact absolute sample count of an itemset (all items present)."""
    if not itemset:
        return int(len(df))
    cols = list(itemset)
    mask = df[cols].astype(bool).all(axis=1)
    return int(mask.sum())
```

---

## 5. Results

Results are reported primarily for the top-100 subsets; §5.7 quantifies the comparative
change when the feature count is raised to 150. All parameters: min support 0.10,
max itemset length 3, min confidence 0.50.

### 5.1 Scale and feasibility of classical ARM

| Selection method | *k* | Frequent itemsets | Association rules |
|---|---|---|---|
| Mutual information | 100 | 154,790 | 316,546 |
| Mutual information | 150 | 518,407 | 1,039,766 |
| Chi-square | 100 | 133,889 | 285,359 |
| Chi-square | 150 | 454,596 | 961,672 |
| *t*-test | 100 | 134,186 | 276,041 |
| *t*-test | 150 | 453,256 | 913,114 |

Counts are **identical for Apriori and FP-Growth** (both enumerate the exact frequent
itemsets). At *k* = 100 the miner produces 1.3–1.5 × 10⁵ frequent itemsets and
2.8–3.2 × 10⁵ rules per method; at *k* = 150 the counts roughly **triple**. The practical
bottleneck of classical ARM on this dataset is therefore rule *volume*, not execution
time.

> **Note:** running Apriori/FP-Growth beyond ~150 features is not computationally
> feasible in this pipeline at the chosen thresholds.

### 5.2 Apriori vs FP-Growth runtime

The two implementations are interchangeable at the level of rule output (an internal
consistency check on every downstream result). What differs is execution time:

| Selection method | *k* | Apriori (s) | FP-Growth (s) | Slowdown |
|---|---|---|---|---|
| Mutual information | 100 | 8.9 | 23.8 | 2.7× |
| Mutual information | 150 | 12.2 | 106.0 | 8.7× |
| Chi-square | 100 | 9.0 | 20.2 | 2.3× |
| Chi-square | 150 | 12.2 | 110.1 | 9.0× |
| *t*-test | 100 | 9.1 | 22.0 | 2.4× |
| *t*-test | 150 | 12.4 | 107.0 | 8.6× |

At *k* = 100 FP-Growth is ~2.5× slower than Apriori; at *k* = 150 the gap widens to
~9×. The median-split input is *dense* — about half of the *k* items are present in every
transaction — so the FP-tree built by `mlxtend`'s pure-Python FP-Growth grows huge and
becomes the bottleneck, while Apriori's candidate generation stays bounded by the small
maximum itemset length. On this dense, high-dimensional binary data the usual expectation
that FP-Growth beats Apriori **does not hold**.

### 5.3 Class-association rules: raw counts vs base-rate-adjusted quality

At *k* = 100 the class-rule counts differ sharply across selection methods. In raw counts
the majority class (`relapse_no`) dominates by 13–46:1:

| Method | relapse_yes rules | relapse_no rules | yes CF>0.4 | no CF>0.4 | no dropped at CF≤0 (%) |
|---|---|---|---|---|---|
| Mutual information | 107 | 4,954 | 1 | 175 | 29.5 |
| Chi-square | 982 | 4,040 | 130 | 1,137 | 22.2 |
| *t*-test | 1,601 | 3,470 | 24 | 415 | 38.6 |

The CF base-rate correction shows the apparent "zero-minority-rule" phenomenon is largely
an artifact: **no relapse_yes rule from any method fails the better-than-baseline test**
(0 dropped at CF ≤ 0), whereas 22–39% of relapse_no rules are dropped as no better than
the majority base rate. At the stronger CF > 0.4 bin the minority rules become scarce
(1, 130, 24), so the minority-class signal is *present but genuinely sparse* rather than
absent. Every rule that passes the CF > 0 filter has positive leverage — it reflects a
co-occurrence stronger than chance, not a single-patient artifact.

### 5.4 Class-relative exclusive itemsets

Mining each class at matched absolute support (20 samples) yields per-class frequent
itemsets and exclusivity profiles that differ across selection methods:

| Method | *k* | FI yes | FI no | Excl. yes | Excl. no | Shared |
|---|---|---|---|---|---|---|
| Mutual info. | 100 | 11,576 | 139,582 | 1,865 | 129,871 | 9,711 |
| Mutual info. | 150 | 33,785 | 461,921 | 6,081 | 434,217 | 27,704 |
| Chi-square | 100 | 23,886 | 107,623 | 10,855 | 94,592 | 13,031 |
| Chi-square | 150 | 58,323 | 380,353 | 29,117 | 351,147 | 29,206 |
| *t*-test | 100 | 48,545 | 94,048 | 18,790 | 64,293 | 29,755 |
| *t*-test | 150 | 153,274 | 314,616 | 62,366 | 223,708 | 90,908 |

Mutual information produces a large volume of relapse-no-exclusive itemsets relative to
relapse-yes-exclusive; chi-square shifts the balance; the *t*-test produces the most
balanced profile. At *k* = 150 all counts grow 3–4×. These raw volumes are not yet
evidence of signal — that requires the permutation evaluation.

### 5.5 Permutation-based significance

Empirical *p*-values for the two exclusivity counts (20 label shuffles; *p* < 0.05 means
no null run reached the observed count):

| Method | *k* | Excl. yes | *p*(yes) | Excl. no | *p*(no) |
|---|---|---|---|---|---|
| Mutual info. | 100 | 1,865 | 0.35 | 129,871 | **0.00** |
| Mutual info. | 150 | 6,081 | 0.40 | 434,217 | 0.05 |
| Chi-square | 100 | 10,855 | **0.00** | 94,592 | 0.05 |
| Chi-square | 150 | 29,117 | **0.00** | 351,147 | 0.05 |
| *t*-test | 100 | 18,790 | **0.00** | 64,293 | 1.00 |
| *t*-test | 150 | 62,366 | **0.00** | 223,708 | 1.00 |

- **Mutual information**: the relapse-yes-exclusive count (1,865 at *k*=100) falls inside
  the null range [424, 3,438] (*p* = 0.35; at *k*=150, 6,081 in [2,127, 11,758],
  *p* = 0.40) — consistent with chance at both sizes. The relapse-no-exclusive signal is
  significant at *k*=100 (129,871, *p* = 0.00) but degrades to marginal at *k*=150
  (*p* = 0.05).
- **Chi-square**: the *minority-class* exclusive signal is significant at both sizes
  (*p* = 0.00); the majority-class signal is marginal (*p* = 0.05).
- ***t*-test**: the minority-class signal is strongly significant at both sizes
  (*p* = 0.00), while majority-class exclusivity is entirely consistent with chance
  (*p* = 1.00).

**Central cross-method finding:** *which class carries the statistically validated signal
depends on the feature-selection method.* Mutual information yields a validated
majority-class (relapse-no) signal with no minority signal; the chi-square and *t*-test
filters surface a validated minority-class (relapse-yes) signal instead.

### 5.6 Characterized validated itemsets

The relapse-no-exclusive two-item itemsets are the most interpretable validated output:

| Method | *k* | 2-item itemsets | Distinct genes |
|---|---|---|---|
| Mutual info. | 100 | 745 | 95 |
| Mutual info. | 150 | 1,486 | 147 |
| Chi-square | 100 | 1,559 | 99 |
| Chi-square | 150 | 3,994 | 150 |
| *t*-test | 100 | 992 | 97 |
| *t*-test | 150 | 2,105 | 148 |

The count approximately doubles from *k*=100 to *k*=150 for every method while the number
of distinct genes involved grows to nearly the full complement of the selected set.

### 5.7 Feature-count sensitivity: *k* = 100 → *k* = 150

Total rule volume triples for every method, but *quality* — measured by permutation
significance and validated two-item signal — behaves differently:

| Method | *k* | Total rules | *p*(yes) | *p*(no) | No-excl. 2-items |
|---|---|---|---|---|---|
| Mutual info. | 100 | 316,546 | 0.35 | 0.00 | 745 |
| Mutual info. | 150 | 1,039,766 | 0.40 | 0.05 | 1,486 |
| Chi-square | 100 | 285,359 | 0.00 | 0.05 | 1,559 |
| Chi-square | 150 | 961,672 | 0.00 | 0.05 | 3,994 |
| *t*-test | 100 | 276,041 | 0.00 | 1.00 | 992 |
| *t*-test | 150 | 913,114 | 0.00 | 1.00 | 2,105 |

- **Chi-square and *t*-test**: the minority-class exclusive signal remains
  permutation-significant (*p* = 0.00) at *k* = 150, and the validated relapse-no
  two-item itemsets grow from 1,559 → 3,994 (χ²) and 992 → 2,105 (*t*-test). The extra
  50 features buy a roughly doubled volume of validated signal without degrading its
  significance.
- **Mutual information**: the additional features increase raw volume but *weaken* the
  validated signal — the relapse-no-exclusive significance drops from *p* = 0.00 to
  marginal *p* = 0.05 at *k* = 150, and the minority class stays non-significant.

In summary, raising the feature count from 100 to 150 multiplies raw output 3–4× but only
translates into doubled *validated* signal for the chi-square and *t*-test rankings;
under mutual-information selection the majority-class exclusivity becomes marginal. This
motivates treating "improvement" in terms of permutation-validated signal rather than
rule count.

### 5.8 Known limitations

1. **Median-split discretization** discards expression-magnitude information and forces
   every gene into a near-50/50 split, inflating the risk of spurious co-occurrences (§4.3).
2. **Same-cohort evaluation**: feature selection, class-relative mining, and the
   permutation test all operate on the same 286 samples; the permutation null is internal
   and does not test generalization.
3. **Permutation resolution**: 20 shuffles give a *p*-value granularity of 0.05;
   "p = 0.00" means no null run reached the observed count, so 0.05-level conclusions are
   indicative rather than exact.
4. **Class imbalance**: the majority-class advantage in raw rule counts is a base-rate
   artifact; after the CF correction minority rules exist but are sparse at high CF.

---

## 6. Conclusion

A working end-to-end pipeline — cleaning, discretization, class-aware feature selection,
classical Apriori-based class-association mining, base-rate-adjusted rule filtering,
class-relative thresholding, and a permutation null test — has been executed on the
286-sample breast-cancer microarray dataset, for all three feature-selection rankings and
both top-100 and top-150 subsets.

Apriori is computationally tractable on this dataset but produces hundreds of thousands
of rules that must be filtered by class-relative quality metrics. The CF correction
resolves the apparent absence of minority-class rules into genuine sparseness: the
`relapse_yes` signal is present but thin at high quality. The permutation test shows that
the statistically validated signal is **method-dependent** — mutual information supports a
majority-class (relapse-no) signal, while chi-square and the *t*-test support a
minority-class (relapse-yes) signal. Moving from 100 to 150 features roughly doubles the
validated two-item signal for the chi-square and *t*-test rankings while degrading mutual
information's majority-class significance, demonstrating that the value of additional
features must be assessed by validated signal count, not raw rule volume.

An independent FP-Growth implementation reproduced every result identically while being
several times slower on this dense data (§5.2), confirming the Apriori findings and
showing that on dense high-dimensional binary inputs the expected FP-Growth speed
advantage does not materialize. Higher feature counts (the exported top-200–top-400
subsets), an ML baseline classifier benchmark, and rule-stability resampling remain as
follow-on work.

---

## References

- Wang Y, Klijn JGM, Zhang Y, Sieuwerts AM, et al. Gene-expression profiles to predict
  distant metastasis of lymph-node-negative primary breast cancer. *Lancet*
  2005;365(9460):671-679. (GEO series GSE2034.)
- Agrawal R, Srikant R. Fast algorithms for mining association rules. *VLDB* 1994.
- Han J, Pei J, Yin Y. Mining frequent patterns without candidate generation.
  *SIGMOD* 2000.
- Liu B, Hsu W, Ma Y. Integrating classification and association rule mining.
  *KDD* 1998.
- Raschka S. MLxtend: Providing machine learning and data science utilities and
  extensions to Python's scientific computing stack. *JOSS* 2018.
- Pedregosa F, et al. Scikit-learn: Machine learning in Python. *JMLR* 2011.