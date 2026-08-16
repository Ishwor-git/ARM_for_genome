# FP-Growth Association Rule Mining

Steps and commands for mining association rules from the discretized,
feature-selected data using `mlxtend.frequent_patterns` FP-Growth, then
filtering the results down to class-association rules that predict `relapse`.

Run preprocessing first so the inputs below exist.

## Environment

- Python 3.13
- `pandas`, `mlxtend` (0.25.0), `scipy`/`sklearn` used upstream
- `mlxtend.frequent_patterns.fpgrowth` and `association_rules` — the FP-Growth
  algorithm is not hand-implemented.

## Files

| Step | Script | Run |
|---|---|---|
| 1. Mine frequent itemsets + rules | `src/ARM/fpgrowth/run_fpgrowth.py` | see below |
| 2. Filter to class-association rules | `src/ARM/fpgrowth/filter_class_rules.py` | see below |
| 3. CF-based class-rule evaluation | `src/ARM/fpgrowth/filter_class_rules_v2.py` | see below |
| 4. Class-relative mining | `src/ARM/fpgrowth/run_class_relative_fpgrowth.py` | see below |
| 5. Permutation null test | `src/ARM/fpgrowth/permutation_test_class_relative.py` | see below |
| 6. Extract 2-item signal | `src/ARM/fpgrowth/extract_relapse_no_2itemsets.py` | see below |

The full pipeline (Steps 1–6 for all 3 selection methods × `top100`/`top150`, plus a
backfill of the missing `apriori_v2` top150 rule runs) can be run in one shot with:

```bash
./run_fpgrowth_experiment.sh            # uses `python`
PY=python3 ./run_fpgrowth_experiment.sh # override interpreter
```

---

## Step 1 — `run_fpgrowth.py`

Mines frequent itemsets and generates association rules from a discretized
input file, and writes results + run metadata.

```bash
python src/ARM/fpgrowth/run_fpgrowth.py \
  --input <path> \
  --output-dir <path> \
  --min-support 0.1 \
  --max-len 3 \
  --min-confidence 0.5
```

Arguments and defaults:

| Arg | Default | Notes |
|---|---|---|
| `--input` | `data/final/feature_selection/top100_genes_discretized.csv` | Discretized top-k genes + `relapse` column |
| `--output-dir` | `data/final/ARM/fpgrowth/top100` | Created if missing |
| `--min-support` | `0.1` | FP-Growth minimum support |
| `--max-len` | `3` | Maximum itemset length |
| `--min-confidence` | `0.5` | Rule metric threshold |

Paths resolve relative to the project root.

Behavior:
- Loads the discretized CSV (samples × items, 0/1).
- **Target re-encoding:** if a `relapse` (bare 0/1) column is present it is
  removed and replaced by two mutually exclusive one-hot columns `relapse_yes`
  and `relapse_no` (booleans). This is required because mlxtend turns a bare
  binary column into a single item `relapse` that only represents relapse=yes,
  leaving no item for relapse=no.
- Runs `fpgrowth(min_support, use_colnames=True, max_len)`.
- Generates rules via `association_rules(itemsets, metric="confidence",
  min_threshold=min_confidence)`.
- Writes outputs and a metadata JSON.

Outputs (in `--output-dir`):

| File | Content |
|---|---|
| `frequent_itemsets.csv` | Columns: `itemsets` (frozenset of item names), `support` |
| `association_rules.csv` | Columns: `antecedents`, `consequents`, `antecedent support`, `consequent support`, `support`, `confidence`, `lift`, `representativity`, `leverage`, `conviction`, `zhangs_metric`, `jaccard`, `certainty`, `kulczynski` |
| `run_metadata.json` | `input_file`, `input_shape`, target-encoding info, `min_support`, `max_len`, `min_confidence`, `n_frequent_itemsets`, `n_association_rules`, FP-Growth / rule-gen / total runtime in seconds |

### Reproduce all runs

Run for each selection method × top-k input. The top100/top150 discretized
inputs are produced by `feature_selection.py` (see `preprocessing.md`):

```bash
for method in mutual_information chi_square t_test; do
  for k in top100 top150; do
    python src/ARM/fpgrowth/run_fpgrowth.py \
      --input  "data/final/feature_selection/$method/${k}_genes_discretized.csv" \
      --output-dir "data/final/ARM/fpgrowth_v2/$method/$k"
  done
done
```

---

## Step 2 — `filter_class_rules.py`

Filters mined rules down to single-item class rules with `relapse` as the
consequent, summarizes them by lift, and compares methods.

```bash
python src/ARM/fpgrowth/filter_class_rules.py \
  --method mutual_information chi_square t_test \
  --base-dir data/final/ARM/fpgrowth_v2
```

Arguments and defaults:

| Arg | Default | Notes |
|---|---|---|
| `--method` | all three (`chi_square`, `mutual_information`, `t_test`) | Space-separated list of methods to process |
| `--base-dir` | `data/final/ARM/fpgrowth_v2` | Root containing `<method>/top100/` subdirs |

Behavior:
- Reads `<base-dir>/<method>/top100/association_rules.csv`.
- Parses the `frozenset({...})` strings in `consequents`.
- Identifies class rules: single-item consequents `relapse_yes` / `relapse_no`
  (one-hot convention) or bare `relapse` (old convention); multi-item
  consequents that merely contain relapse are excluded.
- Computes per-class counts at lift thresholds **1.2 / 1.5 / 2.0** and
  antecedent-support stats (min/max/median, count below 0.07 ≈ 20 samples at
  n=286).
- Prints a cross-method comparison and reports which method produced the most
  `relapse_yes` rules at lift > 1.2.

Outputs (in each `<base-dir>/<method>/top100/`):

| File | Content |
|---|---|
| `class_rules_filtered.csv` | Class rules with original rule columns plus a `class` column (`relapse_yes`/`relapse_no`), sorted by `class`, then `lift` descending |
| `class_rule_summary.json` | Per-method summary: `method`, `input_file`, `total_rules`, `n_parse_failures`, `item_naming_detected`, `class_item_counts`, `relapse_multi_item_consequents`, and `class_rules` stats per class |

---

## Step 3 — `filter_class_rules_v2.py`

Same class-rule extraction as Step 2, but evaluates rules with the
base-rate-adjusted **certainty factor (CF)** instead of raw lift. Also
supports a post-hoc `--cf-positive` pass that keeps only rules with CF > 0.

```bash
python src/ARM/fpgrowth/filter_class_rules_v2.py \
  --method mutual_information chi_square t_test \
  --base-dir data/final/ARM/fpgrowth_v2
```

Post-hoc CF > 0 filter (no re-mining):

```bash
python src/ARM/fpgrowth/filter_class_rules_v2.py \
  --cf-positive \
  --base-dir data/final/ARM/fpgrowth_v2
```

---

## Step 4 — `run_class_relative_fpgrowth.py`

Mines frequent itemsets separately within each relapse class subset at a
matched absolute support threshold (default: 20 samples per class), then
compares class-exclusive itemsets.

```bash
python src/ARM/fpgrowth/run_class_relative_fpgrowth.py \
  --method mutual_information chi_square t_test \
  --top-k top100 \
  --base-dir data/final/ARM/fpgrowth_v2
```

`--top-k` choices are `top100` (default) and `top150`.

Outputs (under `<base-dir>/<method>/<top-k>/class_relative/`):

| File | Content |
|---|---|
| `relapse_yes/frequent_itemsets.csv` | Itemsets frequent in relapse=yes subset |
| `relapse_no/frequent_itemsets.csv` | Itemsets frequent in relapse=no subset |
| `relapse_yes_exclusive_itemsets.csv` | Itemsets frequent in yes but not no |
| `relapse_no_exclusive_itemsets.csv` | Itemsets frequent in no but not yes |
| `class_relative_summary.json` | Per-class stats and exclusivity counts |

---

## Step 5 — `permutation_test_class_relative.py`

Permutation null baseline for class-relative exclusivity. Shuffles relapse
labels (preserving class sizes) and re-runs the class-relative FP-Growth
procedure to quantify how many class-exclusive itemsets appear by chance.

```bash
python src/ARM/fpgrowth/permutation_test_class_relative.py \
  --method mutual_information \
  --top-k top100 \
  --base-dir data/final/ARM/fpgrowth_v2
```

---

## Step 6 — `extract_relapse_no_2itemsets.py`

Extracts and characterises the relapse-no-exclusive 2-item itemsets (the
permutation-validated signal) from the class-relative output, recomputing exact
per-class support by direct counting.

```bash
python src/ARM/fpgrowth/extract_relapse_no_2itemsets.py \
  --method mutual_information \
  --top-k top100 \
  --base-dir data/final/ARM/fpgrowth_v2
```

---

## Output Layout

```
data/final/ARM/
├── fpgrowth/                 # earlier runs (if any): bare `relapse` item
│   ├── chi_square/top100, top150
│   ├── mutual_information/top100, top150
│   └── t_test/top100, top150
├── fpgrowth_v2/              # current runs: one-hot `relapse_yes`/`relapse_no`
│   ├── chi_square/top100, top150
│   ├── mutual_information/top100, top150
│   └── t_test/top100, top150
└── fpgrowth_v3/              # CF > 0 post-filter output
    ├── chi_square/top100
    ├── mutual_information/top100
    └── t_test/top100
```

Each run directory contains `frequent_itemsets.csv`,
`association_rules.csv`, `run_metadata.json`; the top100 directories also
contain filtered class-rule outputs after Steps 2–3. Class-relative outputs
land under `{method}/{top-k}/class_relative/`.

## Apriori vs FP-Growth

Both pipelines use the same inputs, target re-encoding, support/confidence
thresholds, and downstream filtering. FP-Growth builds a compact FP-tree and
mines frequent itemsets without candidate generation, which is typically
faster than Apriori on dense binary data. Rule quality metrics (support,
confidence, lift, CF) should match between algorithms when run with identical
parameters — any differences would indicate a numerical or implementation edge
case, not a methodological change.

See `docs/apriori.md` for the parallel Apriori pipeline and observed Apriori
runtimes/rule counts.
