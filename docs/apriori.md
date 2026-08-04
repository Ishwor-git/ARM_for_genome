# Apriori Association Rule Mining

Steps and commands for mining association rules from the discretized,
feature-selected data using `mlxtend.frequent_patterns` Apriori, then
filtering the results down to class-association rules that predict `relapse`.

Run preprocessing first so the inputs below exist.

## Environment

- Python 3.13
- `pandas`, `mlxtend` (0.25.0), `scipy`/`sklearn` used upstream
- `mlxtend.frequent_patterns.apriori` and `association_rules` — the Apriori
  algorithm is not hand-implemented.

## Files

| Step | Script | Run |
|---|---|---|
| 1. Mine frequent itemsets + rules | `src/ARM/apriori/run_apriori.py` | see below |
| 2. Filter to class-association rules | `src/ARM/apriori/filter_class_rules.py` | see below |

---

## Step 1 — `run_apriori.py`

Mines frequent itemsets and generates association rules from a discretized
input file, and writes results + run metadata.

```bash
python src/ARM/apriori/run_apriori.py \
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
| `--output-dir` | `data/final/ARM/apriori/top100` | Created if missing |
| `--min-support` | `0.1` | Apriori minimum support |
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
- Runs `apriori(min_support, use_colnames=True, max_len)`.
- Generates rules via `association_rules(itemsets, metric="confidence",
  min_threshold=min_confidence)`.
- Writes outputs and a metadata JSON.

Outputs (in `--output-dir`):

| File | Content |
|---|---|
| `frequent_itemsets.csv` | Columns: `itemsets` (frozenset of item names), `support` |
| `association_rules.csv` | Columns: `antecedents`, `consequents`, `antecedent support`, `consequent support`, `support`, `confidence`, `lift`, `representativity`, `leverage`, `conviction`, `zhangs_metric`, `jaccard`, `certainty`, `kulczynski` |
| `run_metadata.json` | `input_file`, `input_shape`, target-encoding info, `min_support`, `max_len`, `min_confidence`, `n_frequent_itemsets`, `n_association_rules`, Apriori / rule-gen / total runtime in seconds |

### Reproduce all runs

Run for each selection method × top-k input. The top100/top150 discretized
inputs are produced by `feature_selection.py` (see `preprocessing.md`):

```bash
for method in mutual_information chi_square t_test; do
  for k in top100 top150; do
    python src/ARM/apriori/run_apriori.py \
      --input  "data/final/feature_selection/$method/${k}_genes_discretized.csv" \
      --output-dir "data/final/ARM/apriori_v2/$method/$k"
  done
done
```

---

## Step 2 — `filter_class_rules.py`

Filters mined rules down to single-item class rules with `relapse` as the
consequent, summarizes them by lift, and compares methods.

```bash
python src/ARM/apriori/filter_class_rules.py \
  --method mutual_information chi_square t_test \
  --base-dir data/final/ARM/apriori_v2
```

Arguments and defaults:

| Arg | Default | Notes |
|---|---|---|
| `--method` | all three (`chi_square`, `mutual_information`, `t_test`) | Space-separated list of methods to process |
| `--base-dir` | `data/final/ARM/apriori` | Root containing `<method>/top100/` subdirs |

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

## Output Layout

```
data/final/ARM/
├── apriori/                  # earlier runs: target NOT re-encoded (bare `relapse` item)
│   ├── chi_square/top100, top150
│   ├── mutual_information/top100, top150
│   └── t_test/top100, top150
└── apriori_v2/               # current runs: one-hot `relapse_yes`/`relapse_no` re-encoding
    ├── chi_square/top100
    ├── mutual_information/top100
    └── t_test/top100
```

Each run directory contains `frequent_itemsets.csv`,
`association_rules.csv`, `run_metadata.json`; the top100 directories also
contain `class_rules_filtered.csv` and `class_rule_summary.json`.

The two top-level variants differ only in target encoding: `apriori/` runs
were generated before the one-hot re-encoding was added to `run_apriori.py`
(input shape 101 for top100 = 100 genes + bare `relapse`), while `apriori_v2/`
uses the current script (input shape 102 = 100 genes + `relapse_yes` +
`relapse_no`). `filter_class_rules.py` handles both item conventions.

## Observed Results (run metadata, min-support 0.1, max-len 3, min-conf 0.5)

| Run | Input shape | Itemsets | Rules | Total runtime (s) |
|---|---|---|---|---|
| apriori/chi_square/top100 | 286 × 101 | 133,889 | 285,359 | 3.3 |
| apriori/chi_square/top150 | 286 × 151 | 454,596 | 961,672 | 12.5 |
| apriori/mutual_information/top100 | 286 × 101 | 154,790 | 316,546 | 3.5 |
| apriori/mutual_information/top150 | 286 × 151 | 518,407 | 1,039,766 | 13.9 |
| apriori/t_test/top100 | 286 × 101 | 134,186 | 276,041 | 3.1 |
| apriori/t_test/top150 | 286 × 151 | 453,256 | 913,114 | 13.1 |
| apriori_v2/chi_square/top100 | 286 × 102 | 138,735 | 296,074 | 11.6 |
| apriori_v2/mutual_information/top100 | 286 × 102 | 159,820 | 328,104 | 11.7 |
| apriori_v2/t_test/top100 | 286 × 102 | 139,101 | 284,389 | 11.7 |

Class-rule counts (lift thresholds) from `class_rule_summary.json`:

| Run (top100) | Naming | relapse_yes total | lift>1.2 / >1.5 / >2.0 | relapse_no total | lift>1.2 / >1.5 / >2.0 |
|---|---|---|---|---|---|
| apriori/chi_square | bare_binary | 982 | 982 / 684 / 0 | 0 | – |
| apriori/mutual_information | bare_binary | 107 | 107 / 14 / 0 | 0 | – |
| apriori/t_test | bare_binary | 1,601 | 1,601 / 417 / 0 | 0 | – |
| apriori_v2/chi_square | one_hot | 982 | 982 / 684 / 0 | 4,040 | 1,415 / 0 / 0 |
| apriori_v2/mutual_information | one_hot | 107 | 107 / 14 / 0 | 4,954 | 428 / 0 / 0 |
| apriori_v2/t_test | one_hot | 1,601 | 1,601 / 417 / 0 | 3,470 | 592 / 0 / 0 |

(The `relapse_no` rules are largely high-confidence rules against a majority
class; `relapse_yes` at lift > 1.5 is the more informative subset. No rule in
either class reaches lift > 2.0 in these runs.)

**Note : Running apriori to features > 150 is not computationally possible at this point**
