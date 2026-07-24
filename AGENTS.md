# AGENTS.md — Association Rule Mining on High-Dimensional Cancer Data

This file gives coding agents (and future-you) the context needed to work on this
project without re-deriving decisions that have already been made. Read this
before writing or modifying any code in this repo.

## 1. Project Summary

**Course:** Data Mining, 1-month project
**Topic:** Association Rule Mining (ARM) on very high-dimensional data (features >> samples)
**Dataset:** Breast cancer gene expression microarray data (Affymetrix HG-U133A platform)

**Scoped objectives (do not expand scope beyond this without explicit sign-off):**
1. Run classical ARM (Apriori and/or FP-Growth) on a filtered feature set; document
   how/why it fails or becomes intractable on the full raw feature count.
2. Apply feature-reduction techniques to make ARM tractable, and compare 2–3
   techniques against each other (not just before/after one fix).
3. Push the feasible feature count as far as possible under each technique and
   report where it breaks again.
4. Benchmark rule-based results against one ML baseline classifier trained on
   the same reduced feature set, using the shared class label.

Timeline is ~2–4 weeks. Prioritize a working, well-analyzed pipeline over
maximal scope. If a technique isn't giving clean results by a set checkpoint,
cut back rather than pushing all 3 techniques through under time pressure.

## 2. Dataset Shape and Structure

- **Raw shape:** 286 samples × 22,289 columns.
- **p/n ratio:** ~78:1 — extreme even for gene-expression ARM standards. Any
  algorithm/step must be validated at small scale before running on the full set.
- **Feature type:** continuous expression intensities, **already median-split
  by the user into high/low per-feature** in earlier work. See Section 4 for
  why this needs to be redone more carefully before proceeding.

### Column groups — DO NOT treat these as interchangeable

| Group | Columns | Treatment |
|---|---|---|
| **Gene expression probes** | ~22,215 columns matching Affymetrix probe ID patterns (`\d+_at`, `\d+_s_at`, `\d+_x_at`, `\d+_i_at`, `\d+_r_at`, `\d+_g_at`, `\d+_f_at`) | Candidate ARM items. Subject to feature selection + discretization. |
| **Affymetrix control probes** | All columns starting with `AFFX-` (~60+ columns: `AFFX-BioB-*`, `AFFX-BioC-*`, `AFFX-CreX-*`, `AFFX-DapX-*`, `AFFX-HUMGAPDH*`, `AFFX-HUMISGF3A*`, `AFFX-HUMRGE*`, `AFFX-LysX-*`, `AFFX-PheX-*`, `AFFX-ThrX-*`, `AFFX-TrpnX-*`, `AFFX-r2-*`, `AFFX-M27830*`) | **Exclude from all analysis.** These are spike-in/hybridization QC controls (bacterial genes at known concentrations, housekeeping controls), not tumor biology. Including them in ARM produces meaningless rules. |
| **Clinical covariates** | `lymph_node_status`, `ER_Status` | Already categorical/binary. **Do NOT median-split these.** Can be included directly as items if desired, with justification. |
| **Outcome / leakage-risk columns** | `relapse`, `time_to_relapse`, `Brain_relapses`, `GEO_accession_number` | See Section 3. |

## 3. Target Variable and Leakage Rules

- **`relapse`** is the intended class label. Use it for:
  - Class Association Rule mining (CARs): rules of the form
    `{gene_X=high, gene_Y=low, ...} → relapse=yes/no`.
  - The ML baseline benchmark target (train a classifier on the same reduced
    gene set, predict `relapse`, compare accuracy/interpretability to CAR rules).
- **`time_to_relapse`** and **`Brain_relapses`** are outcome-adjacent and create
  **leakage risk** if included as predictor items alongside `relapse` — a rule
  using `time_to_relapse` to "predict" `relapse` is close to circular. Exclude
  these from the item/feature set used for rule mining and classification unless
  there's a specific, justified reason to include one (state it explicitly in
  the report if so).
- **`GEO_accession_number`** is a sample identifier. It must never be treated as
  a feature value (do not discretize, do not include as an item). If it currently
  appears as a column in any working dataframe used for modeling, that's a bug —
  it should live only in an index/ID column.

## 4. Known Issue: the Existing Median-Split

The user already applied a per-feature median split (high/low) across the raw
matrix in earlier work. Before continuing, confirm and fix:

1. **Was the split applied only to the ~22,215 expression probes, or across the
   whole raw table including `AFFX-*`, clinical, and outcome columns?** If the
   latter, this is a bug — re-run discretization scoped only to expression probes.
2. **Per-feature median split forces exactly 50% support for every single
   feature's high/low bins**, by construction. This:
   - Destroys information about how differentially expressed a gene actually is
     (a strongly bimodal, biologically meaningful gene and a pure-noise gene
     both get forced into a 50/50 split).
   - Is unsupervised — it ignores `relapse` entirely, so it can't distinguish
     a meaningful split from an arbitrary one.
   - Inflates the risk of spurious high-support, high-confidence itemsets
     purely by chance, given only 286 samples.
3. This is an acceptable **first-pass baseline** but must be explicitly flagged
   as a limitation in the report, not presented as the final method. If time
   allows, compare against one alternative (e.g., class-conditional cutpoint,
   or discretizing only after feature selection).

## 5. Required Pipeline Order

Do not run ARM directly on the full raw feature set — it is very likely
computationally infeasible (even a first pass of frequent 2-itemset generation
across ~22,215 features means checking ~247M pairs before pruning). The pipeline
must go in this order:

1. **Clean columns**: split into the three groups in Section 2's table; drop
   `AFFX-*` controls; separate outcome columns from predictor columns.
2. **Feature selection** (do this BEFORE re-running ARM at scale): reduce from
   ~22,215 genes down to a manageable count (start around 100–500) using a
   class-aware filter — mutual information with `relapse`, chi-square, or
   t-test/fold-change ranking. Do not use unsupervised variance filtering alone;
   given `relapse` is available, use it.
3. **Discretize** the selected genes (median split as baseline; note the
   limitation from Section 4).
4. **Run ARM** (Apriori/FP-Growth via `mlxtend` or similar — do not hand-roll
   these algorithms) on the discretized, filtered set. Prefer class association
   rule mining (rules with `relapse` as consequent) over generic ARM.
5. **Evaluate rules** beyond support/confidence — report lift at minimum, and
   ideally a stability check (rerun on a bootstrap resample, see if top rules
   survive) given the false-discovery risk at n=286.
6. **Push feature count up incrementally** (e.g., 100 → 500 → 1000+) to find
   where ARM becomes intractable again — this is the answer to "how many
   features can be incorporated."
7. **Train one ML baseline** (e.g., decision tree or logistic regression) on
   the same reduced/selected feature set, predicting `relapse`. Compare rule-based
   prediction accuracy vs. classifier accuracy, and discuss rule interpretability
   vs. classifier interpretability.

## 6. Tooling Conventions

- Use existing libraries for ARM (`mlxtend.frequent_patterns` for Apriori/FP-Growth
  in Python, or `arules` in R) — do not implement Apriori/FP-Growth from scratch.
  Time budget goes to analysis, not algorithm reimplementation.
- Use standard libraries for feature selection and classification
  (`scikit-learn`: `SelectKBest`, `mutual_info_classif`, `chi2`, `DecisionTreeClassifier`,
  `LogisticRegression`).
- Keep a fixed random seed for any resampling/bootstrap steps so results are
  reproducible across runs.
- Any script that loads the raw data should assert/check for the presence of
  `AFFX-*` and the known outcome/ID columns and explicitly drop or route them —
  don't rely on manual column slicing by position, which breaks silently.

## 7. Reporting Requirements

The final report/write-up must explicitly state, not just imply:
- Which columns were excluded and why (controls, leakage risk).
- That the median-split discretization is a known limitation (Section 4).
- The success metric used for "acceptable results" (rule count, lift, stability
  under resampling — pick and state it up front, don't leave it implicit).
- The comparison axis used for the ML benchmark (accuracy vs. interpretability
  vs. both) — state this explicitly rather than leaving the comparison vague.

## 8. Out of Scope (unless explicitly requested later)

- Modifying ARM algorithm internals (e.g., writing custom pruning strategies) —
  treat "increasing features incorporated" as a feature-selection problem, not
  an algorithm-engineering problem, given the timeline.
- Full survival analysis using `time_to_relapse` — flagged as a possible
  future extension, not part of the current scope.
- Building a full ML pipeline with multiple models/hyperparameter tuning — one
  baseline classifier is sufficient for the benchmark objective.
