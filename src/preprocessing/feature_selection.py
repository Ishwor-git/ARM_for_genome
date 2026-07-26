import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

SEED = 42
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "final"
OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "final"

LEAKAGE_COLS = {"time_to_relapse", "Brain_relapses", "GEO_accession_number", "GEO_asscession_number"}
CLINICAL_COLS = {"lymph_node_status", "ER_Status"}
TARGET_COL = "relapse"

df = pd.read_csv(DATA_DIR / "cleaned_data.csv")
print(f"Loaded cleaned_data.csv: {df.shape}")

known_typos = {"GEO_asscession_number": "GEO_accession_number"}
df = df.rename(columns=known_typos)

affx_pattern = re.compile(r"^AFFX-", re.IGNORECASE)
affx_cols = [c for c in df.columns if affx_pattern.match(str(c))]
if affx_cols:
    df = df.drop(columns=affx_cols)
    print(f"Dropped {len(affx_cols)} AFFX-* control columns")

leakage_present = LEAKAGE_COLS & set(df.columns)
if leakage_present:
    df = df.drop(columns=list(leakage_present))
    print(f"Dropped leakage columns: {leakage_present}")

if "!series_matrix_table_end" in df.columns:
    df = df.drop(columns=["!series_matrix_table_end"])
    print("Dropped !series_matrix_table_end column")

print(f"Shape after cleaning: {df.shape}")

expression_cols = [c for c in df.columns if c not in CLINICAL_COLS and c != TARGET_COL]
clinical_cols = [c for c in df.columns if c in CLINICAL_COLS]

print(f"Expression probes: {len(expression_cols)}")
print(f"Clinical covariates: {clinical_cols}")
print(f"Target column: {TARGET_COL}")

expr_df = df[expression_cols].copy()
clinical_df = df[clinical_cols].copy() if clinical_cols else None
y = df[TARGET_COL].values

print(f"\nExpression data shape: {expr_df.shape}")
print(f"Target distribution:\n{df[TARGET_COL].value_counts()}")

# Per-feature median split on expression probes only
print("\nApplying per-feature median split (high/low) to expression probes...")
expr_discrete = expr_df.copy()
for col in expr_discrete.columns:
    median_val = expr_discrete[col].median()
    expr_discrete[col] = (expr_discrete[col] > median_val).astype(int)

print(f"Discretized expression shape: {expr_discrete.shape}")
print(f"Unique values in discretized data: {np.unique(expr_discrete.values)}")

print("\nComputing mutual information with relapse label...")
mi_scores = mutual_info_classif(expr_discrete, y, random_state=SEED)
mi_df = pd.DataFrame({"gene_id": expression_cols, "mi_score": mi_scores})
mi_df = mi_df.sort_values("mi_score", ascending=False).reset_index(drop=True)

mi_path = OUTPUT_DIR / "gene_mi_rankings.csv"
mi_df.to_csv(mi_path, index=False)
print(f"\nSaved MI rankings ({len(mi_df)} genes) to {mi_path}")
print(f"Top 10 genes by MI:")
print(mi_df.head(10))
print(f"\nMI score range: [{mi_df['mi_score'].min():.6f}, {mi_df['mi_score'].max():.6f}]")

for k in [100, 500, 1000]:
    top_genes = mi_df.head(k)["gene_id"].tolist()
    subset_df = expr_discrete[top_genes].copy()
    subset_df[TARGET_COL] = y
    out_path = OUTPUT_DIR / f"top{k}_genes_discretized.csv"
    subset_df.to_csv(out_path, index=False)
    print(f"  top{k}_genes_discretized.csv: {subset_df.shape}")
