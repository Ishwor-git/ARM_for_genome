import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from sklearn.feature_selection import chi2, mutual_info_classif

from clean_features import CLINICAL_COLS, TARGET_COL

SEED = 42
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "final"
OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "final" / "feature_selection"

METHOD_CHOICES = ["mutual_information", "chi_square", "t_test"]

TOPK_VALUES = list(range(100, 401, 50))


def load_clean_data() -> pd.DataFrame:
    """Load the cleaned_data.csv already produced by clean_features.py.

    Run clean_features.py first if this file doesn't exist yet.
    """
    cleaned_path = DATA_DIR / "cleaned_data.csv"
    if not cleaned_path.exists():
        raise FileNotFoundError(
            f"{cleaned_path} not found. Run clean_features.py first to generate it."
        )

    df = pd.read_csv(cleaned_path)
    print(f"Loaded cleaned_data.csv: {df.shape}")

    # clean_features.py doesn't know about this series-matrix artifact;
    # kept here since it's specific to the feature-selection step's inputs.
    if "!series_matrix_table_end" in df.columns:
        df = df.drop(columns=["!series_matrix_table_end"])
        print("Dropped !series_matrix_table_end column")
        print(f"Shape after dropping series-matrix column: {df.shape}")

    return df


def compute_scores(method: str, expr_df: pd.DataFrame, expr_discrete: pd.DataFrame, y: np.ndarray):
    """Compute per-gene selection scores for the requested method.

    Parameters
    ----------
    method : str
        One of ``mutual_information``, ``chi_square``, ``t_test``.
    expr_df : pd.DataFrame
        Continuous expression values (samples as rows, probes as columns).
    expr_discrete : pd.DataFrame
        Median-split (0/1) version of ``expr_df``.
    y : np.ndarray
        Binary target labels.

    Returns
    -------
    scores : np.ndarray
        Per-gene scores (higher = more relevant).
    score_col : str
        Name to use for the score column in the rankings output.
    """
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Class-aware gene feature selection using mutual information, "
        "chi-square, or t-test."
    )
    parser.add_argument(
        "--method",
        choices=METHOD_CHOICES,
        default="mutual_information",
        help="Feature selection method (default: mutual_information).",
    )
    args = parser.parse_args()
    method = args.method

    df = load_clean_data()

    expression_cols = [c for c in df.columns if c not in CLINICAL_COLS and c != TARGET_COL]
    clinical_cols = [c for c in df.columns if c in CLINICAL_COLS]

    print(f"Expression probes: {len(expression_cols)}")
    print(f"Clinical covariates: {clinical_cols}")
    print(f"Target column: {TARGET_COL}")

    expr_df = df[expression_cols].copy()
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

    print(f"\nComputing feature scores with {method} against {TARGET_COL} label...")
    scores, score_col = compute_scores(method, expr_df, expr_discrete, y)
    score_df = pd.DataFrame({"gene_id": expression_cols, score_col: scores})
    score_df = score_df.sort_values(score_col, ascending=False).reset_index(drop=True)

    method_dir = OUTPUT_DIR / method
    method_dir.mkdir(parents=True, exist_ok=True)

    ranking_path = method_dir / f"gene_{method}_rankings.csv"
    score_df.to_csv(ranking_path, index=False)
    print(f"\nSaved {method} rankings ({len(score_df)} genes) to {ranking_path}")
    print("Top 10 genes:")
    print(score_df.head(10))
    print(f"\nScore range: [{score_df[score_col].min():.6f}, {score_df[score_col].max():.6f}]")

    print(f"\nSelecting top-k genes (k = {TOPK_VALUES})...")
    for k in TOPK_VALUES:
        top_genes = score_df.head(k)["gene_id"].tolist()
        subset_df = expr_discrete[top_genes].copy()
        subset_df[TARGET_COL] = y
        out_path = method_dir / f"top{k}_genes_discretized.csv"
        subset_df.to_csv(out_path, index=False)
        print(f"  top{k}_genes_discretized.csv: {subset_df.shape}")


if __name__ == "__main__":
    main()
