import argparse
import json
import time
from pathlib import Path

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules


# The discretized input files carry the target as a single bare 0/1 column
# (`relapse`). mlxtend's apriori turns that into one item `relapse`, which
# only ever represents relapse = yes and leaves no item for relapse = no.
# Before mining we re-encode it into two mutually exclusive one-hot items
# (`relapse_yes` / `relapse_no`) so both classes are mineable as consequents.
TARGET_COL = "relapse"


def one_hot_encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """Replace the bare binary target column with two one-hot columns.

    Gene features are left untouched (already one-hot discretized); only the
    target is re-encoded so each class becomes its own item.
    """
    if TARGET_COL not in df.columns:
        return df

    encoded = df.copy()
    target = encoded.pop(TARGET_COL).astype(int)
    encoded[f"{TARGET_COL}_yes"] = (target == 1).astype(bool)
    encoded[f"{TARGET_COL}_no"] = (target == 0).astype(bool)
    return encoded


def main():
    parser = argparse.ArgumentParser(
        description="Run Apriori association rule mining."
    )

    parser.add_argument(
        "--input",
        type=str,
        default="data/final/feature_selection/top100_genes_discretized.csv",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/final/ARM/apriori/top100",
    )

    parser.add_argument(
        "--min-support",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--max-len",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.5,
    )

    args = parser.parse_args()

    # Resolve paths relative to project root
    project_root = Path(__file__).resolve().parents[3]

    input_path = project_root / args.input
    output_dir = project_root / args.output_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("APRIORI ASSOCIATION RULE MINING")
    print("=" * 60)

    print(f"Input:        {input_path}")
    print(f"Output:       {output_dir}")
    print(f"Min support:  {args.min_support}")
    print(f"Max itemset:  {args.max_len}")
    print(f"Confidence:   {args.min_confidence}")
    print()

    # ---------------------------------------------------------
    # Load data
    # ---------------------------------------------------------

    print("Loading data...")
    df = pd.read_csv(input_path)

    print(f"Data shape: {df.shape}")
    print()

    # ---------------------------------------------------------
    # Re-encode target into one-hot relapse_yes / relapse_no items
    # ---------------------------------------------------------

    if TARGET_COL in df.columns:
        n_target_yes = int((df[TARGET_COL] == 1).sum())
        n_target_no = int((df[TARGET_COL] == 0).sum())
        df = one_hot_encode_target(df)
        print(f"One-hot encoded target: {TARGET_COL}_yes ({n_target_yes}), "
              f"{TARGET_COL}_no ({n_target_no})")
        print(f"Transaction data shape after encoding: {df.shape}")
    else:
        print(f"Note: no '{TARGET_COL}' column found; no target re-encoding applied.")

    print()

    # ---------------------------------------------------------
    # Run Apriori
    # ---------------------------------------------------------

    print("Running Apriori...")

    start_time = time.time()

    frequent_itemsets = apriori(
        df,
        min_support=args.min_support,
        use_colnames=True,
        max_len=args.max_len,
    )

    apriori_time = time.time() - start_time

    print(f"Frequent itemsets found: {len(frequent_itemsets)}")
    print(f"Apriori runtime: {apriori_time:.3f} seconds")
    print()

    # ---------------------------------------------------------
    # Generate association rules
    # ---------------------------------------------------------

    print("Generating association rules...")

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

    print(f"Association rules found: {len(rules)}")
    print(f"Rule generation runtime: {rule_time:.3f} seconds")
    print()

    # ---------------------------------------------------------
    # Save results
    # ---------------------------------------------------------

    print("Saving results...")

    frequent_itemsets.to_csv(
        output_dir / "frequent_itemsets.csv",
        index=False,
    )

    rules.to_csv(
        output_dir / "association_rules.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # Save metadata
    # ---------------------------------------------------------

    metadata = {
        "input_file": str(input_path),
        "input_shape": list(df.shape),
        "target_encoding": {
            "applied": TARGET_COL in df.columns,
            "items": [f"{TARGET_COL}_yes", f"{TARGET_COL}_no"]
            if TARGET_COL in df.columns
            else None,
        },
        "min_support": args.min_support,
        "max_len": args.max_len,
        "min_confidence": args.min_confidence,
        "n_frequent_itemsets": len(frequent_itemsets),
        "n_association_rules": len(rules),
        "apriori_runtime_seconds": round(apriori_time, 3),
        "rule_generation_runtime_seconds": round(rule_time, 3),
        "total_runtime_seconds": round(
            apriori_time + rule_time,
            3,
        ),
    }

    with open(
        output_dir / "run_metadata.json",
        "w",
    ) as f:
        json.dump(metadata, f, indent=2)

    print()
    print("=" * 60)
    print("COMPLETED")
    print("=" * 60)

    print(f"Frequent itemsets: {len(frequent_itemsets)}")
    print(f"Association rules: {len(rules)}")
    print(f"Total runtime:      {apriori_time + rule_time:.3f}s")

    print()
    print("Output files:")
    print(f"  {output_dir / 'frequent_itemsets.csv'}")
    print(f"  {output_dir / 'association_rules.csv'}")
    print(f"  {output_dir / 'run_metadata.json'}")


if __name__ == "__main__":
    main()
