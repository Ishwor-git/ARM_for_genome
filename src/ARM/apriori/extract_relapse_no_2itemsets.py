import argparse
import json
from pathlib import Path

import pandas as pd

from run_class_relative_apriori import (
    CLASS_YES,
    CLASS_NO,
    MIN_ABS_SUPPORT,
    absolute_count,
    find_exclusive,
    parse_itemset,
    split_by_class,
)

METHOD = "mutual_information"
TOP_DIR = "top100"
RELATIVE_DIR_NAME = "class_relative"

EXPECTED_SIZE2_EXCLUSIVE = 745

OUTPUT_CSV = "relapse_no_exclusive_2itemsets.csv"
OUTPUT_SUMMARY = "relapse_no_2itemset_summary.json"

HIGH_STATE = "high"


def _fsrows(fi, n):
    """Recover itemsets as frozensets with exact absolute counts, identical
    to the permutation script's bookkeeping."""
    im = {}
    for _, row in fi.iterrows():
        fs = parse_itemset(row["itemsets"])
        if fs is not None:
            im[fs] = {"count": round(row["support"] * n)}
    return im


def main():
    parser = argparse.ArgumentParser(
        description="Extract and characterise the relapse_no-exclusive 2-item "
        "itemsets (the permutation-validated signal) from the class-relative "
        "Apriori output, reusing the identical exclusivity logic."
    )
    parser.add_argument(
        "--fs-dir", type=str, default="data/final/feature_selection",
        help="Dir containing mutual_information/top100_genes_discretized.csv.",
    )
    parser.add_argument(
        "--base-dir", type=str, default="data/final/ARM/apriori_v2",
        help="Root under which outputs/method/top100/class_relative/ is written.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[3]
    input_path = (
        project_root / args.fs_dir / METHOD / f"{TOP_DIR}_genes_discretized.csv"
    )
    rel_root = project_root / args.base_dir / METHOD / TOP_DIR / RELATIVE_DIR_NAME

    no_fi_path = rel_root / CLASS_NO / "frequent_itemsets.csv"
    yes_fi_path = rel_root / CLASS_YES / "frequent_itemsets.csv"

    df = pd.read_csv(input_path)
    yes_df, no_df = split_by_class(df)
    n_yes, n_no = len(yes_df), len(no_df)

    no_fi = pd.read_csv(no_fi_path)
    yes_fi = pd.read_csv(yes_fi_path)

    no_counts = _fsrows(no_fi, n_no)
    yes_counts = _fsrows(yes_fi, n_yes)

    exclusive = find_exclusive(no_counts, yes_counts)
    size2 = [(fs, here, other) for fs, here, other in exclusive if len(fs) == 2]

    if len(size2) != EXPECTED_SIZE2_EXCLUSIVE:
        raise RuntimeError(
            f"Expected {EXPECTED_SIZE2_EXCLUSIVE} relapse_no-exclusive 2-item "
            f"itemsets, got {len(size2)}. Do not proceed with mismatched count."
        )

    records = []
    for fs, count_no, count_other in size2:
        genes = sorted(fs)
        gene1, gene2 = genes
        count_yes = absolute_count(yes_df, fs)
        records.append(
            {
                "gene1": gene1,
                "gene1_state": HIGH_STATE,
                "gene2": gene2,
                "gene2_state": HIGH_STATE,
                "count_relapse_no": count_no,
                "support_relapse_no": round(count_no / n_no, 6),
                "count_relapse_yes": count_yes,
                "support_relapse_yes": round(count_yes / n_yes, 6),
            }
        )

    out_df = pd.DataFrame(
        records,
        columns=[
            "gene1",
            "gene1_state",
            "gene2",
            "gene2_state",
            "count_relapse_no",
            "support_relapse_no",
            "count_relapse_yes",
            "support_relapse_yes",
        ],
    )
    out_df = out_df.sort_values(
        ["count_relapse_no", "gene1", "gene2"],
        ascending=[False, True, True],
    ).reset_index(drop=True)

    out_csv = rel_root / OUTPUT_CSV
    out_df.to_csv(out_csv, index=False)

    gene_counts = {}
    for _, row in out_df.iterrows():
        gene_counts[row["gene1"]] = gene_counts.get(row["gene1"], 0) + 1
        gene_counts[row["gene2"]] = gene_counts.get(row["gene2"], 0) + 1

    top_genes = sorted(gene_counts.items(), key=lambda kv: (-kv[1], kv[0]))

    n_pairs = len(out_df)
    summary = {
        "method": METHOD,
        "top_dir": TOP_DIR,
        "min_absolute_support": MIN_ABS_SUPPORT,
        "class_sizes": {CLASS_YES: n_yes, CLASS_NO: n_no},
        "input": {
            "raw": str(input_path),
            "relapse_no_frequent_itemsets": str(no_fi_path),
            "relapse_yes_frequent_itemsets": str(yes_fi_path),
        },
        "n_exclusive_2itemsets": n_pairs,
        "expected_2itemset_count": EXPECTED_SIZE2_EXCLUSIVE,
        "count_matches_permutation": n_pairs == EXPECTED_SIZE2_EXCLUSIVE,
        "unique_genes_across_pairs": len(gene_counts),
        "genes": {gene: count for gene, count in top_genes},
        "top_genes": [
            {
                "gene": gene,
                "pair_count": count,
                "share_of_pairs": round(count / n_pairs, 6),
            }
            for gene, count in top_genes[:15]
        ],
        "gene_pair_count_distribution": {
            str(k): int(v)
            for k, v in sorted(
                {c: list(gene_counts.values()).count(c) for c in set(gene_counts.values())}.items(),
                reverse=True,
            )
        },
        "interpretation": (
            "Each item in an itemset is a gene probe that is discretized to its "
            "high state (above the per-gene median) in the current binary "
            "encoding; the low state is represented by absence and cannot appear "
            "as an item. support_relapse_no/support_relapse_yes are the exact "
            "absolute counts divided by that class's sample size."
        ),
    }

    out_json = rel_root / OUTPUT_SUMMARY
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Relapse_no-exclusive 2-item itemsets: {n_pairs}")
    print(f"Unique genes across pairs: {len(gene_counts)}")
    print("\nTop genes by pair count:")
    print(f"{'gene':<16}{'pairs':>8}{'share':>10}")
    for gene, count in top_genes[:15]:
        print(f"{gene:<16}{count:>8}{count / n_pairs:>10.4f}")
    print(f"\nWrote: {out_csv}")
    print(f"Wrote: {out_json}")


if __name__ == "__main__":
    main()
