import argparse
import json
import re
import time
from pathlib import Path

import pandas as pd
from mlxtend.frequent_patterns import apriori

METHODS = ["chi_square", "mutual_information", "t_test"]
TOP_DIR = "top100"

# Absolute sample-count floor below which an itemset is treated as not
# frequent. Both class subsets are mined with this same absolute threshold,
# converted to a per-subset fraction (min_support = MIN_ABS / n_class), so the
# two classes are compared on equal footing rather than against the full-data
# base rate. Consistent with the stability floor used earlier in the project.
MIN_ABS_SUPPORT = 20

MAX_LEN = 3

TARGET_COL = "relapse"
CLASS_YES = "relapse_yes"
CLASS_NO = "relapse_no"

# Existing class-relative output root (created inside the normal apriori_v2
# tree, under {method}/{TOP_DIR}/class_relative/).
RELATIVE_DIR_NAME = "class_relative"

SUMMARY_NAME = "class_relative_summary.json"


_FROZENSET_RE = re.compile(r"frozenset\(\{(.*)\}\)", re.DOTALL)
_ITEM_RE = re.compile(r"'([^']*)'")


def parse_itemset(value):
    m = _FROZENSET_RE.fullmatch(str(value))
    if not m:
        return None
    inner = m.group(1).strip()
    if not inner:
        return frozenset()
    return frozenset(_ITEM_RE.findall(inner))


def fmt_itemset(itemset):
    return ", ".join(sorted(itemset))


def absolute_count(df, itemset):
    """Exact absolute sample count of an itemset (all items present)."""
    if not itemset:
        return int(len(df))
    cols = list(itemset)
    mask = df[cols].astype(bool).all(axis=1)
    return int(mask.sum())


def split_by_class(df):
    """Split transactions into relapse_yes / relapse_no gene-only subsets.

    Recognises either the one-hot convention (relapse_yes / relapse_no
    columns) or a bare binary `relapse` column, routes the class columns out of
    the feature set, and returns (yes_df, no_df) of *gene items only* (the
    class label is implicit in which subset you are in, so it is removed).
    """
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
        raise ValueError(
            f"No class column ({CLASS_YES}/{CLASS_NO} or {TARGET_COL}) found "
            f"in input columns."
        )

    gene_cols = [c for c in cols if c not in class_cols]
    return df.loc[yes_mask, gene_cols], df.loc[no_mask, gene_cols]


def mine_subset(subset_df, min_support, max_len):
    if subset_df.empty:
        return pd.DataFrame(columns=["support", "itemsets"])
    return apriori(
        subset_df,
        min_support=min_support,
        use_colnames=True,
        max_len=max_len,
        low_memory=True,
    )


def find_exclusive(needle, haystack_counts):
    """Itemsets frequent in `needle` but below the abs threshold in the other.

    needle: dict {itemset: {"count": int}} from the class we are mining.
    haystack_counts: dict {itemset: {"count": int}} for the other class.
    Returns list of (itemset, count_here, count_other) for itemsets whose count
    in the other class is strictly below MIN_ABS_SUPPORT.
    """
    exclusive = []
    for itemset, info in needle.items():
        count_here = info["count"]
        count_other = haystack_counts.get(itemset, {}).get("count", 0)
        if count_other < MIN_ABS_SUPPORT:
            exclusive.append((itemset, count_here, count_other))
    return exclusive


def run_method(method, fs_dir, out_base, min_abs, max_len):
    fs_dir = Path(fs_dir)
    input_path = fs_dir / method / f"{TOP_DIR}_genes_discretized.csv"

    print("\n" + "=" * 60)
    print(f"METHOD: {method}")
    print("=" * 60)
    print(f"Input: {input_path}")

    df = pd.read_csv(input_path)
    yes_df, no_df = split_by_class(df)
    n_class = {CLASS_YES: len(yes_df), CLASS_NO: len(no_df)}

    print(f"Full data: {df.shape[0]} samples")
    print(f"  {CLASS_YES}: {n_class[CLASS_YES]} samples")
    print(f"  {CLASS_NO}: {n_class[CLASS_NO]} samples")

    out_root = out_base / method / TOP_DIR / RELATIVE_DIR_NAME
    out_root.mkdir(parents=True, exist_ok=True)

    # --- mine each class subset independently -----------------------------
    mined = {}
    stats = {}
    for class_name, subset_df in ((CLASS_YES, yes_df), (CLASS_NO, no_df)):
        n = n_class[class_name]
        min_support = min_abs / n
        start = time.time()
        fi = mine_subset(subset_df, min_support, max_len)
        elapsed = time.time() - start

        itemsets = {}
        for _, row in fi.iterrows():
            itemset = parse_itemset(row["itemsets"])
            if itemset is None:
                continue
            # support fraction is count/n_subset; recover exact count by
            # rounding (min_support is exactly k/n so counts are integral).
            count = round(row["support"] * n)
            itemsets[itemset] = {
                "count": count,
                "support": float(row["support"]),
            }

        class_out = out_root / class_name
        class_out.mkdir(parents=True, exist_ok=True)
        fi.to_csv(class_out / "frequent_itemsets.csv", index=False)

        # sanity: verify the mined min support really maps back to >= min_abs
        stats[class_name] = {
            "n_samples": int(n),
            "min_support_frac": min_abs / n,
            "min_support_used": min_support,
            "min_abs_support": min_abs,
            "n_frequent_itemsets": len(fi),
            "min_count_seen": min(
                (v["count"] for v in itemsets.values()), default=None
            ),
            "itemsets": itemsets,
            "runtime_seconds": round(elapsed, 3),
        }
        print(
            f"  {class_name} (n={n}, min_support={min_support:.4f}): "
            f"{len(itemsets)} frequent itemsets in {elapsed:.2f}s"
        )

    # --- exclusivity comparison at matched absolute support ---------------
    yes_abs = stats[CLASS_YES]["itemsets"]
    no_abs = stats[CLASS_NO]["itemsets"]
    yes_excl = find_exclusive(yes_abs, no_abs)
    no_excl = find_exclusive(no_abs, yes_abs)
    shared = set(yes_abs) & set(no_abs)

    print(f"\n  {CLASS_YES}-exclusive itemsets: {len(yes_excl)}")
    print(f"  {CLASS_NO}-exclusive itemsets:  {len(no_excl)}")
    print(f"  shared itemsets:                {len(shared)}")

    # --- save exclusive itemset detail for manual inspection --------------
    def dump_exclusive(fname, rows, class_name, other_counts):
        records = []
        for itemset, count_here, count_other in rows:
            records.append(
                {
                    "itemset": fmt_itemset(itemset),
                    f"count_{class_name}": count_here,
                    f"count_{other_counts}": count_other,
                    "size": len(itemset),
                }
            )
        pd.DataFrame(records).to_csv(out_root / fname, index=False)

    dump_exclusive(
        f"{CLASS_YES}_exclusive_itemsets.csv",
        yes_excl,
        CLASS_YES,
        CLASS_NO,
    )
    dump_exclusive(
        f"{CLASS_NO}_exclusive_itemsets.csv",
        no_excl,
        CLASS_NO,
        CLASS_YES,
    )

    # --- summary JSON ------------------------------------------------------
    summary = {
        "method": method,
        "input_file": str(input_path),
        "min_absolute_support_used": min_abs,
        "min_abs_support_reference": MIN_ABS_SUPPORT,
        "max_len": max_len,
        "class_subsets": {
            class_name: {
                "n_samples": stats[class_name]["n_samples"],
                "min_support_frac": round(stats[class_name]["min_support_used"], 6),
                "min_abs_support": stats[class_name]["min_abs_support"],
                "n_frequent_itemsets": stats[class_name]["n_frequent_itemsets"],
                "min_count_seen": stats[class_name]["min_count_seen"],
                "runtime_seconds": stats[class_name]["runtime_seconds"],
            }
            for class_name in (CLASS_YES, CLASS_NO)
        },
        "exclusive": {
            f"count_exclusive_{CLASS_YES}": len(yes_excl),
            f"count_exclusive_{CLASS_NO}": len(no_excl),
            "count_shared": len(shared),
        },
        "sanity_min_abs_support_ge_20": {
            CLASS_YES: min_abs >= 20
            and stats[CLASS_YES]["min_count_seen"] is not None
            and stats[CLASS_YES]["min_count_seen"] >= min_abs,
            CLASS_NO: min_abs >= 20
            and stats[CLASS_NO]["min_count_seen"] is not None
            and stats[CLASS_NO]["min_count_seen"] >= min_abs,
        },
    }

    with open(out_root / SUMMARY_NAME, "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Class-relative association rule mining. Runs Apriori "
        "separately on each relapse class subset at a support threshold "
        "relative to that class's own sample count (matched absolute support "
        "across classes), producing class-characteristic gene itemsets, and "
        "compares class-exclusive itemsets across the three feature-selection "
        "methods."
    )
    parser.add_argument(
        "--method",
        type=str,
        nargs="+",
        choices=METHODS,
        default=METHODS,
        help="Feature selection methods to process (default: all).",
    )
    parser.add_argument(
        "--fs-dir",
        type=str,
        default="data/final/feature_selection",
        help="Root dir containing {method}/top100_genes_discretized.csv.",
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default="data/final/ARM/apriori_v2",
        help="Root dir under which {method}/top100/class_relative outputs are written.",
    )
    parser.add_argument(
        "--min-absolute-support",
        type=int,
        default=MIN_ABS_SUPPORT,
        help="Absolute sample-count threshold applied to each class subset "
        "(min_support = value / n_class).",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=MAX_LEN,
        help="Maximum itemset length.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[3]
    fs_dir = project_root / args.fs_dir
    base_dir = project_root / args.base_dir

    summaries = {}
    for method in args.method:
        summaries[method] = run_method(
            method,
            fs_dir,
            base_dir,
            args.min_absolute_support,
            args.max_len,
        )

    print("\n" + "=" * 60)
    print(
        "CROSS-METHOD: class-exclusive itemsets at matched absolute support "
        f"(min abs={args.min_absolute_support})"
    )
    print("=" * 60)
    header = (
        f"{'method':<22}{'yes-excl':>10}{'no-excl':>10}"
        f"{'shared':>10}{'yes total':>10}{'no total':>10}"
    )
    print(header)
    print("-" * len(header))
    for method in args.method:
        s = summaries[method]
        ex = s["exclusive"]
        yes = s["class_subsets"][CLASS_YES]
        no = s["class_subsets"][CLASS_NO]
        print(
            f"{method:<22}{ex['count_exclusive_relapse_yes']:>10}"
            f"{ex['count_exclusive_relapse_no']:>10}{ex['count_shared']:>10}"
            f"{yes['n_frequent_itemsets']:>10}{no['n_frequent_itemsets']:>10}"
        )
    print("=" * 60)

    print("\nSanity check (min absolute support used is >= 20 in both subsets):")
    for method in args.method:
        sc = summaries[method]["sanity_min_abs_support_ge_20"]
        print(f"  {method:<22} relapse_yes={sc[CLASS_YES]}  relapse_no={sc[CLASS_NO]}")


if __name__ == "__main__":
    main()

