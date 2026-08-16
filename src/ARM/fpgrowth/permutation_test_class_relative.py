import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from run_class_relative_fpgrowth import (
    CLASS_NO,
    CLASS_YES,
    MAX_LEN,
    METHODS,
    MIN_ABS_SUPPORT,
    TOP_KIDS,
    find_exclusive,
    mine_subset,
    parse_itemset,
    split_by_class,
)

METHOD = "mutual_information"
TOP_DIR = "top100"
RELATIVE_DIR_NAME = "class_relative"

# Fixed master seed: makes the whole shuffled batch rerunnable deterministically.
MASTER_SEED = 20260806
N_SHUFFLES = 20

PERMUTATION_RESULTS_NAME = "permutation_results.json"

SIZES = (1, 2, 3)


def _fsrows(fi, n):
    """Recover itemsets as frozensets and exact absolute counts."""
    im = {}
    for _, row in fi.iterrows():
        fs = parse_itemset(row["itemsets"])
        if fs is not None:
            im[fs] = {"count": round(row["support"] * n)}
    return im


def class_relative_metrics(df, min_abs, max_len):
    """Run the identical class-relative FP-Growth procedure on `df`.

    Mirrors run_method()'s subset mining + exclusivity comparison on the given
    dataframe (real or label-shuffled). Returns per-subset totals, exclusivity
    counts, and exclusivity counts split by itemset size.
    """
    yes_df, no_df = split_by_class(df)
    subsets = {
        CLASS_YES: (yes_df, len(yes_df)),
        CLASS_NO: (no_df, len(no_df)),
    }

    found = {}
    for class_name, (subset_df, n) in subsets.items():
        min_support = min_abs / n
        fi = mine_subset(subset_df, min_support, max_len)
        found[class_name] = _fsrows(fi, n)

    yes_excl = find_exclusive(found[CLASS_YES], found[CLASS_NO])
    no_excl = find_exclusive(found[CLASS_NO], found[CLASS_YES])

    def size_split(rows):
        counts = {s: 0 for s in SIZES}
        for itemset, _here, _other in rows:
            s = len(itemset)
            counts[s] = counts.get(s, 0) + 1
        return counts

    return {
        "total_yes": len(found[CLASS_YES]),
        "total_no": len(found[CLASS_NO]),
        "yes_excl": len(yes_excl),
        "no_excl": len(no_excl),
        "yes_excl_by_size": size_split(yes_excl),
        "no_excl_by_size": size_split(no_excl),
    }


def shuffled_labels(rng, n_total, n_yes):
    """Permutation-preserving assignment: n_yes ones / rest zeros."""
    labels = np.zeros(n_total, dtype=int)
    labels[:n_yes] = 1
    rng.shuffle(labels)
    return labels


def summarize(arr):
    a = np.asarray(arr, dtype=float)
    return {
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "std": float(a.std()),
        "min": float(a.min()),
        "max": float(a.max()),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Permutation null baseline for class-relative exclusivity. "
        "Shuffles the relapse labels (preserving the 107/179 split) and "
        "re-runs the class-relative FP-Growth procedure to quantify how many "
        "class-exclusive itemsets appear by chance alone."
    )
    parser.add_argument(
        "--method", type=str, choices=METHODS, default=METHOD,
        help="Feature selection method to test (default: mutual_information).",
    )
    parser.add_argument(
        "--top-k", type=str, choices=TOP_KIDS, default=TOP_DIR,
        help="Which discretized gene subset to test (top100/top150).",
    )
    parser.add_argument(
        "--fs-dir", type=str, default="data/final/feature_selection",
        help="Dir containing {method}/{top-k}_genes_discretized.csv.",
    )
    parser.add_argument(
        "--base-dir", type=str, default="data/final/ARM/fpgrowth_v2",
        help="Root under which outputs/method/top100/class_relative/ is written.",
    )
    parser.add_argument(
        "--min-absolute-support", type=int, default=MIN_ABS_SUPPORT,
    )
    parser.add_argument("--max-len", type=int, default=MAX_LEN)
    parser.add_argument(
        "--n-shuffles", type=int, default=N_SHUFFLES,
        help="Number of label permutations (default 20).",
    )
    parser.add_argument("--master-seed", type=int, default=MASTER_SEED)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[3]
    method = args.method
    top_k = args.top_k
    input_path = project_root / args.fs_dir / method / f"{top_k}_genes_discretized.csv"
    out_root = project_root / args.base_dir / method / top_k / RELATIVE_DIR_NAME
    out_root.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PERMUTATION NULL TEST — class-relative exclusivity")
    print("=" * 60)
    print(f"Input: {input_path}")

    df = pd.read_csv(input_path)
    n_total = len(df)
    if "relapse" in df.columns:
        class_col = "relapse"
    elif CLASS_YES in df.columns:
        class_col = CLASS_YES  # fallback, treat 0/1 in relapse_yes
    else:
        raise ValueError("No binary class column usable for shuffling.")
    n_yes = int(df[class_col].astype(bool).sum())
    n_no = n_total - n_yes

    gene_cols = [c for c in df.columns if c not in (CLASS_YES, CLASS_NO, class_col)]
    work = df[gene_cols].copy()
    work[class_col] = df[class_col].astype(int).values

    # --- unshuffled (real labels) metrics ----------------------------------
    real = class_relative_metrics(
        work.copy(), args.min_absolute_support, args.max_len
    )
    print(
        f"\nREAL (unshuffled): yes-excl={real['yes_excl']}, "
        f"no-excl={real['no_excl']} "
        f"(totals yes={real['total_yes']}, no={real['total_no']})"
    )
    print(f"  yes-excl by size: {real['yes_excl_by_size']}")
    print(f"  no-excl  by size: {real['no_excl_by_size']}")
    print(f"  class sizes preserved: yes={n_yes}, no={n_no}")

    # --- permutation runs ---------------------------------------------------
    yes_excl_counts, no_excl_counts = [], []
    yes_by_size = {s: [] for s in SIZES}
    no_by_size = {s: [] for s in SIZES}
    totals_yes, totals_no = [], []
    shuffle_runs = []

    master_rng = np.random.default_rng(args.master_seed)
    for i in range(args.n_shuffles):
        labels = shuffled_labels(master_rng, n_total, n_yes)
        perm = work.copy()
        perm[class_col] = labels

        start = time.time()
        m = class_relative_metrics(perm, args.min_absolute_support, args.max_len)
        elapsed = time.time() - start

        yes_excl_counts.append(m["yes_excl"])
        no_excl_counts.append(m["no_excl"])
        totals_yes.append(m["total_yes"])
        totals_no.append(m["total_no"])
        for s in SIZES:
            yes_by_size[s].append(m["yes_excl_by_size"][s])
            no_by_size[s].append(m["no_excl_by_size"][s])

        shuffle_runs.append(
            {
                "shuffle_index": i + 1,
                "yes_excl": m["yes_excl"],
                "no_excl": m["no_excl"],
                "total_yes": m["total_yes"],
                "total_no": m["total_no"],
                "yes_excl_by_size": m["yes_excl_by_size"],
                "no_excl_by_size": m["no_excl_by_size"],
                "runtime_seconds": round(elapsed, 3),
            }
        )
        print(
            f"  shuffle {i+1:>2}/{args.n_shuffles}: "
            f"yes-excl={m['yes_excl']:>6}, no-excl={m['no_excl']:>8} ({elapsed:.2f}s)"
        )

    # --- empirical p-value-style stats -------------------------------------
    p_yes = float(np.mean(np.asarray(yes_excl_counts) >= real["yes_excl"]))
    p_no = float(np.mean(np.asarray(no_excl_counts) >= real["no_excl"]))
    yes_size_mean = {s: float(np.mean(yes_by_size[s])) for s in SIZES}
    no_size_mean = {s: float(np.mean(no_by_size[s])) for s in SIZES}

    print("\n" + "=" * 60)
    print("PERMUTATION NULL DISTRIBUTION")
    print("=" * 60)
    print(f"  yes-exclusive: {summarize(yes_excl_counts)}")
    print(f"  no-exclusive : {summarize(no_excl_counts)}")
    print(
        f"\n  REAL yes-excl={real['yes_excl']}; "
        f"fraction of {args.n_shuffles} shuffles >= real: {p_yes:.2f}"
    )
    print(
        f"  REAL no-excl ={real['no_excl']}; "
        f"fraction of {args.n_shuffles} shuffles >= real: {p_no:.2f}"
    )
    print("\n  yes-exclusive by itemset size: real vs mean of shuffles")
    for s in SIZES:
        print(
            f"    size {s}-item: real={real['yes_excl_by_size'][s]:>7}  "
            f"mean_shuffled={yes_size_mean[s]:>8.1f}"
        )
    print("  no-exclusive by itemset size: real vs mean of shuffles")
    for s in SIZES:
        print(
            f"    size {s}-item: real={real['no_excl_by_size'][s]:>9}  "
            f"mean_shuffled={no_size_mean[s]:>8.1f}"
        )

    # --- verdict on yes-exclusive ------------------------------------------
    lo, hi = min(yes_excl_counts), max(yes_excl_counts)
    if real["yes_excl"] > hi:
        verdict = (
            f"OUTSIDE null range: real yes-excl={real['yes_excl']} > "
            f"max(shuffled)={hi}. Signal not explained by chance alone "
            f"(p~{p_yes:.2f})."
        )
    else:
        verdict = (
            f"INSIDE null range: real yes-excl={real['yes_excl']} falls within "
            f"shuffled [{lo}, {hi}]. Consistent with pure chance."
        )
    print("\n" + "=" * 60)
    print("VERDICT — yes-exclusive: how do you know this isn't just noise?")
    print("=" * 60)
    print("  " + verdict)

    out = {
        "method": method,
        "input_file": str(input_path),
        "n_samples": n_total,
        "class_sizes": {CLASS_YES: n_yes, CLASS_NO: n_no},
        "min_absolute_support": args.min_absolute_support,
        "max_len": args.max_len,
        "n_shuffles": args.n_shuffles,
        "master_seed": args.master_seed,
        "real": {
            "yes_excl": real["yes_excl"],
            "no_excl": real["no_excl"],
            "total_yes": real["total_yes"],
            "total_no": real["total_no"],
            "yes_excl_by_size": real["yes_excl_by_size"],
            "no_excl_by_size": real["no_excl_by_size"],
        },
        "real_reference": {
            "yes_excl": real["yes_excl"],
            "no_excl": real["no_excl"],
        },
        "shuffles": shuffle_runs,
        "null_distribution": {
            "yes_excl": summarize(yes_excl_counts),
            "no_excl": summarize(no_excl_counts),
        },
        "yes_excl_by_size_mean_of_shuffles": yes_size_mean,
        "no_excl_by_size_mean_of_shuffles": no_size_mean,
        "empirical_p_fraction_shuffles_ge_real": {
            "yes_excl": p_yes,
            "no_excl": p_no,
        },
        "verdict": {
            "yes_excl_outside_null": real["yes_excl"] > hi,
            "text": verdict,
        },
    }

    out_file = out_root / PERMUTATION_RESULTS_NAME
    with open(out_file, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote: {out_file}")


if __name__ == "__main__":
    main()