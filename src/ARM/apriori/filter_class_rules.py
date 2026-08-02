import argparse
import json
import re
from pathlib import Path

import pandas as pd

METHODS = ["chi_square", "mutual_information", "t_test"]

TOP_DIR = "top100"
RULES_NAME = "association_rules.csv"

# Canonical single-item consequents that indicate a class-association rule.
# The current pipeline feeds a binary `relapse` column (0/1) into mlxtend's
# apriori, which turns the column into the bare item `relapse` (present == 1,
# i.e. relapse = yes). A one-hot discretization would instead produce
# `relapse_yes` / `relapse_no` items. Both conventions are handled below.
RELAPSE_YES_ITEMS = frozenset({"relapse_yes"})
RELAPSE_NO_ITEMS = frozenset({"relapse_no"})
RELAPSE_BARE_ITEMS = frozenset({"relapse"})

RELAPSE_ITEM_NAMES = ("relapse", "relapse_yes", "relapse_no")

LIFT_THRESHOLDS = (1.2, 1.5, 2.0)
SUPPORT_FLAG = 0.07  # ~20 samples at n=286


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


def summarize_class_rules(rules, mask):
    sub = rules[mask]
    n = len(sub)
    lift = sub["lift"] if n else pd.Series(dtype=float)
    sup = sub["antecedent support"] if n else pd.Series(dtype=float)
    return {
        "count": int(n),
        "count_lift_gt_1.2": int((lift > LIFT_THRESHOLDS[0]).sum()),
        "count_lift_gt_1.5": int((lift > LIFT_THRESHOLDS[1]).sum()),
        "count_lift_gt_2.0": int((lift > LIFT_THRESHOLDS[2]).sum()),
        "antecedent_support": {
            "min": round(float(sup.min()), 6) if n else None,
            "max": round(float(sup.max()), 6) if n else None,
            "median": round(float(sup.median()), 6) if n else None,
            "count_lt_0.07": int((sup < SUPPORT_FLAG).sum()),
        },
    }


def print_summary(method, summary):
    print(f"\n--- {method} ---")
    print(
        f"  Item naming detected: {summary['item_naming_detected']}"
        f"  (exact consequents: relapse_yes={summary['class_item_counts']['relapse_yes']}, "
        f"relapse_no={summary['class_item_counts']['relapse_no']}, "
        f"relapse={summary['class_item_counts']['relapse']})"
    )
    for class_name in ("relapse_yes", "relapse_no"):
        info = summary["class_rules"][class_name]
        print(f"  {class_name}:")
        print(f"    total rules:                {info['count']}")
        print(f"    lift > 1.2 / > 1.5 / > 2.0: "
              f"{info['count_lift_gt_1.2']} / {info['count_lift_gt_1.5']} / {info['count_lift_gt_2.0']}")
        asup = info["antecedent_support"]
        print(
            f"    antecedent support min/max/median: "
            f"{asup['min']} / {asup['max']} / {asup['median']}"
        )
        print(f"    antecedent support < 0.07 (~20 samples): {asup['count_lt_0.07']}")


def process_method(method, rules_path, out_dir):
    df = pd.read_csv(rules_path)
    parsed = df["consequents"].map(parse_itemsets)
    n_parse_failures = int(parsed.isna().sum())

    yes_mask = parsed.eq(RELAPSE_YES_ITEMS) | parsed.eq(RELAPSE_BARE_ITEMS)
    no_mask = parsed.eq(RELAPSE_NO_ITEMS)
    class_mask = yes_mask | no_mask

    # Multi-item consequents that also mention relapse (supersets to exclude).
    superset_mask = parsed.map(
        lambda s: isinstance(s, frozenset) and len(s) > 1 and contains_relapse(s)
    )

    class_item_counts = {
        "relapse_yes": int(parsed.eq(RELAPSE_YES_ITEMS).sum()),
        "relapse_no": int(parsed.eq(RELAPSE_NO_ITEMS).sum()),
        "relapse": int(parsed.eq(RELAPSE_BARE_ITEMS).sum()),
    }

    item_naming = (
        "one_hot"
        if class_item_counts["relapse_yes"] or class_item_counts["relapse_no"]
        else "bare_binary"
    )

    summary = {
        "method": method,
        "input_file": str(rules_path),
        "total_rules": int(len(df)),
        "n_parse_failures": n_parse_failures,
        "item_naming_detected": item_naming,
        "class_item_counts": class_item_counts,
        "relapse_multi_item_consequents": int(superset_mask.sum()),
        "class_rules": {
            "relapse_yes": summarize_class_rules(df, yes_mask),
            "relapse_no": summarize_class_rules(df, no_mask),
        },
    }

    class_rules = df[class_mask].copy()
    class_rules["class"] = parsed[class_mask].map(assign_class)
    class_rules = class_rules.sort_values(["class", "lift"], ascending=[True, False])

    class_rules.to_csv(out_dir / "class_rules_filtered.csv", index=False)
    with open(out_dir / "class_rule_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print_summary(method, summary)
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Filter Apriori rules down to class-association rules "
        "predicting relapse and assess class imbalance."
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
        "--base-dir",
        type=str,
        default="data/final/ARM/apriori",
        help="Root dir containing {method}/top100 subdirs (default: "
        "data/final/ARM/apriori).",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[3]

    print("=" * 60)
    print("FILTERING CLASS-ASSOCIATION RULES")
    print("=" * 60)

    summaries = {}
    for method in args.method:
        base = project_root / args.base_dir / method / TOP_DIR
        summary = process_method(method, base / RULES_NAME, base)
        summaries[method] = summary["class_rules"]["relapse_yes"][
            "count_lift_gt_1.2"
        ]

    print()
    print("=" * 60)
    print("CROSS-METHOD COMPARISON")
    print("=" * 60)
    print(f"{'method':<22}{'relapse_yes rules, lift > 1.2':>28}")
    for method in args.method:
        print(f"{method:<22}{summaries[method]:>28}")
    best = max(args.method, key=lambda m: summaries[m])
    print()
    print(
        f"Method with most relapse_yes rules at lift > 1.2: {best} "
        f"({summaries[best]})"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
