#!/usr/bin/env python3
"""Generate side-by-side Venn diagrams of feature-selection overlap.

For each top-k subset (k in {100, 150}), the genes selected by the three
class-aware ranking methods (mutual information, chi-square, t-test) are
partitioned into the seven disjoint Venn regions and drawn as a 3-set Venn
diagram showing how few genes the rankers agree on.

Outputs a single vector PDF ``report/figures/feature_overlap_venn.pdf``.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib_venn import venn3

REPO = Path(__file__).resolve().parents[2]
BASE = REPO / "data" / "final" / "feature_selection"
OUT = Path(__file__).resolve().parent / "feature_overlap_venn.pdf"

METHODS = {"mir": "mutual_information", "chi": "chi_square", "t": "t_test"}
KS = (100, 150)
COLORS = ["#4C72B0", "#DD8452", "#55A868"]


def load_sets(k: int) -> dict:
    sets = {}
    for tag, meth in METHODS.items():
        df = pd.read_csv(BASE / meth / f"top{k}_genes_discretized.csv")
        sets[tag] = set(c for c in df.columns if c != "relapse")
    return sets


def venn_subsets(s: dict) -> tuple:
    """Return subset sizes in matplotlib_venn order:
    (A, B, AB, C, AC, BC, ABC)."""
    A, B, C = s["mir"], s["chi"], s["t"]
    abc = A & B & C
    return (
        len(A - B - C),      # 100 MI only
        len(B - A - C),      # 010 chi2 only
        len((A & B) - C),    # 110 MI & chi2
        len(C - A - B),      # 001 t only
        len((A & C) - B),    # 101 MI & t
        len((B & C) - A),    # 011 chi2 & t
        len(abc),            # 111 all three
    )


def main() -> None:
    subsets = {k: venn_subsets(load_sets(k)) for k in KS}

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.6))

    for ax, k in zip(axes, KS):
        labels = ["MI", "$\\chi^2$", "$t$-test"]
        v = venn3(
            subsets=subsets[k],
            set_labels=labels,
            set_colors=COLORS,
            alpha=0.45,
            ax=ax,
        )
        for text in v.set_labels:
            text.set_fontsize(11)
            text.set_fontweight("bold")
        for text in v.subset_labels:
            text.set_fontsize(9)

        ax.set_title(f"Top-{k} genes (union = {sum(subsets[k])})", fontsize=11)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    print(f"Saved {OUT}")

    for k in KS:
        print(f"\nk={k}: subset sizes (MI only, chi2 only, MI&chi2, t only, "
              f"MI&t, chi2&t, all three) = {subsets[k]}")


if __name__ == "__main__":
    main()