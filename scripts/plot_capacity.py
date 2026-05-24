"""
Plot distilled student accuracy as a function of student capacity (parameters),
overlaid for the depth ablation (varying number of residual blocks at width=16)
and the width ablation (varying channel multiplier at depth=20).

Both series share the ResNet-20 (depth=20, width=16) pivot point.

Reference lines:
  - Hard-label ResNet-20 baseline (67.63%) - what distillation must beat
  - Teacher ceiling (73.48%) - the upper bound

Tolerant of missing cells: any cell whose summary.json is absent is skipped
silently (useful while Step 3 sweep is still in flight).
"""

import os
import sys
import json
import argparse

import matplotlib.pyplot as plt


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)


# (label, results-dir-name, expected param count) for each cell on the two axes.
DEPTH_CELLS = [
    ("ResNet-20", "resnet20_t8_a0p5",         278_324),
    ("ResNet-32", "resnet32_t8_a0p5",         472_756),
    ("ResNet-44", "resnet44_t8_a0p5",         667_188),
    ("ResNet-56", "resnet56_t8_a0p5",         861_620),
]

WIDTH_CELLS = [
    ("w=0.5", "resnet20_w0p5_t8_a0p5",         71_756),
    ("w=1",   "resnet20_t8_a0p5",             278_324),
    ("w=2",   "resnet20_w2_t8_a0p5",        1_096_196),
]

BASELINE_TOP1 = 67.63      # hard-label ResNet-20 w=1 baseline
TEACHER_TOP1 = 73.48       # ResNet-110 teacher


def load_top1(results_dir, cell_name):
    summary_path = os.path.join(results_dir, cell_name, "summary.json")
    if not os.path.exists(summary_path):
        return None
    with open(summary_path) as f:
        s = json.load(f)
    return s["best_test_top1"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root",
                        default=os.path.join(PROJECT_ROOT, "results/distillation"))
    parser.add_argument("--out",
                        default=os.path.join(PROJECT_ROOT, "plots/analysis/capacity.png"))
    args = parser.parse_args()

    fig, ax = plt.subplots(figsize=(10, 6))

    # Depth series
    depth_params, depth_top1, depth_labels = [], [], []
    for label, dir_name, params in DEPTH_CELLS:
        top1 = load_top1(args.results_root, dir_name)
        if top1 is None:
            print(f"[skip] depth cell missing: {dir_name}")
            continue
        depth_params.append(params); depth_top1.append(top1); depth_labels.append(label)

    if depth_params:
        ax.plot(depth_params, depth_top1, "o-", color="C0",
                linewidth=2, markersize=10, label="Depth (width fixed at 16)")
        for x, y, lab in zip(depth_params, depth_top1, depth_labels):
            ax.annotate(f"{lab}\n{y:.2f}%", xy=(x, y), xytext=(8, -4),
                        textcoords="offset points", fontsize=9,
                        color="C0", fontweight="bold")

    # Width series
    width_params, width_top1, width_labels = [], [], []
    for label, dir_name, params in WIDTH_CELLS:
        top1 = load_top1(args.results_root, dir_name)
        if top1 is None:
            print(f"[skip] width cell missing: {dir_name}")
            continue
        width_params.append(params); width_top1.append(top1); width_labels.append(label)

    if width_params:
        ax.plot(width_params, width_top1, "s-", color="C3",
                linewidth=2, markersize=10, label="Width (depth fixed at 20)")
        for x, y, lab in zip(width_params, width_top1, width_labels):
            ax.annotate(f"{lab}\n{y:.2f}%", xy=(x, y), xytext=(8, 10),
                        textcoords="offset points", fontsize=9,
                        color="C3", fontweight="bold")

    # Reference lines
    ax.axhline(y=BASELINE_TOP1, color="gray", linestyle="--", linewidth=1.5,
               label=f"Hard-label ResNet-20 baseline ({BASELINE_TOP1:.2f}%)")
    ax.axhline(y=TEACHER_TOP1, color="black", linestyle=":", linewidth=1.5,
               label=f"Teacher ceiling ResNet-110 ({TEACHER_TOP1:.2f}%)")

    ax.set_xscale("log")
    ax.set_xlabel("Student parameters", fontsize=11)
    ax.set_ylabel("Test Top-1 (%)", fontsize=11)
    ax.set_title(
        "Distilled student accuracy vs. capacity (τ=8, α=0.5)",
        fontsize=12, fontweight="bold",
    )
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="lower right", fontsize=9)

    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
