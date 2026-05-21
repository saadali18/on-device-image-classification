"""
Plot consolidated training curves from a metrics.csv into a single figure.

Auto-detects mode:
  - "basic"        : baseline / teacher runs (no hard/soft loss columns)
                     Produces a 2x2 grid: Loss, Top-1, Top-5, LR.
  - "distillation" : distillation runs (has train_loss_hard, train_loss_soft)
                     Produces a 2x3 grid: Total Loss, Hard CE, Soft KL, Top-1, Top-5, LR.

This replaces the four-separate-PNG style with a single self-contained figure
per experiment, per the team's plotting convention.
"""

import os
import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)


REQUIRED_BASIC = [
    "epoch",
    "lr",
    "train_loss",
    "train_top1",
    "train_top5",
    "test_loss",
    "test_top1",
    "test_top5",
]

REQUIRED_DISTILL_EXTRA = ["train_loss_hard", "train_loss_soft"]


def detect_mode(df: pd.DataFrame) -> str:
    if all(col in df.columns for col in REQUIRED_DISTILL_EXTRA):
        return "distillation"
    return "basic"


def validate_columns(df: pd.DataFrame, mode: str) -> None:
    missing = [c for c in REQUIRED_BASIC if c not in df.columns]
    if mode == "distillation":
        missing += [c for c in REQUIRED_DISTILL_EXTRA if c not in df.columns]
    if missing:
        raise ValueError(f"metrics.csv is missing required columns: {missing}")


def _setup_axis(ax, xlabel, ylabel, title, legend=True):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if legend:
        ax.legend()
    ax.grid(True, alpha=0.3)


def plot_basic(df: pd.DataFrame, out_path: str, title: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    ax = axes[0, 0]
    ax.plot(df["epoch"], df["train_loss"], label="Train")
    ax.plot(df["epoch"], df["test_loss"], label="Test")
    _setup_axis(ax, "Epoch", "Loss", "Loss")

    ax = axes[0, 1]
    ax.plot(df["epoch"], df["train_top1"], label="Train")
    ax.plot(df["epoch"], df["test_top1"], label="Test")
    _setup_axis(ax, "Epoch", "Top-1 Accuracy (%)", "Top-1 Accuracy")

    ax = axes[1, 0]
    ax.plot(df["epoch"], df["train_top5"], label="Train")
    ax.plot(df["epoch"], df["test_top5"], label="Test")
    _setup_axis(ax, "Epoch", "Top-5 Accuracy (%)", "Top-5 Accuracy")

    ax = axes[1, 1]
    ax.plot(df["epoch"], df["lr"])
    _setup_axis(ax, "Epoch", "Learning Rate", "LR schedule", legend=False)
    ax.set_yscale("log")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_distillation(df: pd.DataFrame, out_path: str, title: str) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))

    ax = axes[0, 0]
    ax.plot(df["epoch"], df["train_loss"], label="Train (total)")
    ax.plot(df["epoch"], df["test_loss"], label="Test (CE only)")
    _setup_axis(ax, "Epoch", "Loss", "Total Loss")

    ax = axes[0, 1]
    ax.plot(df["epoch"], df["train_loss_hard"], color="C0")
    _setup_axis(ax, "Epoch", "Hard CE", "Hard-Label CE Component (train)", legend=False)

    ax = axes[0, 2]
    ax.plot(df["epoch"], df["train_loss_soft"], color="C1")
    _setup_axis(ax, "Epoch", "Soft KL", "Soft-Label KL Component (train)", legend=False)

    ax = axes[1, 0]
    ax.plot(df["epoch"], df["train_top1"], label="Train")
    ax.plot(df["epoch"], df["test_top1"], label="Test")
    _setup_axis(ax, "Epoch", "Top-1 Accuracy (%)", "Top-1 Accuracy")

    ax = axes[1, 1]
    ax.plot(df["epoch"], df["train_top5"], label="Train")
    ax.plot(df["epoch"], df["test_top5"], label="Test")
    _setup_axis(ax, "Epoch", "Top-5 Accuracy (%)", "Top-5 Accuracy")

    ax = axes[1, 2]
    ax.plot(df["epoch"], df["lr"])
    _setup_axis(ax, "Epoch", "Learning Rate", "LR schedule", legend=False)
    ax.set_yscale("log")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def print_summary(df: pd.DataFrame, mode: str) -> None:
    best = df.loc[df["test_top1"].idxmax()]
    print("=" * 80)
    print(f"BEST EPOCH SUMMARY  (mode={mode})")
    print("=" * 80)
    print(f"Best Epoch          : {int(best['epoch'])}")
    print(f"Test Top-1          : {best['test_top1']:.2f}%")
    print(f"Test Top-5          : {best['test_top5']:.2f}%")
    print(f"Train Top-1 @ best  : {best['train_top1']:.2f}%")
    print(f"Train Top-5 @ best  : {best['train_top5']:.2f}%")
    if mode == "distillation":
        print(f"Train Hard @ best   : {best['train_loss_hard']:.4f}")
        print(f"Train Soft @ best   : {best['train_loss_soft']:.4f}")
    print(f"Train Loss @ best   : {best['train_loss']:.4f}")
    print(f"Test  Loss @ best   : {best['test_loss']:.4f}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics",
        required=True,
        help="Path to metrics.csv (e.g. results/baseline/resnet20/metrics.csv).",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output PNG path. If omitted, saved as <metrics-dir-name>/training_curves.png "
        "next to the metrics file (under plots/, mirroring the structure).",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Figure title. Defaults to the immediate parent directory name.",
    )
    args = parser.parse_args()

    metrics_path = args.metrics
    if not os.path.isabs(metrics_path):
        metrics_path = os.path.abspath(metrics_path)
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"metrics.csv not found: {metrics_path}")

    df = pd.read_csv(metrics_path)
    mode = detect_mode(df)
    validate_columns(df, mode)

    # Default output: mirror results/ -> plots/ structure.
    if args.out is None:
        parts = os.path.normpath(metrics_path).split(os.sep)
        if "results" in parts:
            idx = parts.index("results")
            parts[idx] = "plots"
            out_dir = os.sep.join(parts[: -1])
        else:
            out_dir = os.path.dirname(metrics_path)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "training_curves.png")
    else:
        out_path = args.out
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    title = args.title or os.path.basename(os.path.dirname(metrics_path))

    print_summary(df, mode)

    if mode == "distillation":
        plot_distillation(df, out_path, title)
    else:
        plot_basic(df, out_path, title)

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
