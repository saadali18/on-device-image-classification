import os
import sys
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)


def plot_loss(metrics_df: pd.DataFrame, output_dir: str):
    plt.figure(figsize=(10, 6))

    plt.plot(metrics_df["epoch"], metrics_df["train_loss"], label="Train Loss")
    plt.plot(metrics_df["epoch"], metrics_df["test_loss"], label="Test Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Baseline ResNet-20 CIFAR-100 Loss Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)

    output_path = os.path.join(output_dir, "loss_curve.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved loss curve to: {output_path}")


def plot_accuracy(metrics_df: pd.DataFrame, output_dir: str):
    plt.figure(figsize=(10, 6))

    plt.plot(metrics_df["epoch"], metrics_df["train_top1"], label="Train Top-1")
    plt.plot(metrics_df["epoch"], metrics_df["test_top1"], label="Test Top-1")
    plt.plot(metrics_df["epoch"], metrics_df["train_top5"], label="Train Top-5")
    plt.plot(metrics_df["epoch"], metrics_df["test_top5"], label="Test Top-5")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Baseline ResNet-20 CIFAR-100 Accuracy Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)

    output_path = os.path.join(output_dir, "accuracy_curve.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved accuracy curve to: {output_path}")


def plot_top1_accuracy(metrics_df: pd.DataFrame, output_dir: str):
    plt.figure(figsize=(10, 6))

    plt.plot(metrics_df["epoch"], metrics_df["train_top1"], label="Train Top-1")
    plt.plot(metrics_df["epoch"], metrics_df["test_top1"], label="Test Top-1")

    plt.xlabel("Epoch")
    plt.ylabel("Top-1 Accuracy (%)")
    plt.title("Baseline ResNet-20 CIFAR-100 Top-1 Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)

    output_path = os.path.join(output_dir, "top1_accuracy_curve.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved Top-1 accuracy curve to: {output_path}")


def plot_learning_rate(metrics_df: pd.DataFrame, output_dir: str):
    plt.figure(figsize=(10, 6))

    plt.plot(metrics_df["epoch"], metrics_df["lr"], label="Learning Rate")

    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("Baseline ResNet-20 Learning Rate Schedule")
    plt.legend()
    plt.grid(True, alpha=0.3)

    output_path = os.path.join(output_dir, "learning_rate_curve.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved learning rate curve to: {output_path}")


def print_summary(metrics_df: pd.DataFrame):
    best_row = metrics_df.loc[metrics_df["test_top1"].idxmax()]

    print("=" * 80)
    print("BASELINE SUMMARY")
    print("=" * 80)
    print(f"Best Epoch: {int(best_row['epoch'])}")
    print(f"Best Test Top-1: {best_row['test_top1']:.2f}%")
    print(f"Best Test Top-5: {best_row['test_top5']:.2f}%")
    print(f"Train Top-1 at Best Epoch: {best_row['train_top1']:.2f}%")
    print(f"Train Top-5 at Best Epoch: {best_row['train_top5']:.2f}%")
    print(f"Train Loss at Best Epoch: {best_row['train_loss']:.4f}")
    print(f"Test Loss at Best Epoch: {best_row['test_loss']:.4f}")
    print("=" * 80)


def main():
    metrics_path = os.path.join(
        PROJECT_ROOT,
        "results",
        "baseline",
        "resnet20",
        "metrics.csv",
    )

    output_dir = os.path.join(
        PROJECT_ROOT,
        "plots",
        "baseline",
        "resnet20",
    )

    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(metrics_path):
        raise FileNotFoundError(
            f"Metrics file not found: {metrics_path}\n"
            "Run baseline training first: python scripts/train_baseline.py"
        )

    metrics_df = pd.read_csv(metrics_path)

    required_columns = [
        "epoch",
        "lr",
        "train_loss",
        "train_top1",
        "train_top5",
        "test_loss",
        "test_top1",
        "test_top5",
    ]

    missing_columns = [
        col for col in required_columns if col not in metrics_df.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing columns in metrics.csv: {missing_columns}")

    print_summary(metrics_df)

    plot_loss(metrics_df, output_dir)
    plot_accuracy(metrics_df, output_dir)
    plot_top1_accuracy(metrics_df, output_dir)
    plot_learning_rate(metrics_df, output_dir)

    print("Done.")


if __name__ == "__main__":
    main()