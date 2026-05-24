"""
Compute Expected Calibration Error (ECE) and produce reliability diagrams
for the teacher, the hard-label baseline student, and the distilled student.

ECE measures the gap between a model's confidence (max softmax prob) and its
actual accuracy, averaged over confidence-bucketed bins. A well-calibrated
classifier has ECE near 0 (the reliability diagram lies on the diagonal).

The interesting finding (from the reference final report, Table 8): KD students
often *inherit* teacher overconfidence and have HIGHER ECE than the hard-label
student despite having higher accuracy. This is a non-obvious result that
makes for a strong discussion point.

Usage:
    python scripts/plot_calibration.py \
        --distilled-checkpoint checkpoints/distillation/resnet20_t4_a0p5/best.pth \
        --out plots/analysis/calibration.png
"""

import os
import sys
import argparse

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from models.resnet_cifar import build_resnet_cifar
from utils.checkpoint import load_checkpoint


CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


def get_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def load_model(arch, width, num_classes, ckpt_path, device):
    model = build_resnet_cifar(arch, num_classes=num_classes, width=width).to(device)
    load_checkpoint(ckpt_path, model=model, device=device)
    model.eval()
    return model


@torch.no_grad()
def collect_predictions(model, loader, device):
    """Returns (max_probs, predictions, labels) all as 1-D numpy arrays."""
    max_probs_all, preds_all, labels_all = [], [], []
    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        probs = F.softmax(logits, dim=-1)
        max_probs, preds = probs.max(dim=-1)
        max_probs_all.append(max_probs.cpu().numpy())
        preds_all.append(preds.cpu().numpy())
        labels_all.append(labels.numpy())
    return (
        np.concatenate(max_probs_all),
        np.concatenate(preds_all),
        np.concatenate(labels_all),
    )


def expected_calibration_error(max_probs, preds, labels, n_bins=15):
    """
    ECE = sum_b (n_b / N) * |acc_b - conf_b|
    where acc_b, conf_b are accuracy and avg confidence in bin b.
    Returns (ece, bin_accuracies, bin_confidences, bin_counts).
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_lowers = bin_edges[:-1]
    bin_uppers = bin_edges[1:]

    correct = (preds == labels).astype(np.float32)

    bin_acc = np.zeros(n_bins)
    bin_conf = np.zeros(n_bins)
    bin_count = np.zeros(n_bins)

    for i in range(n_bins):
        # Right-inclusive bins; left-inclusive for the lowest bin.
        if i == 0:
            in_bin = (max_probs >= bin_lowers[i]) & (max_probs <= bin_uppers[i])
        else:
            in_bin = (max_probs > bin_lowers[i]) & (max_probs <= bin_uppers[i])

        count = in_bin.sum()
        bin_count[i] = count
        if count > 0:
            bin_acc[i] = correct[in_bin].mean()
            bin_conf[i] = max_probs[in_bin].mean()

    n = len(max_probs)
    ece = float(np.sum((bin_count / n) * np.abs(bin_acc - bin_conf)))
    return ece, bin_acc, bin_conf, bin_count


def plot_reliability(ax, bin_acc, bin_conf, bin_count, ece, title, n_bins=15):
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    width = 1.0 / n_bins

    ax.bar(bin_centers, bin_acc, width=width, edgecolor="black",
           color="C0", alpha=0.7, label="Empirical accuracy")
    # Gap bars (red): conf - acc
    gap = bin_conf - bin_acc
    ax.bar(bin_centers, gap, width=width, bottom=bin_acc, edgecolor="black",
           color="C3", alpha=0.4, hatch="//", label="Gap (conf − acc)")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence (max softmax prob)")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{title}\nECE = {ece * 100:.2f}%")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)


def load_test_loader(data_dir, batch_size=256):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ]
    )
    dataset = torchvision.datasets.CIFAR100(
        root=data_dir, train=False, download=False, transform=transform
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=0, pin_memory=False)
    return loader, dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--teacher-checkpoint",
        default=os.path.join(PROJECT_ROOT, "checkpoints/teacher/resnet110/best.pth"),
    )
    parser.add_argument("--teacher-arch", default="resnet110")
    parser.add_argument("--teacher-width", type=int, default=16)

    parser.add_argument(
        "--baseline-checkpoint",
        default=os.path.join(PROJECT_ROOT, "checkpoints/baseline/resnet20/best.pth"),
    )
    parser.add_argument("--baseline-arch", default="resnet20")
    parser.add_argument("--baseline-width", type=int, default=16)

    parser.add_argument("--distilled-checkpoint", required=True)
    parser.add_argument("--distilled-arch", default="resnet20")
    parser.add_argument("--distilled-width", type=int, default=16)

    parser.add_argument("--n-bins", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--data-dir", default=os.path.join(PROJECT_ROOT, "data"))
    parser.add_argument(
        "--out",
        default=os.path.join(PROJECT_ROOT, "plots/analysis/calibration.png"),
    )
    parser.add_argument("--device", default="auto")

    args = parser.parse_args()

    device = get_device(args.device)
    print(f"Device: {device}")

    loader, dataset = load_test_loader(args.data_dir, batch_size=args.batch_size)
    num_classes = len(dataset.classes)

    models = {
        "Teacher (ResNet-110)": load_model(
            args.teacher_arch, args.teacher_width, num_classes,
            args.teacher_checkpoint, device,
        ),
        "Hard-label student": load_model(
            args.baseline_arch, args.baseline_width, num_classes,
            args.baseline_checkpoint, device,
        ),
        "Distilled student": load_model(
            args.distilled_arch, args.distilled_width, num_classes,
            args.distilled_checkpoint, device,
        ),
    }

    results = {}
    print("\n" + "=" * 80)
    print(f"CALIBRATION  (n_bins={args.n_bins})")
    print("=" * 80)
    for name, model in models.items():
        max_probs, preds, labels = collect_predictions(model, loader, device)
        acc = (preds == labels).mean()
        ece, bin_acc, bin_conf, bin_count = expected_calibration_error(
            max_probs, preds, labels, n_bins=args.n_bins
        )
        results[name] = {
            "max_probs": max_probs,
            "preds": preds,
            "labels": labels,
            "acc": acc,
            "ece": ece,
            "bin_acc": bin_acc,
            "bin_conf": bin_conf,
            "bin_count": bin_count,
            "mean_confidence": max_probs.mean(),
        }
        print(
            f"{name:25s} Top-1: {acc * 100:5.2f}%  "
            f"ECE: {ece * 100:5.2f}%  "
            f"mean max-prob: {max_probs.mean():.3f}"
        )
    print("=" * 80)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, (name, r) in zip(axes, results.items()):
        plot_reliability(
            ax, r["bin_acc"], r["bin_conf"], r["bin_count"], r["ece"],
            title=name, n_bins=args.n_bins,
        )

    fig.suptitle(
        "Reliability diagrams — KD students often inherit teacher overconfidence",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
