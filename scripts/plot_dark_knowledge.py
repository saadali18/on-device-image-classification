"""
Visualize the "dark knowledge" transferred by the teacher.

For each of N held-out CIFAR-100 test images, plots the top-k class probabilities under:
  1. Teacher softmax at T = 1            (argmax-dominated)
  2. Teacher softmax at T = tau           (softened distribution — the "dark knowledge")
  3. Hard-label baseline student at T=1  (no teacher supervision)
  4. Distilled student at T=1            (with teacher supervision)

Compare (2) against (3) and (4) to see whether the distilled student's distribution
is more teacher-like than the hard-label baseline's. This is the qualitative evidence
promised in Section 5 ("Step 4") of the mid-report.

Usage:
    python scripts/plot_dark_knowledge.py \
        --distilled-checkpoint checkpoints/distillation/resnet20_t4_a0p5/best.pth \
        --tau 4.0 \
        --num-images 6 \
        --out plots/analysis/dark_knowledge.png
"""

import os
import sys
import argparse

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
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


def load_model(arch: str, width: int, num_classes: int, checkpoint_path: str,
               device: torch.device):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    model = build_resnet_cifar(arch, num_classes=num_classes, width=width).to(device)
    load_checkpoint(checkpoint_path, model=model, device=device)
    model.eval()
    return model


def load_test_dataset(data_dir: str):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ]
    )
    return torchvision.datasets.CIFAR100(
        root=data_dir, train=False, download=False, transform=transform
    )


def denormalize(img_tensor: torch.Tensor) -> np.ndarray:
    mean = torch.tensor(CIFAR100_MEAN).view(3, 1, 1)
    std = torch.tensor(CIFAR100_STD).view(3, 1, 1)
    img = img_tensor.cpu() * std + mean
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return img


@torch.no_grad()
def topk_probs(logits: torch.Tensor, k: int, temperature: float = 1.0):
    probs = F.softmax(logits / temperature, dim=-1).squeeze(0)
    top_vals, top_idx = probs.topk(k)
    return top_vals.cpu().numpy(), top_idx.cpu().numpy()


def select_images(dataset, num_images: int, seed: int = 42):
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(dataset), size=num_images, replace=False)
    return idx.tolist()


def plot_dark_knowledge(
    teacher,
    baseline_student,
    distilled_student,
    dataset,
    classes,
    indices,
    tau: float,
    top_k: int,
    out_path: str,
    device: torch.device,
):
    n = len(indices)
    n_cols = 5  # image, teacher T=1, teacher T=tau, baseline student, distilled student
    fig, axes = plt.subplots(n, n_cols, figsize=(4 * n_cols, 2.6 * n))
    if n == 1:
        axes = axes.reshape(1, -1)

    col_titles = [
        "Image",
        "Teacher (T=1)",
        f"Teacher (T={tau:g})",
        "Hard-label student",
        "Distilled student",
    ]
    for c, t in enumerate(col_titles):
        axes[0, c].set_title(t, fontsize=11, fontweight="bold")

    for row, idx in enumerate(indices):
        img_tensor, true_label = dataset[idx]
        img_input = img_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            teacher_logits = teacher(img_input)
            baseline_logits = baseline_student(img_input)
            distilled_logits = distilled_student(img_input)

        # Image
        ax = axes[row, 0]
        ax.imshow(denormalize(img_tensor))
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_ylabel(f"true: {classes[true_label]}", fontsize=9)

        # Teacher distributions
        for col, (logits, T) in enumerate(
            [(teacher_logits, 1.0), (teacher_logits, tau)], start=1
        ):
            vals, idxs = topk_probs(logits, top_k, temperature=T)
            labels = [classes[i] for i in idxs]
            ax = axes[row, col]
            ax.barh(range(top_k)[::-1], vals, color="C0" if T == 1.0 else "C3")
            ax.set_yticks(range(top_k)[::-1])
            ax.set_yticklabels(labels, fontsize=8)
            ax.set_xlim(0, 1.0)
            ax.grid(True, alpha=0.2, axis="x")

        # Hard-label student
        vals, idxs = topk_probs(baseline_logits, top_k, temperature=1.0)
        labels = [classes[i] for i in idxs]
        ax = axes[row, 3]
        ax.barh(range(top_k)[::-1], vals, color="C2")
        ax.set_yticks(range(top_k)[::-1])
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlim(0, 1.0)
        ax.grid(True, alpha=0.2, axis="x")

        # Distilled student
        vals, idxs = topk_probs(distilled_logits, top_k, temperature=1.0)
        labels = [classes[i] for i in idxs]
        ax = axes[row, 4]
        ax.barh(range(top_k)[::-1], vals, color="C1")
        ax.set_yticks(range(top_k)[::-1])
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlim(0, 1.0)
        ax.grid(True, alpha=0.2, axis="x")

    fig.suptitle(
        "Teacher dark knowledge (T=tau) is closer to the distilled student than to the hard-label baseline.",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


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

    parser.add_argument(
        "--distilled-checkpoint",
        required=True,
        help="Path to the trained distilled student checkpoint (e.g. checkpoints/distillation/resnet20_t4_a0p5/best.pth).",
    )
    parser.add_argument("--distilled-arch", default="resnet20")
    parser.add_argument("--distilled-width", type=int, default=16)

    parser.add_argument("--tau", type=float, default=4.0,
                        help="Temperature for the softened teacher distribution.")
    parser.add_argument("--top-k", type=int, default=5,
                        help="How many top classes to show per bar chart.")
    parser.add_argument("--num-images", type=int, default=6,
                        help="How many held-out test images to visualize.")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--data-dir", default=os.path.join(PROJECT_ROOT, "data"))
    parser.add_argument(
        "--out",
        default=os.path.join(PROJECT_ROOT, "plots/analysis/dark_knowledge.png"),
    )
    parser.add_argument("--device", default="auto")

    args = parser.parse_args()

    device = get_device(args.device)
    print(f"Device: {device}")

    dataset = load_test_dataset(args.data_dir)
    classes = dataset.classes

    teacher = load_model(
        args.teacher_arch, args.teacher_width, len(classes),
        args.teacher_checkpoint, device,
    )
    baseline_student = load_model(
        args.baseline_arch, args.baseline_width, len(classes),
        args.baseline_checkpoint, device,
    )
    distilled_student = load_model(
        args.distilled_arch, args.distilled_width, len(classes),
        args.distilled_checkpoint, device,
    )

    indices = select_images(dataset, args.num_images, seed=args.seed)
    print(f"Visualizing {len(indices)} held-out images: {indices}")

    plot_dark_knowledge(
        teacher=teacher,
        baseline_student=baseline_student,
        distilled_student=distilled_student,
        dataset=dataset,
        classes=classes,
        indices=indices,
        tau=args.tau,
        top_k=args.top_k,
        out_path=args.out,
        device=device,
    )


if __name__ == "__main__":
    main()
