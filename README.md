# Knowledge Distillation for Efficient On-Device Image Classification

Deploying deep learning models on resource-constrained devices — mobile phones,
embedded sensors, edge nodes — requires models that are **accurate, fast, and small**.
Knowledge distillation offers an elegant solution: train a compact *student* network
to mimic the output distribution (soft labels) of a large pretrained *teacher* network,
recovering much of the teacher's accuracy at a fraction of the computational cost.

This project implements the **Hinton et al. distillation objective** — a weighted
combination of hard-label cross-entropy and KL-divergence against softened teacher
logits — and conducts a systematic study of:

- the effect of the **temperature** hyperparameter on soft-label sharpness
- the **accuracy gap** between teacher and students of varying depth / width
- the role of **"dark knowledge"** (soft probabilities over incorrect classes)

---

## Repository Layout

```
on-device-image-classification/
├── src/                        # All importable Python source modules
│   ├── resnet_cifar.py         # ResNet-{20,32,44,56,110} CIFAR architectures
│   ├── train_teacher.py        # End-to-end teacher training script
│   └── utils.py                # Shared helpers (train_one_epoch, evaluate, topk_accuracy)
│
├── notebooks/                  # Exploratory / interactive Jupyter notebooks
│   └── 01_train_teacher.ipynb  # Notebook version of the teacher training pipeline
│
├── weights/                    # Saved model checkpoints (not committed to git by default)
│   └── teacher/
│       └── resnet110_best.pth  # Best teacher checkpoint (by Test Top-1)
│
├── logs/                       # Per-epoch CSV logs produced during training
│   └── teacher_train_log.csv   # Loss + Top-1/Top-5 metrics for the teacher run
│
├── figs/                       # Generated plots and figures
│   └── teacher_training_curve.png
│
├── paper/                      # ICML-formatted LaTeX report
│   ├── report.tex
│   ├── references.bib
│   ├── icml2025.sty / .bst
│   ├── fancyhdr.sty
│   └── imgs/                   # Figures embedded in the paper
│
└── README.md
```

---

## Where to Put Each New Script

Use the table below as the **canonical placement guide** whenever you add a new file.

| What you are adding | Correct location | Naming convention |
|---|---|---|
| Distillation training loop | `src/` | `train_student.py` (or `train_distill_<variant>.py`) |
| Shared helpers / loss functions | `src/` | add to `utils.py` **or** create `losses.py` |
| One-off exploration / analysis | `notebooks/` | `NN_<short_description>.ipynb` (numbered in order) |
| Saved model checkpoint | `weights/<stage>/` | `<arch>_best.pth` / `<arch>_epoch<N>.pth` |
| Training log (CSV) | `logs/` | `<stage>_train_log.csv` e.g. `student_train_log.csv` |
| Plot / figure output | `figs/` | `<descriptive_name>.png` |
| LaTeX source / paper edits | `paper/` | keep all `.tex`, `.bib`, `.sty` here |
| Figures used *inside* the paper | `paper/imgs/` | match filename to `\includegraphics{}` call |

---


## Project Roadmap

- [x] Student baseline: ResNet-20 trained on CIFAR-100 (hard labels only)
- [x] Teacher: ResNet-110 trained on CIFAR-100
- [ ] Implement Hinton et al. distillation loss (KL divergence + hard-label CE + τ² scaling)
- [ ] Step 2a — Temperature ablation: fix α=0.5, sweep τ ∈ {2, 4, 8, 16}
- [ ] Step 2b — Weighting ablation: fix τ=τ*, sweep α ∈ {0.1, 0.5, 0.9}
- [ ] Step 3a — Depth ablation: ResNet-{20, 32, 44, 56} at optimal (τ*, α*)
- [ ] Step 3b — Width ablation: ResNet-20 at w ∈ {0.5, 1, 2} at optimal (τ*, α*)
- [ ] Step 4 — Dark knowledge analysis: plot soft probability distributions (teacher vs. distilled student vs. hard-label student)
- [ ] Final report write-up (ICML format, 4–6 pages)

---


## Models (`src/resnet_cifar.py`)

All teacher and student architectures are defined in a single file using the
He et al. CIFAR ResNet family: total layers = 6n + 2, with n blocks per stage.

### Currently defined

| Function | n | Layers | Params | Use |
|---|---|---|---|---|
| `resnet110(num_classes=100)` | 18 | 110 | ~1.73M | Teacher |

### Adding student variants (Step 2 onwards)

Add the following constructor functions to the bottom of `resnet_cifar.py`:

```python
def resnet20(num_classes=100):
    """ResNet-20: n=3, ~0.27M params. Primary student."""
    return ResNet(BasicBlock, [3, 3, 3], num_classes=num_classes)

def resnet32(num_classes=100):
    """ResNet-32: n=5, ~0.46M params. Depth ablation."""
    return ResNet(BasicBlock, [5, 5, 5], num_classes=num_classes)

def resnet44(num_classes=100):
    """ResNet-44: n=7, ~0.66M params. Depth ablation."""
    return ResNet(BasicBlock, [7, 7, 7], num_classes=num_classes)

def resnet56(num_classes=100):
    """ResNet-56: n=9, ~0.85M params. Depth ablation."""
    return ResNet(BasicBlock, [9, 9, 9], num_classes=num_classes)
```

### Width variants (Step 3 — width ablation)

The `ResNet` class does not yet support a width multiplier. Add a `width`
parameter to the constructor:

```python
# Change the ResNet.__init__ signature from:
def __init__(self, block, num_blocks, num_classes=100):
    ...
    self.in_planes = 16
    self.conv1 = nn.Conv2d(3, 16, ...)
    self.layer1 = self._make_layer(block, 16,  num_blocks[0], stride=1)
    self.layer2 = self._make_layer(block, 32,  num_blocks[1], stride=2)
    self.layer3 = self._make_layer(block, 64,  num_blocks[2], stride=2)
    self.fc = nn.Linear(64, num_classes)

# To:
def __init__(self, block, num_blocks, num_classes=100, width=1):
    ...
    base = max(1, int(16 * width))
    self.in_planes = base
    self.conv1 = nn.Conv2d(3, base, ...)
    self.layer1 = self._make_layer(block, base,     num_blocks[0], stride=1)
    self.layer2 = self._make_layer(block, base * 2, num_blocks[1], stride=2)
    self.layer3 = self._make_layer(block, base * 4, num_blocks[2], stride=2)
    self.fc = nn.Linear(base * 4, num_classes)
```

Then update the constructors to pass `width` through:

```python
def resnet20(num_classes=100, width=1):
    return ResNet(BasicBlock, [3, 3, 3], num_classes=num_classes, width=width)
```

Width multiplier reference:

| `width` | Base channels | Approx params (ResNet-20) |
|---|---|---|
| 0.5 | 8 | ~0.07M |
| 1 | 16 | ~0.27M |
| 2 | 32 | ~1.07M |

---

## References

- Hinton, G., Vinyals, O., & Dean, J. (2015). *Distilling the Knowledge in a Neural Network.* NIPS Deep Learning Workshop.
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for Image Recognition.* CVPR.
