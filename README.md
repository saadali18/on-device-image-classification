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
├── configs/                        # YAML hyperparameter configs (one per experiment)
│   ├── baseline_resnet20.yaml
│   └── teacher_resnet110.yaml
│
├── datasets/                       # Data loading
│   └── cifar100.py
│
├── models/                         # Model definitions
│   └── resnet_cifar.py             # ResNet-{20,32,56,110} + build_resnet_cifar() factory
│
├── trainers/                       # Training loop classes (one per training type)
│   ├── baseline_trainer.py         # Hard-label CE only
│   └── teacher_trainer.py          # Hard-label CE only (large model)
│
├── scripts/                        # Entry points — `python scripts/<name>.py`
│   ├── train_baseline.py
│   ├── train_teacher.py
│   ├── plot_baseline_metrics.py
│   └── plot_teacher_metrics.py
│
├── utils/                          # Shared helpers
│   ├── metrics.py                  # topk_accuracy
│   ├── checkpoint.py               # save_checkpoint / load_checkpoint
│   ├── seed.py                     # set_seed
│   └── logger.py
│
├── checkpoints/<stage>/<arch>/     # Trained model checkpoints
│   └── best.pth
│
├── results/<stage>/<arch>/         # Per-epoch CSV + summary JSON
│   ├── metrics.csv
│   └── summary.json
│
├── plots/<stage>/<arch>/           # Generated training curves
│
├── paper/                          # ICML-formatted LaTeX report
│   ├── report.tex
│   ├── references.bib
│   ├── icml2025.sty / .bst, fancyhdr.sty
│   └── imgs/
│
├── requirements.txt
└── README.md
```

---

## Reproducing Existing Results

```bash
pip install -r requirements.txt

# Train baseline ResNet-20 (hard labels only)
python scripts/train_baseline.py

# Train teacher ResNet-110 (hard labels only)
python scripts/train_teacher.py

# Plot curves
python scripts/plot_baseline_metrics.py
python scripts/plot_teacher_metrics.py
```

Hyperparameters are read from `configs/<experiment>.yaml`. Outputs go to
`checkpoints/`, `results/`, and `plots/` under matching subpaths.

---

## Current Results

| Stage | Model | Params | Test Top-1 | Test Top-5 | Best Epoch | Config |
|---|---|---|---|---|---|---|
| Baseline (hard labels) | ResNet-20 (w=1) | 278,324 | **67.63%** | 91.31% | 89 / 100 | `configs/baseline_resnet20.yaml` |
| Teacher (hard labels) | ResNet-110 (w=1) | 1,736,564 | **73.48%** | 92.56% | 129 / 160 | `configs/teacher_resnet110.yaml` |

> **Note on teacher config drift**: the report's Table 1 specifies 100 epochs / milestones [60, 80] for *all* runs.
> The committed teacher was actually trained for 160 epochs / milestones [80, 120].
> Decision deferred — either retrain teacher under the report's config or update Table 1 in the final report.

---

## Project Roadmap

- [x] Student baseline: ResNet-20 trained on CIFAR-100 (hard labels only)
- [x] Teacher: ResNet-110 trained on CIFAR-100
- [ ] Implement Hinton et al. distillation loss (KL divergence + hard-label CE + τ² scaling)
- [ ] Step 2a — Temperature ablation: fix α=0.5, sweep τ ∈ {2, 4, 8, 16}
- [ ] Step 2b — Weighting ablation: fix τ=τ\*, sweep α ∈ {0.1, 0.5, 0.9}
- [ ] Step 3a — Depth ablation: ResNet-{20, 32, 44, 56} at optimal (τ\*, α\*)
- [ ] Step 3b — Width ablation: ResNet-20 at w ∈ {0.5, 1, 2} at optimal (τ\*, α\*)
- [ ] Step 4 — Dark knowledge analysis: soft-probability distributions (teacher vs. distilled student vs. hard-label student)
- [ ] Final report write-up (ICML format, 4–6 pages)

---

## Adding a New Experiment

When implementing distillation (Step 2 onwards), follow the existing pattern:

| What you are adding | Where it goes |
|---|---|
| New trainer (e.g. `DistillationTrainer`) | `trainers/distillation_trainer.py` |
| New entry point | `scripts/train_distillation.py` |
| Hyperparameter config | `configs/distillation_<variant>.yaml` (e.g. `distillation_tau4_alpha0p5.yaml`) |
| Loss function | `utils/losses.py` (create if absent) |
| Checkpoint | `checkpoints/distillation/<variant>/best.pth` |
| Per-epoch CSV + summary | `results/distillation/<variant>/` |
| Curves | `plots/distillation/<variant>/` |

The `BaselineTrainer` class is the closest template — most of the distillation
trainer will reuse its `train_one_epoch` / `evaluate` / `fit` skeleton, with
the loss term swapped for the combined hard + soft objective and the frozen
teacher passed in at construction.

---

## References

- Hinton, G., Vinyals, O., & Dean, J. (2015). *Distilling the Knowledge in a Neural Network.* NIPS Deep Learning Workshop.
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for Image Recognition.* CVPR.
