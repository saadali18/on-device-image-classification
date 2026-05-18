"""
Train ResNet-110 teacher on CIFAR-100.

Outputs:
    resnet110_best.pth      <- best checkpoint by Test Top-1
    teacher_train_log.csv   <- per-epoch metrics
"""

import os, csv, time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from src.resnet_cifar import resnet110
from src.utils import train_one_epoch, evaluate

EPOCHS = 100
BATCH_SIZE = 128
LR = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
LR_MILESTONES = [60, 80]
LR_GAMMA = 0.1


def get_loaders():
    mean = (0.5071, 0.4867, 0.4408) # (RGB)
    std = (0.2675, 0.2565, 0.2761)

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    train_set = datasets.CIFAR100(root="./data", train=True, download=True, transform=train_transform)
    test_set = datasets.CIFAR100(root="./data", train=False, download=True, transform=test_transform)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    
    return train_loader, test_loader


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    train_loader, test_loader = get_loaders()

    model = resnet110(num_classes=100).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"ResNet-110 parameters: {n_params / 1e6:.3f}M")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=LR,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
        nesterov=True,
    )

    scheduler = MultiStepLR(optimizer, milestones=LR_MILESTONES, gamma=LR_GAMMA)

    best_top1 = 0.0
    log_fields = [
        "epoch",
        "train_loss",
        "train_top1",
        "train_top5",
        "test_loss",
        "test_top1",
        "test_top5",
    ]

    with open("teacher_train_log.csv", "w", newline="") as f:
        csv.DictWriter(f, fieldnames=log_fields).writeheader()

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        tr_loss, tr1, tr5 = train_one_epoch(model, train_loader, criterion, optimizer, device)
        te_loss, te1, te5 = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:3d}/{EPOCHS} | "
            f"Train  L={tr_loss:.4f}  Top1={tr1:.2f}  Top5={tr5:.2f} | "
            f"Test   L={te_loss:.4f}  Top1={te1:.2f}  Top5={te5:.2f} | "
            f"{elapsed:.0f}s"
        )

        with open("teacher_train_log.csv", "a", newline="") as f:
            csv.DictWriter(f, fieldnames=log_fields).writerow(
                dict(
                    epoch=epoch,
                    train_loss=round(tr_loss, 4),
                    train_top1=round(tr1, 2),
                    train_top5=round(tr5, 2),
                    test_loss=round(te_loss, 4),
                    test_top1=round(te1, 2),
                    test_top5=round(te5, 2),
                )
            )

        if te1 > best_top1:
            best_top1 = te1
            torch.save(
                {   
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "test_top1": te1,
                    "test_top5": te5,
                },
                "resnet110_best.pth",
            )
            print(f"-> New best: {best_top1:.2f}%  (saved resnet110_best.pth)")

    print(f"\nDone. Best Test Top-1: {best_top1:.2f}%")


if __name__ == "__main__":
    main()
