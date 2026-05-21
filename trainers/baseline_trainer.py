import os
import time
import json
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from utils.metrics import topk_accuracy
from utils.checkpoint import save_checkpoint


class BaselineTrainer:
    def __init__(
        self,
        model,
        train_loader,
        test_loader,
        config,
        device,
        num_parameters: int,
    ):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config
        self.device = device
        self.num_parameters = num_parameters

        training_cfg = config["training"]
        scheduler_cfg = config["scheduler"]
        output_cfg = config["output"]

        self.criterion = nn.CrossEntropyLoss()

        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=training_cfg["learning_rate"],
            momentum=training_cfg["momentum"],
            weight_decay=training_cfg["weight_decay"],
        )

        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=scheduler_cfg["milestones"],
            gamma=scheduler_cfg["gamma"],
        )

        self.epochs = training_cfg["epochs"]

        self.checkpoint_dir = output_cfg["checkpoint_dir"]
        self.result_dir = output_cfg["result_dir"]

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.result_dir, exist_ok=True)

        self.history = []
        self.best_acc = 0.0
        self.best_epoch = 0

    def train_one_epoch(self, epoch: int):
        self.model.train()

        total_loss = 0.0
        total_top1 = 0.0
        total_top5 = 0.0
        total_samples = 0

        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            logits = self.model(images)

            if logits.shape[1] != self.config["dataset"]["num_classes"]:
                raise ValueError(
                    f"Expected logits with {self.config['dataset']['num_classes']} classes, "
                    f"got {logits.shape}"
                )

            loss = self.criterion(logits, labels)

            loss.backward()
            self.optimizer.step()

            batch_size = labels.size(0)

            top1, top5 = topk_accuracy(logits.detach(), labels.detach(), topk=(1, 5))

            total_loss += loss.item() * batch_size
            total_top1 += top1 * batch_size
            total_top5 += top5 * batch_size
            total_samples += batch_size

        avg_loss = total_loss / total_samples
        avg_top1 = total_top1 / total_samples
        avg_top5 = total_top5 / total_samples

        return avg_loss, avg_top1, avg_top5

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()

        total_loss = 0.0
        total_top1 = 0.0
        total_top5 = 0.0
        total_samples = 0

        for batch_idx, (images, labels) in enumerate(self.test_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)

            logits = self.model(images)

            if logits.shape[1] != self.config["dataset"]["num_classes"]:
                raise ValueError(
                    f"Expected logits with {self.config['dataset']['num_classes']} classes, "
                    f"got {logits.shape}"
                )

            loss = self.criterion(logits, labels)

            batch_size = labels.size(0)

            top1, top5 = topk_accuracy(logits, labels, topk=(1, 5))

            total_loss += loss.item() * batch_size
            total_top1 += top1 * batch_size
            total_top5 += top5 * batch_size
            total_samples += batch_size

        avg_loss = total_loss / total_samples
        avg_top1 = total_top1 / total_samples
        avg_top5 = total_top5 / total_samples

        return avg_loss, avg_top1, avg_top5

    def fit(self):
        print("=" * 80)
        print("Starting baseline training")
        print(f"Model parameters: {self.num_parameters:,}")
        print(f"Device: {self.device}")
        print("=" * 80)

        for epoch in range(1, self.epochs + 1):
            start_time = time.time()

            train_loss, train_top1, train_top5 = self.train_one_epoch(epoch)
            test_loss, test_top1, test_top5 = self.evaluate()

            self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start_time

            is_best = test_top1 > self.best_acc

            if is_best:
                self.best_acc = test_top1
                self.best_epoch = epoch

                save_checkpoint(
                    checkpoint_dir=self.checkpoint_dir,
                    filename="best.pth",
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch,
                    best_acc=self.best_acc,
                )

            row = {
                "epoch": epoch,
                "lr": current_lr,
                "train_loss": train_loss,
                "train_top1": train_top1,
                "train_top5": train_top5,
                "test_loss": test_loss,
                "test_top1": test_top1,
                "test_top5": test_top5,
                "best_test_top1": self.best_acc,
                "epoch_time_sec": elapsed,
            }

            self.history.append(row)
            self.save_metrics()

            print(
                f"Epoch [{epoch:03d}/{self.epochs}] "
                f"LR: {current_lr:.5f} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Train Top-1: {train_top1:.2f}% | "
                f"Train Top-5: {train_top5:.2f}% | "
                f"Test Loss: {test_loss:.4f} | "
                f"Test Top-1: {test_top1:.2f}% | "
                f"Test Top-5: {test_top5:.2f}% | "
                f"Best: {self.best_acc:.2f}% | "
                f"Time: {elapsed:.1f}s"
            )

        self.save_summary()

        print("=" * 80)
        print("Training complete")
        print(f"Best Test Top-1 Accuracy: {self.best_acc:.2f}%")
        print(f"Best Epoch: {self.best_epoch}")
        print("=" * 80)

    def save_metrics(self):
        metrics_path = os.path.join(self.result_dir, "metrics.csv")
        df = pd.DataFrame(self.history)
        df.to_csv(metrics_path, index=False)

    def save_summary(self):
        summary = {
            "experiment_name": self.config["experiment_name"],
            "model": self.config["model"]["name"],
            "width": self.config["model"]["width"],
            "num_parameters": self.num_parameters,
            "best_test_top1": self.best_acc,
            "best_epoch": self.best_epoch,
            "epochs": self.epochs,
            "loss": "cross_entropy",
            "training_type": "hard_labels_only",
        }

        summary_path = os.path.join(self.result_dir, "summary.json")

        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=4)