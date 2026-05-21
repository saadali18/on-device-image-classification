import os
import time
import json
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from utils.metrics import topk_accuracy
from utils.checkpoint import save_checkpoint
from utils.losses import KDLoss


class DistillationTrainer:
    def __init__(
        self,
        student,
        teacher,
        train_loader,
        test_loader,
        config,
        device,
        num_parameters: int,
    ):
        self.student = student
        self.teacher = teacher
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config
        self.device = device
        self.num_parameters = num_parameters

        training_cfg = config["training"]
        scheduler_cfg = config["scheduler"]
        output_cfg = config["output"]
        distill_cfg = config["distillation"]

        self.alpha = float(distill_cfg["alpha"])
        self.temperature = float(distill_cfg["temperature"])

        # Freeze teacher: no grad, eval mode (disables dropout & freezes BN stats).
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

        self.criterion = KDLoss(alpha=self.alpha, temperature=self.temperature)
        self.test_criterion = nn.CrossEntropyLoss()

        self.optimizer = optim.SGD(
            self.student.parameters(),
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
        self.student.train()
        # Defensive: re-assert teacher.eval() in case anything upstream toggled it.
        self.teacher.eval()

        total_loss = 0.0
        total_loss_hard = 0.0
        total_loss_soft = 0.0
        total_top1 = 0.0
        total_top5 = 0.0
        total_samples = 0

        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            student_logits = self.student(images)
            with torch.no_grad():
                teacher_logits = self.teacher(images)

            loss, loss_hard, loss_soft = self.criterion(
                student_logits, teacher_logits, labels
            )
            loss.backward()
            self.optimizer.step()

            batch_size = labels.size(0)
            top1, top5 = topk_accuracy(
                student_logits.detach(), labels.detach(), topk=(1, 5)
            )

            total_loss += loss.item() * batch_size
            total_loss_hard += loss_hard.item() * batch_size
            total_loss_soft += loss_soft.item() * batch_size
            total_top1 += top1 * batch_size
            total_top5 += top5 * batch_size
            total_samples += batch_size

            if epoch == 1 and batch_idx == 0:
                print("-" * 80)
                print("FIRST DISTILLATION TRAIN BATCH DEBUG")
                print("-" * 80)
                print(f"alpha={self.alpha}  temperature={self.temperature}")
                print(f"Total loss: {loss.item():.4f}")
                print(f"  Hard CE : {loss_hard.item():.4f}")
                print(f"  Soft KL : {loss_soft.item():.4f}  (x T^2 = {self.temperature**2:.1f})")
                print(f"Student top-1: {top1:.2f}%")
                print(
                    "Teacher top-1: "
                    f"{topk_accuracy(teacher_logits, labels.detach(), topk=(1,))[0]:.2f}%"
                )
                print("-" * 80)

        return (
            total_loss / total_samples,
            total_loss_hard / total_samples,
            total_loss_soft / total_samples,
            total_top1 / total_samples,
            total_top5 / total_samples,
        )

    @torch.no_grad()
    def evaluate(self):
        self.student.eval()

        total_loss = 0.0
        total_top1 = 0.0
        total_top5 = 0.0
        total_samples = 0

        for images, labels in self.test_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            logits = self.student(images)
            loss = self.test_criterion(logits, labels)

            batch_size = labels.size(0)
            top1, top5 = topk_accuracy(logits, labels, topk=(1, 5))

            total_loss += loss.item() * batch_size
            total_top1 += top1 * batch_size
            total_top5 += top5 * batch_size
            total_samples += batch_size

        return (
            total_loss / total_samples,
            total_top1 / total_samples,
            total_top5 / total_samples,
        )

    def fit(self):
        print("=" * 80)
        print("Starting distillation training")
        print(f"Student: {self.config['model']['name']}  ({self.num_parameters:,} params)")
        print(f"Teacher: {self.config['distillation'].get('teacher_model')}")
        print(f"alpha (hard weight) = {self.alpha}")
        print(f"temperature         = {self.temperature}")
        print(f"Device: {self.device}")
        print("=" * 80)

        for epoch in range(1, self.epochs + 1):
            start_time = time.time()

            (
                train_loss,
                train_loss_hard,
                train_loss_soft,
                train_top1,
                train_top5,
            ) = self.train_one_epoch(epoch)
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
                    model=self.student,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch,
                    best_acc=self.best_acc,
                )

            row = {
                "epoch": epoch,
                "lr": current_lr,
                "train_loss": train_loss,
                "train_loss_hard": train_loss_hard,
                "train_loss_soft": train_loss_soft,
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
                f"Train L={train_loss:.4f} "
                f"(H={train_loss_hard:.4f} S={train_loss_soft:.4f}) "
                f"T1={train_top1:.2f}% | "
                f"Test L={test_loss:.4f} T1={test_top1:.2f}% T5={test_top5:.2f}% | "
                f"Best: {self.best_acc:.2f}% | "
                f"{elapsed:.1f}s"
            )

        self.save_summary()

        print("=" * 80)
        print("Distillation training complete")
        print(f"Best Test Top-1: {self.best_acc:.2f}%  (epoch {self.best_epoch})")
        print("=" * 80)

    def save_metrics(self):
        metrics_path = os.path.join(self.result_dir, "metrics.csv")
        df = pd.DataFrame(self.history)
        df.to_csv(metrics_path, index=False)

    def save_summary(self):
        distill_cfg = self.config["distillation"]
        summary = {
            "experiment_name": self.config["experiment_name"],
            "model": self.config["model"]["name"],
            "width": self.config["model"]["width"],
            "num_parameters": self.num_parameters,
            "alpha": self.alpha,
            "temperature": self.temperature,
            "teacher_model": distill_cfg.get("teacher_model"),
            "teacher_checkpoint": distill_cfg.get("teacher_checkpoint"),
            "best_test_top1": self.best_acc,
            "best_epoch": self.best_epoch,
            "epochs": self.epochs,
            "loss": "kd_hinton",
            "training_type": "distillation",
        }

        summary_path = os.path.join(self.result_dir, "summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=4)
