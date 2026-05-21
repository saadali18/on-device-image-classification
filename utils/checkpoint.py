import os
import torch


def save_checkpoint(
    checkpoint_dir: str,
    filename: str,
    model,
    optimizer,
    scheduler,
    epoch: int,
    best_acc: float,
) -> None:
    os.makedirs(checkpoint_dir, exist_ok=True)

    path = os.path.join(checkpoint_dir, filename)

    torch.save(
        {
            "epoch": epoch,
            "best_acc": best_acc,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        },
        path,
    )


def load_checkpoint(path: str, model, optimizer=None, scheduler=None, device="cpu"):
    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint