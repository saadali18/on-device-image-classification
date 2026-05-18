"""
Shared training utilities for the knowledge distillation project.
"""

import torch
from tqdm import tqdm


def topk_accuracy(outputs, labels, topk=(1, 5)):
    """
    Computes top-1 and top-5 accuracy for a single batch.

    Returns:
        List of accuracy percentages, one per k in topk.
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = labels.size(0)

        # Get the indices of the top-maxk predictions for each sample
        _, pred = outputs.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()  # (maxk, N)
        correct = pred.eq(labels.unsqueeze(0).expand_as(pred))  # (maxk, N)

        results = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum()
            results.append(100.0 * correct_k / batch_size)
        return results


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Runs one full training epoch.

    Returns:
        train_loss  (float) : mean loss over the epoch
        train_top1  (float) : top-1 accuracy %
        train_top5  (float) : top-5 accuracy %
    """
    model.train()

    running_loss = 0.0
    top1_sum = top5_sum = total = 0

    for inputs, labels in tqdm(loader, desc="Train", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        n = labels.size(0)
        running_loss += loss.item() * n
        t1, t5 = topk_accuracy(outputs, labels, topk=(1, 5))
        top1_sum += t1.item() * n
        top5_sum += t5.item() * n
        total += n

    return running_loss / total, top1_sum / total, top5_sum / total


def evaluate(model, loader, criterion, device):
    """
    Runs one full evaluation pass (no gradient computation).

    Returns:
        val_loss  (float) : mean loss over the dataset
        val_top1  (float) : top-1 accuracy %
        val_top5  (float) : top-5 accuracy %
    """
    model.eval()

    running_loss = 0.0
    top1_sum = top5_sum = total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Eval ", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            n = labels.size(0)
            running_loss += loss.item() * n
            t1, t5 = topk_accuracy(outputs, labels, topk=(1, 5))
            top1_sum += t1.item() * n
            top5_sum += t5.item() * n
            total += n

    return running_loss / total, top1_sum / total, top5_sum / total
