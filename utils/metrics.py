import torch


@torch.no_grad()
def topk_accuracy(logits: torch.Tensor, labels: torch.Tensor, topk=(1,)):
    """
    Computes top-k accuracy in percentage.

    logits: [batch_size, num_classes]
    labels: [batch_size]
    """
    if logits.ndim != 2:
        raise ValueError(f"logits must be 2D [B, C], got shape {logits.shape}")

    if labels.ndim != 1:
        raise ValueError(f"labels must be 1D [B], got shape {labels.shape}")

    batch_size = labels.size(0)
    max_k = max(topk)

    _, pred = logits.topk(max_k, dim=1, largest=True, sorted=True)
    pred = pred.t()  # [max_k, batch_size]

    correct = pred.eq(labels.view(1, -1).expand_as(pred))

    accuracies = []

    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum().item()
        acc = 100.0 * correct_k / batch_size
        accuracies.append(acc)

    return accuracies


@torch.no_grad()
def top1_accuracy(logits: torch.Tensor, labels: torch.Tensor):
    preds = logits.argmax(dim=1)
    correct = preds.eq(labels).sum().item()
    total = labels.size(0)
    return 100.0 * correct / total