import torch
import torch.nn as nn
import torch.nn.functional as F


class KDLoss(nn.Module):
    """
    Hinton et al. (2015) knowledge distillation loss, per the project's
    submitted report Eq. (11):

        L_KD = alpha * L_hard + (1 - alpha) * (T ** 2) * L_soft

    where
        L_hard = CE(softmax(z_s), y)                       (one-hot labels, T=1)
        L_soft = KL( softmax(z_t / T) || softmax(z_s / T) ) (softened distributions)

    Note: `alpha` is the HARD-label weight. Several KD references (including
    the reference final report from the parallel group) use alpha as the SOFT
    weight; the conventions are opposite. We follow the convention pinned by
    our own mid-report so the experiments match the equations on paper.
    """

    def __init__(self, alpha: float, temperature: float):
        super().__init__()
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")

        self.alpha = alpha
        self.temperature = temperature
        self.hard_criterion = nn.CrossEntropyLoss()

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
    ):
        loss_hard = self.hard_criterion(student_logits, labels)

        T = self.temperature
        # KL divergence requires log-probs on the input side and probs on the target side.
        # reduction='batchmean' divides by batch size (mathematically correct KL).
        student_log_probs = F.log_softmax(student_logits / T, dim=1)
        teacher_probs = F.softmax(teacher_logits / T, dim=1)
        loss_soft = F.kl_div(
            student_log_probs,
            teacher_probs,
            reduction="batchmean",
        )

        loss = self.alpha * loss_hard + (1.0 - self.alpha) * (T ** 2) * loss_soft

        return loss, loss_hard.detach(), loss_soft.detach()
