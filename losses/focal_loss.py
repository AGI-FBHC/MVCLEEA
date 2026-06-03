import torch
import torch.nn as nn
import torch.nn.functional as F
from .contrastive_loss import CrossViewContrastiveLoss


class FocalLoss(nn.Module):
    """
    Focal Loss for multi-label classification with extreme class imbalance.

    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)

    Args:
        gamma: Focusing parameter (default: 3.0)
        alpha: Class balance weight (default: None)
        pos_weight: Weight for positive samples in BCE
    """

    def __init__(self, gamma: float = 3.0, alpha: float = None, pos_weight: float = 1.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.pos_weight = pos_weight

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # pos_weight for BCE
        pw = torch.full_like(inputs[0:1, :], self.pos_weight)
        ce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, pos_weight=pw, reduction='none'
        )

        p = torch.sigmoid(inputs)
        p_t = p * targets + (1 - p) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_loss = alpha_t * focal_weight * ce_loss
        else:
            focal_loss = focal_weight * ce_loss

        return focal_loss.mean()


class JointFocalLoss(nn.Module):
    """
    Joint Loss with Focal Loss for classification + Contrastive Loss for views.

    L = (1 - lambda) * L_Focal + lambda * L_Contrastive

    Args:
        num_classes: Number of EC classes
        lambda_val: Balance between focal and contrastive (default: 0.02)
        temperature: Temperature for contrastive loss (default: 0.07)
        gamma: Focal loss gamma (default: 3.0)
        pos_weight: Positive sample weight for focal loss (default: 1.0)
        label_smoothing: Label smoothing factor (default: 0.05)
    """

    def __init__(self, num_classes: int = 588, lambda_val: float = 0.02,
                 temperature: float = 0.07, gamma: float = 3.0,
                 pos_weight: float = 1.0, label_smoothing: float = 0.05):
        super().__init__()
        self.lambda_val = lambda_val
        self.label_smoothing = label_smoothing
        self.focal_loss = FocalLoss(gamma=gamma, alpha=0.99, pos_weight=pos_weight)
        self.contrastive_loss = CrossViewContrastiveLoss(temperature=temperature)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                Z_E: torch.Tensor, Z_P: torch.Tensor,
                Z_O: torch.Tensor, Z_S: torch.Tensor):
        # Label smoothing
        if self.label_smoothing > 0:
            targets_smooth = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        else:
            targets_smooth = targets

        loss_focal = self.focal_loss(logits, targets_smooth)

        L_ES, L_PS, L_OS = self.contrastive_loss(Z_E, Z_P, Z_O, Z_S)
        contrastive_term = (L_ES + L_PS + L_OS) / 3.0

        total_loss = (1 - self.lambda_val) * loss_focal + self.lambda_val * contrastive_term

        return total_loss, loss_focal, L_ES, L_PS, L_OS
