import torch
import torch.nn as nn
import torch.nn.functional as F
from .contrastive_loss import CrossViewContrastiveLoss


class JointLoss(nn.Module):
    """
    Joint Loss Function for MVCLEEA

    Combines binary cross-entropy (BCE) classification loss with
    cross-view contrastive losses:

        L = (1 - lambda) * L_BCE + (lambda / 3) * (L_{E-S} + L_{P-S} + L_{O-S})

    Args:
        num_classes: Number of EC classes (C)
        lambda_val:  Weight balancing hyperparameter (default: 0.15)
        temperature: Temperature for contrastive loss (default: 0.07)
        pos_weight:  Weight for positive samples (default: computed from data)
    """

    def __init__(self, num_classes: int = 588, lambda_val: float = 0.15,
                 temperature: float = 0.07, pos_weight: float = 1.0,
                 label_smoothing: float = 0.05):
        super().__init__()
        self.lambda_val = lambda_val
        self.pos_weight_val = pos_weight
        self.label_smoothing = label_smoothing
        self.contrastive_loss = CrossViewContrastiveLoss(temperature=temperature)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                Z_E: torch.Tensor, Z_P: torch.Tensor,
                Z_O: torch.Tensor, Z_S: torch.Tensor):
        """
        Compute the joint loss.

        Args:
            logits:  Raw logits from classifier, (batch, C)
            targets: Ground truth labels (multi-hot), (batch, C)
            Z_E:     Semantic features, (batch, 128)
            Z_P:     Evolutionary features, (batch, 128)
            Z_O:     Syntactic features, (batch, 128)
            Z_S:     Shared view features, (batch, 128)

        Returns:
            total_loss:  Combined joint loss (scalar)
            loss_bce:    BCE classification loss (scalar)
            L_ES:        Semantic-Shared contrastive loss (scalar)
            L_PS:        Evolutionary-Shared contrastive loss (scalar)
            L_OS:        Syntactic-Shared contrastive loss (scalar)
        """
        # Label smoothing: 0 -> smooth, 1 -> 1-smooth
        if self.label_smoothing > 0:
            targets_smooth = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        else:
            targets_smooth = targets

        # Weighted BCE loss for handling class imbalance
        pos_weight = torch.full_like(logits[0:1, :], self.pos_weight_val)
        loss_bce = F.binary_cross_entropy_with_logits(
            logits, targets_smooth,
            pos_weight=pos_weight,
            reduction='mean'
        )

        # Contrastive losses
        L_ES, L_PS, L_OS = self.contrastive_loss(Z_E, Z_P, Z_O, Z_S)

        # Joint loss
        contrastive_term = (L_ES + L_PS + L_OS) / 3.0
        total_loss = (1 - self.lambda_val) * loss_bce + self.lambda_val * contrastive_term

        return total_loss, loss_bce, L_ES, L_PS, L_OS
