import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossViewContrastiveLoss(nn.Module):
    """
    Cross-View Contrastive Loss (InfoNCE-style)

    Computes three contrastive losses between each specific view feature
    and the shared view feature:
        L_{E-S}: Semantic vs Shared
        L_{P-S}: Evolutionary vs Shared
        L_{O-S}: Syntactic vs Shared

    For each pair, the loss pulls together the anchor (specific view) and
    positive (shared view) from the same sample, while pushing apart
    features from different samples in the batch.

    Loss formula (for a single pair type):
        L = -log( exp(sim(a, p+) / tau) / sum_j(exp(sim(a, p_j) / tau)) )

    where the sum is over all samples in the batch (including the positive).
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def _infonce_loss(self, anchor: torch.Tensor,
                      positive: torch.Tensor) -> torch.Tensor:
        """
        Compute InfoNCE loss for a single anchor-positive pair type.

        Args:
            anchor:   (batch, dim) — features from a specific view
            positive: (batch, dim) — features from the shared view
        Returns:
            Scalar loss value
        """
        # L2 normalize features
        anchor = F.normalize(anchor, dim=1)
        positive = F.normalize(positive, dim=1)

        # Cosine similarity matrix: (batch, batch)
        sim_matrix = torch.matmul(anchor, positive.T) / self.temperature

        # Labels: diagonal entries are positive pairs
        batch_size = anchor.size(0)
        labels = torch.arange(batch_size, device=anchor.device)

        # Cross-entropy over similarity matrix
        loss = F.cross_entropy(sim_matrix, labels)

        return loss

    def forward(self, Z_E: torch.Tensor, Z_P: torch.Tensor,
                Z_O: torch.Tensor, Z_S: torch.Tensor):
        """
        Compute all three cross-view contrastive losses.

        Args:
            Z_E: Semantic features,     (batch, dim)
            Z_P: Evolutionary features,  (batch, dim)
            Z_O: Syntactic features,     (batch, dim)
            Z_S: Shared view features,   (batch, dim)

        Returns:
            L_ES: Semantic-Shared contrastive loss
            L_PS: Evolutionary-Shared contrastive loss
            L_OS: Syntactic-Shared contrastive loss
        """
        L_ES = self._infonce_loss(Z_E, Z_S)
        L_PS = self._infonce_loss(Z_P, Z_S)
        L_OS = self._infonce_loss(Z_O, Z_S)

        return L_ES, L_PS, L_OS
