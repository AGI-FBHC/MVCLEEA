import torch
import torch.nn as nn


class AdaptiveFeatureFusion(nn.Module):
    """
    Adaptive Feature Fusion Module

    Computes a weighted combination of four feature streams using learnable
    parameters (alpha_1, alpha_2, alpha_3, alpha_4):

        Z_F = alpha_1 * Z_E + alpha_2 * Z_P + alpha_3 * Z_O + alpha_4 * Z_S

    The alphas are normalized via Softmax to ensure they sum to 1.
    """

    def __init__(self, num_views: int = 4):
        super().__init__()
        # Learnable fusion weights, initialized uniformly
        self.alpha = nn.Parameter(torch.ones(num_views) / num_views)

    def forward(self, Z_E: torch.Tensor, Z_P: torch.Tensor,
                Z_O: torch.Tensor, Z_S: torch.Tensor) -> torch.Tensor:
        """
        Args:
            Z_E: Semantic features,     (batch_size, 128)
            Z_P: Evolutionary features,  (batch_size, 128)
            Z_O: Syntactic features,     (batch_size, 128)
            Z_S: Shared view features,   (batch_size, 128)
        Returns:
            Z_F: Fused feature vector,   (batch_size, 128)
        """
        # Normalize weights via Softmax
        weights = torch.softmax(self.alpha, dim=0)

        # Weighted sum
        Z_F = (weights[0] * Z_E + weights[1] * Z_P +
               weights[2] * Z_O + weights[3] * Z_S)

        return Z_F
