import torch
import torch.nn as nn


class HierarchicalClassifier(nn.Module):
    """
    Hierarchical Classifier for EC Number Prediction

    MLP-based classifier that maps the fused feature Z_F to C class
    logits (raw scores before Sigmoid).

    Input:  Z_F (batch, 128)
    Output: Logits (batch, C) — apply Sigmoid externally for probabilities
    """

    def __init__(self, input_dim: int = 128, hidden_dim: int = 256,
                 num_classes: int = 588):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, Z_F: torch.Tensor) -> torch.Tensor:
        """
        Args:
            Z_F: Fused feature vector, (batch_size, 128)
        Returns:
            logits: Raw logits, (batch_size, num_classes)
        """
        return self.classifier(Z_F)
