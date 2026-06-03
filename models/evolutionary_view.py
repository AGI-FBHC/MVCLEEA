import torch
import torch.nn as nn


class EvolutionaryViewNetwork(nn.Module):
    """
    Evolutionary View Network (Z_P)

    Extracts evolutionary features from Position-Specific Scoring Matrix (PSSM).
    Architecture: Two-layer CNN with max pooling, followed by FC layers.

    Input:  PSSM matrix (batch, 1, 1024, 20) — treated as single-channel 2D image
    Output: Evolutionary feature Z_P (batch, 128)

    Dimension flow:
        Input:       1 x 1024 x 20
        Conv1(5x5):  16 x 1020 x 16
        MaxPool:     16 x 510 x 8
        Conv2(5x5):  32 x 506 x 4
        MaxPool:     32 x 253 x 2
        Flatten:     16192
        FC1:         512
        FC2:         128
    """

    def __init__(self, output_dim: int = 128):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1),   # -> 16 x 1020 x 16
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),        # -> 16 x 510 x 8
            nn.Conv2d(16, 32, kernel_size=5, stride=1),   # -> 32 x 506 x 4
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),        # -> 32 x 253 x 2
        )

        self.flatten_dim = 32 * 253 * 2  # 16192

        self.classifier = nn.Sequential(
            nn.Linear(self.flatten_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, output_dim),
        )

    def forward(self, pssm: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pssm: PSSM matrix with Sigmoid normalization applied,
                  shape (batch_size, 1, 1024, 20)
        Returns:
            Z_P: Evolutionary features, shape (batch_size, 128)
        """
        x = self.features(pssm)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
