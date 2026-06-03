import torch
import torch.nn as nn


class SyntacticViewNetwork(nn.Module):
    """
    Syntactic View Network (Z_O)

    Extracts syntactic features from One-hot encoded protein sequences.
    Architecture: Two-layer CNN with BatchNorm, followed by FC layers.

    Input:  One-hot matrix (batch, 1, 1024, 21) — 20 amino acids + 'X'
    Output: Syntactic feature Z_O ( batch, 128)

    Dimension flow:
        Input:        1 x 1024 x 21
        Conv1(11x11): 16 x 1014 x 11
        BatchNorm:    16 x 1014 x 11
        Conv2(11x11): 16 x 1004 x 1
        Flatten:      16064
        FC1:          512
        FC2:          128
    """

    def __init__(self, output_dim: int = 128):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=11, stride=1),   # -> 16 x 1014 x 11
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=11, stride=1),  # -> 16 x 1004 x 1
            nn.ReLU(inplace=True),
        )

        self.flatten_dim = 16 * 1004 * 1  # 16064

        self.classifier = nn.Sequential(
            nn.Linear(self.flatten_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, output_dim),
        )

    def forward(self, onehot: torch.Tensor) -> torch.Tensor:
        """
        Args:
            onehot: One-hot encoded sequence matrix,
                    shape (batch_size, 1, 1024, 21)
        Returns:
            Z_O: Syntactic features, shape (batch_size, 128)
        """
        x = self.features(onehot)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
