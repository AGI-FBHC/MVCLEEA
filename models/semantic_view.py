import torch
import torch.nn as nn


class SemanticViewNetwork(nn.Module):
    """
    Semantic View Network (Z_E)

    Extracts semantic features from pre-trained ESM2 embeddings (1280-dim).
    Architecture: MLP with progressive dimensionality reduction.
    Input:  ESM2 feature vector (batch, 1280)
    Output: Semantic feature Z_E (batch, 128)
    """

    def __init__(self, input_dim: int = 1280, hidden_dims: list = None,
                 output_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256]

        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: ESM2 embeddings, shape (batch_size, 1280)
        Returns:
            Z_E: Semantic features, shape (batch_size, 128)
        """
        return self.network(x)
