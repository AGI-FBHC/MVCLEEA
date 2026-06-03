import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CrossAttentionPooling(nn.Module):
    """
    Cross-Attention Pooling Module

    Produces a unified Shared View (Z_S) from multi-view features via
    cross-attention, where a learnable Global Token serves as the Query (Q)
    and the stacked multi-view features serve as Keys (K) and Values (V).

    Input:  Z_E, Z_P, Z_O — each (batch, 128)
    Output: Z_S — (batch, 128)

    Mechanism:
        Multi-view features are stacked into a sequence of 3 tokens (K, V).
        A learnable Global Token is used as the single Query (Q).
        Cross-attention output is projected to produce Z_S.
    """

    def __init__(self, feature_dim: int = 128, num_heads: int = 4):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        assert feature_dim % num_heads == 0, "feature_dim must be divisible by num_heads"

        # Learnable Global Token (Query)
        self.global_token = nn.Parameter(torch.randn(1, 1, feature_dim))

        # Q, K, V projection matrices
        self.W_q = nn.Linear(feature_dim, feature_dim)
        self.W_k = nn.Linear(feature_dim, feature_dim)
        self.W_v = nn.Linear(feature_dim, feature_dim)

        # Output projection
        self.W_o = nn.Linear(feature_dim, feature_dim)
        self.layer_norm = nn.LayerNorm(feature_dim)

    def forward(self, Z_E: torch.Tensor, Z_P: torch.Tensor,
                Z_O: torch.Tensor) -> torch.Tensor:
        """
        Args:
            Z_E: Semantic features,    (batch_size, 128)
            Z_P: Evolutionary features, (batch_size, 128)
            Z_O: Syntactic features,    (batch_size, 128)
        Returns:
            Z_S: Shared view features,  (batch_size, 128)
        """
        batch_size = Z_E.size(0)

        # Stack multi-view features: (batch, 3, 128)
        multi_view = torch.stack([Z_E, Z_P, Z_O], dim=1)

        # Expand global token for batch: (batch, 1, 128)
        Q = self.global_token.expand(batch_size, -1, -1)

        # Project Q, K, V
        Q = self.W_q(Q)  # (batch, 1, 128)
        K = self.W_k(multi_view)  # (batch, 3, 128)
        V = self.W_v(multi_view)  # (batch, 3, 128)

        # Reshape for multi-head attention
        Q = Q.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, 3, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, 3, self.num_heads, self.head_dim).transpose(1, 2)
        # Q: (batch, heads, 1, head_dim)
        # K: (batch, heads, 3, head_dim)
        # V: (batch, heads, 3, head_dim)

        # Scaled dot-product attention
        scale = math.sqrt(self.head_dim)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / scale
        # (batch, heads, 1, 3)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)
        # (batch, heads, 1, head_dim)

        # Concatenate heads: (batch, 1, 128)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, 1, self.feature_dim
        )

        # Output projection + residual + LayerNorm
        Z_S = self.W_o(attn_output)  # (batch, 1, 128)
        Z_S = self.layer_norm(Z_S + Q.transpose(1, 2).contiguous().view(
            batch_size, 1, self.feature_dim
        ))

        # Remove sequence dimension: (batch, 128)
        Z_S = Z_S.squeeze(1)

        return Z_S
