"""Self-contained neural network components used by MVCLEEA v2."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PLMViewEncoder(nn.Module):
    """MLP encoder for ESM-2 embeddings with a residual projection."""

    def __init__(self, input_dim, output_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )
        self.residual = (
            nn.Linear(input_dim, output_dim)
            if input_dim != output_dim
            else nn.Identity()
        )

    def forward(self, x):
        return self.net(x) + self.residual(x)


class CrossAttentionPooling(nn.Module):
    """Use a learnable global token to aggregate feature views."""

    def __init__(self, feature_dim=256, num_heads=8):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        self.global_token = nn.Parameter(torch.randn(1, 1, feature_dim))
        self.q_proj = nn.Linear(feature_dim, feature_dim)
        self.k_proj = nn.Linear(feature_dim, feature_dim)
        self.v_proj = nn.Linear(feature_dim, feature_dim)
        self.out_proj = nn.Linear(feature_dim, feature_dim)
        self.layer_norm = nn.LayerNorm(feature_dim)

    def forward(self, esm, pssm, onehot):
        return self.forward_views([esm, pssm, onehot])

    def forward_views(self, views):
        """Aggregate a variable number of feature views with the global token."""
        batch_size = views[0].size(0)
        stacked = torch.stack(views, dim=1)
        n_views = stacked.size(1)
        query = self.q_proj(
            self.global_token.expand(batch_size, -1, -1)
        )
        keys = self.k_proj(stacked)
        values = self.v_proj(stacked)

        query = query.view(
            batch_size, 1, self.num_heads, self.head_dim
        ).transpose(1, 2)
        keys = keys.view(
            batch_size, n_views, self.num_heads, self.head_dim
        ).transpose(1, 2)
        values = values.view(
            batch_size, n_views, self.num_heads, self.head_dim
        ).transpose(1, 2)

        attention = torch.matmul(
            query, keys.transpose(-2, -1)
        ) / (self.head_dim ** 0.5)
        attention = F.softmax(attention, dim=-1)
        output = torch.matmul(attention, values)
        output = output.transpose(1, 2).contiguous().view(
            batch_size, 1, self.feature_dim
        )
        output = self.out_proj(output.squeeze(1))
        pooled = self.layer_norm(output + query.transpose(1, 2).reshape(
            batch_size, 1, self.feature_dim
        ).squeeze(1))
        return pooled, attention.squeeze(2)


class ProjectionHead(nn.Module):
    """Map fused features into the contrastive embedding space."""

    def __init__(self, input_dim=256, hidden_dim=512, output_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


def get_ec_level(ec, level):
    """Return the requested prefix level of an EC number."""
    parts = ec.split('.')
    return '.'.join(parts[:level]) if level <= len(parts) else ec
