"""Visual projector — unifies encoder output dimensionality to dim_model.

All encoders project through this module so that the ACT transformer head
sees the same input dimension regardless of which encoder produced z_vis.
This is critical for fair comparison — without it, observed differences
could be confounded with encoder capacity.

Two input formats handled:
- (B, D) — single token vector (some encoders, configurable)
- (B, N, D) — sequence of N spatial tokens (standard for ACT-style heads)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class VisualProjector(nn.Module):
    """MLP projector mapping encoder features to ACT dim_model.

    If `num_spatial_tokens > 1`, expects input shape (B, N, input_dim)
    and applies projection per token.

    If `num_spatial_tokens == 1`, expects input shape (B, input_dim)
    and outputs (B, 1, output_dim).
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 256,
        num_spatial_tokens: int = 1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_spatial_tokens = num_spatial_tokens

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, input_dim) if num_spatial_tokens == 1
               (B, N, input_dim) if num_spatial_tokens > 1
        Returns:
            (B, N, output_dim) — always with sequence dim, N>=1
        """
        if z.dim() == 2:
            # (B, D) -> (B, 1, D) -> project -> (B, 1, output_dim)
            z = z.unsqueeze(1)

        return self.mlp(z)
