"""Base interface for all visual prior encoders.

All encoders implement this interface so that VisualPriorACTPolicy can
treat them uniformly regardless of internal architecture.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class VisualPriorEncoder(nn.Module, ABC):
    """Common interface for all visual encoders.

    Subclasses must:
    - Set `self.output_dim` in __init__ (size of feature dimension)
    - Set `self.num_spatial_tokens` (1 for vector output, N for sequence)
    - Implement forward(images) -> features
    """

    output_dim: int  # required
    num_spatial_tokens: int = 1  # default: single token

    @abstractmethod
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (B, C, H, W) — normalized RGB images
        Returns:
            features: (B, output_dim) if num_spatial_tokens == 1
                      (B, N, output_dim) if num_spatial_tokens > 1
        """
        ...

    def freeze(self) -> None:
        """Disable gradients and set to eval mode."""
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def get_optim_params(
        self, lr: float, lr_backbone: float | None = None
    ) -> list[dict]:
        """Return optimizer parameter groups.

        Returns empty list if encoder is frozen (no trainable params).
        """
        trainable = [p for p in self.parameters() if p.requires_grad]
        if not trainable:
            return []
        return [{"params": trainable, "lr": lr_backbone or lr}]
