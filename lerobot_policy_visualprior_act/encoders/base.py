"""Base interface for all visual prior encoders.

All encoders implement this interface so that VisualPriorACTPolicy can
treat them uniformly regardless of internal architecture.

NORMALIZATION CONTRACT
======================
The policy preprocessor pipeline now uses `NormalizationMode.IDENTITY` for
VISUAL features, so all encoders receive images in `[0, 1]` (as stored in
the LeRobotDataset). Each encoder is responsible for applying its own
normalization in `forward()` if its pretrained backbone expects something
specific (e.g. ImageNet mean/std for ResNet / YOLO / U-Net / DINOv2,
custom stats for SAM2). The VAE / β-VAE / VQ-VAE family pretrains on raw
[0, 1] images, so they do NOT normalize.

Helpers `_register_imagenet_norm()` and `_imagenet_normalize()` are provided
for the common ImageNet-stats case. Encoders with custom stats should
register their own buffers and apply them explicitly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


# ImageNet RGB mean/std (used by every encoder pretrained on ImageNet).
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class VisualPriorEncoder(nn.Module, ABC):
    """Common interface for all visual encoders.

    Subclasses must:
    - Set `self.output_dim` in __init__ (size of feature dimension)
    - Set `self.num_spatial_tokens` (1 for vector output, N for sequence)
    - Implement forward(images) -> features

    Subclasses that need ImageNet-style normalization should call
    `self._register_imagenet_norm()` in __init__ and apply
    `self._imagenet_normalize(images)` at the start of forward().
    """

    output_dim: int  # required
    num_spatial_tokens: int = 1  # default: single token

    # ---------------- Normalization helpers ----------------

    def _register_imagenet_norm(self) -> None:
        """Register `_in_mean` / `_in_std` buffers for ImageNet normalization.

        Buffers are moved with the encoder via `.to(device)` automatically.
        """
        self.register_buffer(
            "_in_mean",
            torch.tensor(_IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_in_std",
            torch.tensor(_IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

    def _imagenet_normalize(self, images: torch.Tensor) -> torch.Tensor:
        """Apply ImageNet normalization if `_register_imagenet_norm()` was called.

        Safe no-op if the buffers weren't registered (returns input unchanged).
        Assumes input is in `[0, 1]`. If you can't guarantee that (e.g. some
        upstream component double-normalizes), clamp before calling.
        """
        if not hasattr(self, "_in_mean"):
            return images
        return (images - self._in_mean) / self._in_std

    # ---------------- Forward interface ----------------

    @abstractmethod
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (B, C, H, W) — RGB images in [0, 1] (NO normalization
                applied by the preprocessor; do it yourself if needed).
        Returns:
            features: (B, output_dim) if num_spatial_tokens == 1
                      (B, N, output_dim) if num_spatial_tokens > 1
        """
        ...

    # ---------------- Freeze / optim helpers ----------------

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
