"""DINOv2 encoder (Family D — foundation models).

Uses pretrained DINOv2 (self-supervised ViT) as frozen feature extractor.
DINOv2 was trained with ImageNet-style normalization, so we apply
(x - imagenet_mean) / imagenet_std inside forward().

Requires `transformers` >= 4.40. Install:
    pip install -e ".[foundation]"
"""

from __future__ import annotations

import torch

from .base import VisualPriorEncoder


class DINOv2Encoder(VisualPriorEncoder):
    """DINOv2 ViT features.

    Always frozen. Uses last_hidden_state — sequence of patch tokens + CLS.
    The CLS token is dropped to keep only spatial tokens.

    For default dinov2-small with 224x224 input:
        patch size 14 -> 16x16 = 256 patches, hidden_dim=384
        Output shape: (B, 256, 384)
    """

    def __init__(self, model_name: str = "facebook/dinov2-small"):
        super().__init__()
        try:
            from transformers import AutoModel
        except ImportError as e:
            raise ImportError(
                "DINOv2 requires `transformers`. "
                'Install: pip install -e ".[foundation]"'
            ) from e

        self.model = AutoModel.from_pretrained(model_name)

        # DINOv2 normalization == ImageNet stats.
        self._register_imagenet_norm()

        self.output_dim, self.num_spatial_tokens = self._infer_output_shape()

    def _infer_output_shape(self) -> tuple[int, int]:
        self.eval()
        with torch.no_grad():
            dummy = torch.full((1, 3, 224, 224), 0.5)
            x = self._imagenet_normalize(dummy)
            out = self.model(x)
        hidden = out.last_hidden_state  # (1, N+1, C) — N patch tokens + CLS
        _, n, c = hidden.shape
        # Drop CLS: keep only patch tokens (the spatial sequence)
        return c, n - 1

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self._imagenet_normalize(images)
        out = self.model(x)
        hidden = out.last_hidden_state  # (B, N+1, C)
        return hidden[:, 1:, :]  # drop CLS at index 0
