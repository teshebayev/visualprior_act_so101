"""DINOv2 encoder (Family D — foundation models).

Uses pretrained DINOv2 (self-supervised ViT) as frozen feature extractor.
Requires `transformers` >= 4.40. Install:
    pip install -e ".[foundation]"
"""

from __future__ import annotations

import torch

from .base import VisualPriorEncoder


class DINOv2Encoder(VisualPriorEncoder):
    """DINOv2 ViT-S/14 features.

    Always frozen. Uses last_hidden_state — sequence of patch tokens + CLS.

    Output: (B, N, C) — spatial token sequence.
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

        self.output_dim, self.num_spatial_tokens = self._infer_output_shape()

    def _infer_output_shape(self) -> tuple[int, int]:
        self.eval()
        with torch.no_grad():
            # DINOv2 expects 224x224 or 518x518 depending on variant
            dummy = torch.zeros(1, 3, 224, 224)
            out = self.model(dummy)
        hidden = out.last_hidden_state  # (1, N+1, C) — N patch tokens + CLS
        _, n, c = hidden.shape
        # Drop CLS token, keep only patch tokens for spatial structure
        return c, n - 1

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        out = self.model(images)
        hidden = out.last_hidden_state  # (B, N+1, C)
        # Drop CLS (index 0), keep patch tokens
        return hidden[:, 1:, :]
