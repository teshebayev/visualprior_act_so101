"""SAM2 encoder (Family D — foundation models).

Uses SAM2 image encoder as a frozen feature extractor.

Notes
-----
1. SAM2 was trained on ImageNet-normalized inputs at 1024x1024 resolution.
   We pass 224x224 (matching every other encoder in this plugin) so the
   model has to internally interpolate position embeddings or accept the
   smaller patch grid. This works for hiera-tiny but may produce suboptimal
   features compared to native 1024-res inference. If quality matters,
   evaluate on native resolution by setting `sam2_input_size=1024` in your
   policy preprocessor (and accept the much higher compute cost).

2. We register SAM2-specific normalization buffers in __init__ and apply
   them inside forward(). Input is expected in [0, 1].

Requires `transformers` >= 4.40. Install:
    pip install -e ".[foundation]"
"""

from __future__ import annotations

import torch

from .base import VisualPriorEncoder


# SAM2 uses ImageNet-style stats (this is the documented preprocessing for
# Hiera-based SAM2; the Mask R-CNN family of SAM ports also use ImageNet).
_SAM2_MEAN = (0.485, 0.456, 0.406)
_SAM2_STD = (0.229, 0.224, 0.225)


class SAM2Encoder(VisualPriorEncoder):
    """SAM2 image encoder features.

    Always frozen — SAM2 is too big to finetune on small datasets.

    Output: (B, N, C) — spatial token sequence from SAM2's vision tower.
    """

    def __init__(self, model_name: str = "facebook/sam2-hiera-tiny"):
        super().__init__()
        try:
            from transformers import AutoModel
        except ImportError as e:
            raise ImportError(
                "SAM2 requires `transformers`. "
                'Install: pip install -e ".[foundation]"'
            ) from e

        # Load full model, extract image/vision encoder.
        full = AutoModel.from_pretrained(model_name)
        if hasattr(full, "vision_encoder"):
            self.vision_encoder = full.vision_encoder
        elif hasattr(full, "image_encoder"):
            self.vision_encoder = full.image_encoder
        else:
            raise AttributeError(
                f"Could not find vision/image encoder on {type(full)}. "
                f"Check transformers version (SAM2 API moves between releases)."
            )

        # Register normalization buffers (same shape conventions as base helper
        # but with SAM2-specific stats — kept identical to ImageNet because
        # that's what hiera-tiny was trained with).
        self.register_buffer(
            "_sam2_mean",
            torch.tensor(_SAM2_MEAN, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_sam2_std",
            torch.tensor(_SAM2_STD, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

        self.output_dim, self.num_spatial_tokens = self._infer_output_shape()

    def _normalize(self, images: torch.Tensor) -> torch.Tensor:
        """Apply SAM2's expected normalization. Input in [0, 1]."""
        return (images - self._sam2_mean) / self._sam2_std

    def _infer_output_shape(self) -> tuple[int, int]:
        self.eval()
        with torch.no_grad():
            dummy = torch.full((1, 3, 224, 224), 0.5)
            x = self._normalize(dummy)
            out = self.vision_encoder(x)
            features = self._extract_features(out)
        if features.dim() == 3:
            _, n, c = features.shape
            return c, n
        elif features.dim() == 4:
            _, c, h, w = features.shape
            return c, h * w
        else:
            raise RuntimeError(f"Unexpected SAM2 feature shape: {features.shape}")

    def _extract_features(self, out) -> torch.Tensor:
        """Extract feature tensor from SAM2 encoder output."""
        if hasattr(out, "last_hidden_state"):
            return out.last_hidden_state
        elif hasattr(out, "vision_features"):
            return out.vision_features
        elif isinstance(out, torch.Tensor):
            return out
        else:
            raise RuntimeError(f"Unknown SAM2 encoder output type: {type(out)}")

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self._normalize(images)
        out = self.vision_encoder(x)
        features = self._extract_features(out)

        if features.dim() == 4:
            # (B, C, H, W) -> (B, H*W, C)
            b, c, h, w = features.shape
            features = features.permute(0, 2, 3, 1).reshape(b, h * w, c)
        # else already (B, N, C)
        return features
