"""U-Net encoder (Family C — task-supervised priors).

Uses encoder branch of a pretrained U-Net (typically ImageNet/COCO segmentation).
Requires `segmentation-models-pytorch`. Install:
    pip install -e ".[unet]"
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .base import VisualPriorEncoder


class UNetEncoder(VisualPriorEncoder):
    """Encoder branch of pretrained U-Net.

    smp.Unet wraps a backbone (resnet34 by default) that was pretrained on
    ImageNet, then segmentation head is trained on COCO/other. We use only
    the encoder (downsampling path).

    Output: (B, N, C) — spatial token sequence from deepest encoder feature map.
    """

    def __init__(
        self, encoder_name: str = "resnet34", pretrained: str = "imagenet"
    ):
        super().__init__()
        try:
            import segmentation_models_pytorch as smp
        except ImportError as e:
            raise ImportError(
                "U-Net encoder requires `segmentation-models-pytorch`. "
                'Install: pip install -e ".[unet]"'
            ) from e

        # Build full U-Net and take only encoder
        full = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=pretrained,
            in_channels=3,
            classes=1,  # dummy — we discard decoder
        )
        self.encoder = full.encoder

        # Determine output shape via dummy forward
        self.output_dim, self.num_spatial_tokens = self._infer_output_shape()

    def _infer_output_shape(self) -> tuple[int, int]:
        self.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            features = self.encoder(dummy)
        # encoder returns list of feature maps at different scales
        # Take deepest (smallest spatial, most channels)
        deepest = features[-1]
        _, c, h, w = deepest.shape
        return c, h * w

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.encoder(images)  # list of feature maps
        deepest = features[-1]  # (B, C, H', W')
        b, c, h, w = deepest.shape
        return deepest.permute(0, 2, 3, 1).reshape(b, h * w, c)
