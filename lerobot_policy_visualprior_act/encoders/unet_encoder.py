"""U-Net encoder (Family C — task-supervised priors).

Uses encoder branch of a pretrained U-Net. With default `encoder_name='resnet34'`
and `pretrained='imagenet'`, the backbone expects ImageNet-normalized inputs.

Requires `segmentation-models-pytorch`. Install:
    pip install -e ".[unet]"
"""

from __future__ import annotations

import torch

from .base import VisualPriorEncoder


class UNetEncoder(VisualPriorEncoder):
    """Encoder branch of pretrained U-Net.

    smp.Unet wraps a backbone (resnet34 by default) pretrained on ImageNet.
    We use only the encoder (downsampling path).

    Output: (B, N, C) — spatial token sequence from deepest encoder feature map.
    """

    # If you swap to a backbone with different stats (e.g. SwinV2 trained on
    # a non-ImageNet corpus), override these or set them after __init__.
    _NORMALIZE = True

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

        full = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=pretrained,
            in_channels=3,
            classes=1,  # dummy — we discard decoder
        )
        self.encoder = full.encoder

        # ImageNet stats are correct for the default ResNet/EfficientNet/etc
        # SMP backbones when pretrained='imagenet'. If pretrained=None, the
        # backbone is randomly init'd — normalization stats are irrelevant
        # but applying them won't hurt.
        if self._NORMALIZE and pretrained == "imagenet":
            self._register_imagenet_norm()

        self.output_dim, self.num_spatial_tokens = self._infer_output_shape()

    def _infer_output_shape(self) -> tuple[int, int]:
        self.eval()
        with torch.no_grad():
            dummy = torch.full((1, 3, 224, 224), 0.5)
            x = self._imagenet_normalize(dummy)
            features = self.encoder(x)
        deepest = features[-1]
        _, c, h, w = deepest.shape
        return c, h * w

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self._imagenet_normalize(images)
        features = self.encoder(x)
        deepest = features[-1]  # (B, C, H', W')
        b, c, h, w = deepest.shape
        return deepest.permute(0, 2, 3, 1).reshape(b, h * w, c)
