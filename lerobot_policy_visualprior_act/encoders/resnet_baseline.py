"""ResNet-18 baseline encoder (Family A).

Two variants:
- M0: pretrained ResNet-18, full feature map (matches standard ACT)
- M1: same backbone + linear bottleneck control variant
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

from .base import VisualPriorEncoder


class ResNetBaselineEncoder(VisualPriorEncoder):
    """ResNet-18 with optional linear bottleneck.

    Without bottleneck (M0): mimics standard ACT visual frontend.
    With bottleneck (M1): adds Linear -> R^bottleneck_dim, controls for
    dimensionality reduction effect.

    Output:
    - If use_linear_bottleneck: (B, bottleneck_dim), single token
    - Otherwise: (B, 49, 512) — 7x7 spatial grid of 512-dim features
                 (matches ImageNet ResNet-18 output before global pool)
    """

    def __init__(
        self,
        use_linear_bottleneck: bool = False,
        bottleneck_dim: int = 32,
        pretrained: bool = True,
    ):
        super().__init__()
        self.use_linear_bottleneck = use_linear_bottleneck

        # Load pretrained ResNet-18 backbone
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        full = resnet18(weights=weights)

        # Strip avgpool + fc, keep up to layer4 -> (B, 512, 7, 7) for 224x224
        self.backbone = nn.Sequential(*list(full.children())[:-2])

        if use_linear_bottleneck:
            # Flatten + project to bottleneck_dim — single-vector output
            self.bottleneck = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(512, bottleneck_dim),
            )
            self.output_dim = bottleneck_dim
            self.num_spatial_tokens = 1
        else:
            # Spatial tokens preserved (standard ACT pattern)
            self.bottleneck = None
            self.output_dim = 512
            self.num_spatial_tokens = 49  # 7x7

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)  # (B, 512, 7, 7)

        if self.bottleneck is not None:
            return self.bottleneck(features)  # (B, bottleneck_dim)
        else:
            # Spatial flatten: (B, 512, 7, 7) -> (B, 49, 512)
            b, c, h, w = features.shape
            return features.permute(0, 2, 3, 1).reshape(b, h * w, c)
