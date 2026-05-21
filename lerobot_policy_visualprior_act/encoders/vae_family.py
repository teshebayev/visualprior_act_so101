"""VAE / β-VAE / VQ-VAE encoders (Family B).

All three share a common conv backbone. At policy time:
- VAE / β-VAE return the mean (μ) of the latent distribution
- VQ-VAE returns quantized latents from the codebook

Decoder is used only at pretraining (in pretraining/cli.py), not here.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .base import VisualPriorEncoder


class _ConvBackbone(nn.Module):
    """5-layer strided conv backbone, 224x224 -> 7x7 feature map."""

    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, 2, 1),
            nn.ReLU(inplace=True),  # 224 -> 112
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(inplace=True),  # 112 -> 56
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(inplace=True),  # 56 -> 28
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.ReLU(inplace=True),  # 28 -> 14
            nn.Conv2d(256, 256, 4, 2, 1),
            nn.ReLU(inplace=True),  # 14 -> 7
        )

    def forward(self, x):
        return self.layers(x)  # (B, 256, 7, 7)


class VAEEncoder(VisualPriorEncoder):
    """Continuous reconstructive VAE encoder.

    At policy time: returns μ (mean of latent distribution), no sampling.
    At pretraining (when in training mode + return_mode='sample'):
    reparametrized sample for KL loss computation.

    Output: (B, latent_dim) — single token vector.
    """

    def __init__(self, latent_dim: int = 32, return_mode: str = "mean"):
        super().__init__()
        self.latent_dim = latent_dim
        self.return_mode = return_mode

        self.backbone = _ConvBackbone()
        flat_dim = 256 * 7 * 7  # 12544

        self.fc_mu = nn.Linear(flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(flat_dim, latent_dim)

        self.output_dim = latent_dim
        self.num_spatial_tokens = 1

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        h = self.backbone(images)
        h_flat = h.flatten(start_dim=1)
        mu = self.fc_mu(h_flat)

        # In policy mode (eval or return_mode='mean'), always return μ
        if self.return_mode == "mean" or not self.training:
            return mu

        # In pretraining sample mode: reparametrize for KL loss
        logvar = self.fc_logvar(h_flat)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode_with_dist(self, images: torch.Tensor):
        """For pretraining: returns (μ, logσ²) tuple for KL loss."""
        h = self.backbone(images)
        h_flat = h.flatten(start_dim=1)
        return self.fc_mu(h_flat), self.fc_logvar(h_flat)


class BetaVAEEncoder(VAEEncoder):
    """β-VAE — same architecture as VAE.

    Disentanglement is enforced at pretraining via β·KL weighting.
    Inference behavior identical to VAE; the β only matters at pretraining.
    """

    pass


class VQVAEEncoder(VisualPriorEncoder):
    """Discrete latent via vector quantization.

    Uses `vector-quantize-pytorch` package. Codebook usage can be inspected
    via `get_codebook_usage()` — important diagnostic to catch collapse.

    Output: (B, grid_size*grid_size, latent_dim) — sequence of discrete tokens.
    """

    def __init__(
        self,
        latent_dim: int = 32,
        codebook_size: int = 512,
        grid_size: int = 4,
        commitment_weight: float = 0.25,
    ):
        super().__init__()
        try:
            from vector_quantize_pytorch import VectorQuantize
        except ImportError as e:
            raise ImportError(
                "VQ-VAE requires `vector-quantize-pytorch`. Install with:\n"
                '    pip install -e ".[vae]"'
            ) from e

        self.latent_dim = latent_dim
        self.codebook_size = codebook_size
        self.grid_size = grid_size

        self.backbone = _ConvBackbone()
        self.pool = nn.AdaptiveAvgPool2d(grid_size)
        self.pre_quant = nn.Conv2d(256, latent_dim, 1)

        self.quantizer = VectorQuantize(
            dim=latent_dim,
            codebook_size=codebook_size,
            commitment_weight=commitment_weight,
        )

        # Spatial sequence output
        self.output_dim = latent_dim
        self.num_spatial_tokens = grid_size * grid_size

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        h = self.backbone(images)  # (B, 256, 7, 7)
        h = self.pool(h)  # (B, 256, grid, grid)
        h = self.pre_quant(h)  # (B, latent_dim, grid, grid)

        b, c, gh, gw = h.shape
        h_flat = h.permute(0, 2, 3, 1).reshape(b, gh * gw, c)

        quantized, _, _ = self.quantizer(h_flat)
        # quantized: (B, grid*grid, latent_dim)
        return quantized

    def get_codebook_usage(self) -> float:
        """% of codebook entries that have been used. <30% indicates collapse."""
        # vector-quantize-pytorch tracks usage in cluster_size
        if hasattr(self.quantizer, "_codebook"):
            cluster_size = getattr(
                self.quantizer._codebook, "cluster_size", None
            )
            if cluster_size is not None:
                used = (cluster_size > 0).sum().item()
                return used / self.codebook_size
        return -1.0  # unable to determine
