"""VAE / β-VAE / VQ-VAE encoders (Family B).

All three share a common conv backbone. At policy time:
- VAE / β-VAE return the mean (μ) of the latent distribution
- VQ-VAE returns quantized latents from the codebook

Decoder is used only at pretraining (in pretraining/cli.py), not here.

SPATIAL vs FLAT mode (VAE / β-VAE)
==================================
`spatial=True` (default, RECOMMENDED): keeps the 7x7 grid from the backbone
and projects channels to latent_dim via a 1x1 conv. Output: (B, 49, latent_dim).
This gives the ACT transformer many more visual tokens to attend over, matching
the spatial-token convention of the ResNet baseline.

`spatial=False` (legacy): flattens the 7x7 feature map and projects to a single
latent_dim vector. Output: (B, latent_dim). Extreme bottleneck — kept only for
backward compatibility with old pretrained checkpoints and as an ablation.
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
    """Reconstructive VAE encoder with optional spatial output.

    At policy time: returns μ (mean of latent distribution), no sampling.
    At pretraining (when training mode + return_mode='sample'):
    reparametrized sample for KL loss computation.

    Args:
        latent_dim: per-token / per-vector latent dimension
        spatial: if True (default), produce 49 spatial tokens of latent_dim;
            if False, collapse to a single latent_dim vector (legacy behavior).
        return_mode: 'mean' (default for policy) or 'sample' (for pretraining).
    """

    def __init__(
        self,
        latent_dim: int = 32,
        spatial: bool = True,
        return_mode: str = "mean",
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.spatial = spatial
        self.return_mode = return_mode

        self.backbone = _ConvBackbone()

        if spatial:
            # 1x1 conv across channels keeps spatial grid intact.
            # (B, 256, 7, 7) -> (B, latent_dim, 7, 7) -> (B, 49, latent_dim)
            self.conv_mu = nn.Conv2d(256, latent_dim, 1)
            self.conv_logvar = nn.Conv2d(256, latent_dim, 1)
            self.output_dim = latent_dim
            self.num_spatial_tokens = 49  # 7 * 7
        else:
            # Legacy single-token bottleneck.
            flat_dim = 256 * 7 * 7  # 12544
            self.fc_mu = nn.Linear(flat_dim, latent_dim)
            self.fc_logvar = nn.Linear(flat_dim, latent_dim)
            self.output_dim = latent_dim
            self.num_spatial_tokens = 1

    def _project(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run mu/logvar projection. Returns (mu, logvar) with consistent shape."""
        if self.spatial:
            mu = self.conv_mu(h)  # (B, latent_dim, 7, 7)
            logvar = self.conv_logvar(h)
            # to (B, 49, latent_dim)
            b, c, gh, gw = mu.shape
            mu = mu.permute(0, 2, 3, 1).reshape(b, gh * gw, c)
            logvar = logvar.permute(0, 2, 3, 1).reshape(b, gh * gw, c)
        else:
            h_flat = h.flatten(start_dim=1)
            mu = self.fc_mu(h_flat)
            logvar = self.fc_logvar(h_flat)
        return mu, logvar

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        h = self.backbone(images)
        mu, logvar = self._project(h)

        # Policy mode (eval or explicit mean): just return μ
        if self.return_mode == "mean" or not self.training:
            return mu

        # Pretraining sample mode: reparametrize
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode_with_dist(
        self, images: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """For pretraining: returns (μ, logσ²) for KL loss computation."""
        h = self.backbone(images)
        return self._project(h)


class BetaVAEEncoder(VAEEncoder):
    """β-VAE — same architecture as VAE.

    Disentanglement is enforced at pretraining via β·KL weighting.
    Inference behavior is identical to VAE; β only matters at pretraining time.
    """

    pass


class VQVAEEncoder(VisualPriorEncoder):
    """Discrete latent via vector quantization.

    Uses `vector-quantize-pytorch` package. Codebook usage can be inspected
    via `get_codebook_usage()` — important diagnostic to catch collapse.

    Args:
        latent_dim: codebook entry dimension.
        codebook_size: number of discrete codes in the codebook.
        grid_size: spatial grid the backbone is pooled to before quantization.
            Default 7 (no pooling — full backbone resolution). Use smaller
            values (e.g. 4) for coarser tokens / smaller sequence length.
        commitment_weight: VQ commitment loss weight at pretraining.

    Output: (B, grid_size*grid_size, latent_dim) — sequence of discrete tokens.
    """

    def __init__(
        self,
        latent_dim: int = 32,
        codebook_size: int = 512,
        grid_size: int = 7,
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
        # If grid_size == 7, AdaptiveAvgPool2d is an identity but cheap, so keep it
        # unconditionally for code uniformity.
        self.pool = nn.AdaptiveAvgPool2d(grid_size)
        self.pre_quant = nn.Conv2d(256, latent_dim, 1)

        self.quantizer = VectorQuantize(
            dim=latent_dim,
            codebook_size=codebook_size,
            commitment_weight=commitment_weight,
        )

        self.output_dim = latent_dim
        self.num_spatial_tokens = grid_size * grid_size

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        h = self.backbone(images)  # (B, 256, 7, 7)
        h = self.pool(h)  # (B, 256, grid, grid)
        h = self.pre_quant(h)  # (B, latent_dim, grid, grid)

        b, c, gh, gw = h.shape
        h_flat = h.permute(0, 2, 3, 1).reshape(b, gh * gw, c)

        quantized, _, _ = self.quantizer(h_flat)
        return quantized  # (B, grid*grid, latent_dim)

    def get_codebook_usage(self) -> float:
        """% of codebook entries that have been used. <30% indicates collapse."""
        if hasattr(self.quantizer, "_codebook"):
            cluster_size = getattr(
                self.quantizer._codebook, "cluster_size", None
            )
            if cluster_size is not None:
                used = (cluster_size > 0).sum().item()
                return used / self.codebook_size
        return -1.0
