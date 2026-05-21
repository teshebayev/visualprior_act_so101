"""Smoke tests for encoders — do not require real LeRobot dataset.

Tests that the encoder classes can be instantiated and produce output of
expected shape. Does not test that the output is semantically meaningful.

Run:
    pytest tests/test_encoders.py -v
"""

from __future__ import annotations

import pytest
import torch

from lerobot_policy_visualprior_act.encoders import (
    ResNetBaselineEncoder,
    VAEEncoder,
    BetaVAEEncoder,
    VQVAEEncoder,
)


# ============================================================
#               Family A: ResNet baseline
# ============================================================


def test_resnet_baseline_no_bottleneck():
    """M0 baseline — spatial tokens preserved."""
    enc = ResNetBaselineEncoder(use_linear_bottleneck=False, pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    out = enc(x)
    assert out.shape == (2, 49, 512), f"Expected (2, 49, 512), got {out.shape}"
    assert enc.output_dim == 512
    assert enc.num_spatial_tokens == 49


def test_resnet_baseline_with_bottleneck():
    """M1 linear bottleneck control — single token output."""
    enc = ResNetBaselineEncoder(
        use_linear_bottleneck=True, bottleneck_dim=32, pretrained=False
    )
    x = torch.randn(2, 3, 224, 224)
    out = enc(x)
    assert out.shape == (2, 32)
    assert enc.output_dim == 32
    assert enc.num_spatial_tokens == 1


# ============================================================
#               Family B: VAE / β-VAE / VQ-VAE
# ============================================================


@pytest.mark.parametrize("latent_dim", [16, 32, 64])
def test_vae_encoder_eval(latent_dim):
    """VAE in eval mode returns μ (no sampling)."""
    enc = VAEEncoder(latent_dim=latent_dim)
    enc.eval()
    x = torch.randn(2, 3, 224, 224)
    out = enc(x)
    assert out.shape == (2, latent_dim)


def test_vae_encoder_deterministic_in_eval():
    """VAE must be deterministic in eval mode."""
    enc = VAEEncoder(latent_dim=32)
    enc.eval()
    x = torch.randn(2, 3, 224, 224)
    out1 = enc(x)
    out2 = enc(x)
    assert torch.allclose(out1, out2)


def test_vae_encode_with_dist():
    """encode_with_dist returns (μ, logσ²) for pretraining."""
    enc = VAEEncoder(latent_dim=32)
    x = torch.randn(2, 3, 224, 224)
    mu, logvar = enc.encode_with_dist(x)
    assert mu.shape == (2, 32)
    assert logvar.shape == (2, 32)


def test_beta_vae_same_architecture_as_vae():
    """β-VAE inference is identical to VAE; β only matters at pretraining."""
    enc = BetaVAEEncoder(latent_dim=32)
    enc.eval()
    x = torch.randn(2, 3, 224, 224)
    out = enc(x)
    assert out.shape == (2, 32)


def test_vqvae_encoder():
    """VQ-VAE returns spatial token sequence."""
    pytest.importorskip("vector_quantize_pytorch")
    enc = VQVAEEncoder(latent_dim=32, codebook_size=128, grid_size=4)
    x = torch.randn(2, 3, 224, 224)
    out = enc(x)
    # (B, grid*grid, latent_dim)
    assert out.shape == (2, 16, 32)
    assert enc.num_spatial_tokens == 16


# ============================================================
#               Freeze behavior
# ============================================================


def test_freeze_disables_gradients():
    """freeze() must disable all gradients."""
    enc = VAEEncoder(latent_dim=32)
    enc.freeze()
    for p in enc.parameters():
        assert not p.requires_grad


def test_freeze_returns_empty_optim_params():
    """Frozen encoder contributes no trainable params to optimizer."""
    enc = VAEEncoder(latent_dim=32)
    enc.freeze()
    assert enc.get_optim_params(lr=1e-4) == []


def test_unfrozen_returns_params():
    """Trainable encoder returns its params for optimizer."""
    enc = VAEEncoder(latent_dim=32)
    groups = enc.get_optim_params(lr=1e-4)
    assert len(groups) == 1
    assert groups[0]["lr"] == 1e-4
    assert len(groups[0]["params"]) > 0


# ============================================================
#               Output dimension contract
# ============================================================


@pytest.mark.parametrize(
    "encoder_factory",
    [
        lambda: ResNetBaselineEncoder(pretrained=False),
        lambda: ResNetBaselineEncoder(
            use_linear_bottleneck=True, bottleneck_dim=32, pretrained=False
        ),
        lambda: VAEEncoder(latent_dim=32),
    ],
)
def test_encoder_has_required_attrs(encoder_factory):
    """All encoders must expose output_dim and num_spatial_tokens."""
    enc = encoder_factory()
    assert hasattr(enc, "output_dim")
    assert hasattr(enc, "num_spatial_tokens")
    assert isinstance(enc.output_dim, int)
    assert enc.output_dim > 0
    assert isinstance(enc.num_spatial_tokens, int)
    assert enc.num_spatial_tokens >= 1
