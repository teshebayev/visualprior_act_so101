"""Smoke tests for VisualPriorACTConfig.

Tests config validation and registration. Requires lerobot to be installed.
"""

from __future__ import annotations

import pytest

pytest.importorskip("lerobot")


def test_config_imports():
    """Config can be imported."""
    from lerobot_policy_visualprior_act import VisualPriorACTConfig
    assert VisualPriorACTConfig is not None


def test_config_resnet_baseline():
    """M0 baseline config is valid."""
    from lerobot_policy_visualprior_act import VisualPriorACTConfig

    config = _make_minimal_config(encoder="resnet18")
    assert config.encoder == "resnet18"
    assert not config.use_linear_bottleneck


def test_config_linear_bottleneck_requires_resnet18():
    """M1 control config rejects non-resnet encoder."""
    from lerobot_policy_visualprior_act import VisualPriorACTConfig

    with pytest.raises(ValueError, match="linear_bottleneck"):
        _make_minimal_config(
            encoder="vae",
            use_linear_bottleneck=True,
            vae_pretrained_path="/tmp/fake.safetensors",
        )


def test_config_vae_requires_pretrained_path():
    """VAE-family requires vae_pretrained_path."""
    with pytest.raises(ValueError, match="vae_pretrained_path"):
        _make_minimal_config(encoder="vqvae", vae_pretrained_path=None)


def test_config_foundation_forces_freeze():
    """SAM2/DINOv2 auto-set freeze_encoder=True."""
    config = _make_minimal_config(encoder="sam2", freeze_encoder=False)
    assert config.freeze_encoder is True


# Helpers


def _make_minimal_config(**overrides):
    """Create a minimal valid config for testing."""
    from lerobot_policy_visualprior_act import VisualPriorACTConfig
    from lerobot.configs.types import FeatureType, PolicyFeature

    # Minimal required input/output features
    defaults = dict(
        input_features={
            "observation.images.front": PolicyFeature(
                type=FeatureType.VISUAL, shape=(3, 224, 224)
            ),
            "observation.state": PolicyFeature(
                type=FeatureType.STATE, shape=(6,)
            ),
        },
        output_features={
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(6,)),
        },
    )
    defaults.update(overrides)
    return VisualPriorACTConfig(**defaults)
