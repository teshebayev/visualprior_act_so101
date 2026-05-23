"""Visual prior encoders.

Provides a factory function `build_encoder(config)` that returns the right
encoder class based on config.encoder string.
"""

from __future__ import annotations

import warnings
from pathlib import Path

from .base import VisualPriorEncoder
from .resnet_baseline import ResNetBaselineEncoder
from .vae_family import BetaVAEEncoder, VAEEncoder, VQVAEEncoder


def build_encoder(config) -> VisualPriorEncoder:
    """Factory dispatching on config.encoder."""
    enc = config.encoder

    if enc == "resnet18":
        return ResNetBaselineEncoder(
            use_linear_bottleneck=config.use_linear_bottleneck,
            bottleneck_dim=config.bottleneck_dim,
        )

    elif enc == "vae":
        e = VAEEncoder(
            latent_dim=config.vae_latent_dim,
            spatial=config.vae_spatial,
        )
        _load_pretrained(e, config.vae_pretrained_path)
        return e

    elif enc == "beta_vae":
        e = BetaVAEEncoder(
            latent_dim=config.vae_latent_dim,
            spatial=config.vae_spatial,
        )
        _load_pretrained(e, config.vae_pretrained_path)
        return e

    elif enc == "vqvae":
        e = VQVAEEncoder(
            latent_dim=config.vae_latent_dim,
            codebook_size=config.vqvae_codebook_size,
            grid_size=config.vqvae_grid_size,
        )
        _load_pretrained(e, config.vae_pretrained_path)
        return e

    elif enc == "yolo":
        from .yolo_encoder import YOLOBackboneEncoder

        return YOLOBackboneEncoder(
            model_name=config.yolo_model_name,
            feature_level=config.yolo_feature_level,
        )

    elif enc == "yolo_bbox":
        from .yolo_encoder import YOLOBBoxEncoder

        return YOLOBBoxEncoder(
            model_name=config.yolo_model_name, topk=config.yolo_topk_boxes
        )

    elif enc == "unet":
        from .unet_encoder import UNetEncoder

        return UNetEncoder(
            encoder_name=config.unet_encoder_name,
            pretrained=config.unet_pretrained,
        )

    elif enc == "sam2":
        from .sam2_encoder import SAM2Encoder

        return SAM2Encoder(model_name=config.sam2_model_name)

    elif enc == "dinov2":
        from .dinov2_encoder import DINOv2Encoder

        return DINOv2Encoder(model_name=config.dinov2_model_name)

    else:
        raise ValueError(f"Unknown encoder: {enc}")


def _load_pretrained(encoder: VisualPriorEncoder, path: str | None) -> None:
    """Load pretrained encoder weights from .safetensors file.

    Tolerant of missing path / missing file: emits a warning and leaves the
    encoder freshly initialized. This is required for two cases:

    1. Loading a saved policy via PreTrainedPolicy.from_pretrained — the
       policy state_dict will overwrite encoder weights AFTER __init__, so
       a missing pretrained file is fine here.
    2. Loading a config from the Hub on a different machine, where the local
       path doesn't exist. The user is expected to load weights from the
       saved policy checkpoint, not from a separate .safetensors.

    Hard-error only happens in pretraining/training flows that have not
    saved a checkpoint yet — and even then, the caller should set
    vae_pretrained_path explicitly.
    """
    if path is None:
        warnings.warn(
            "vae_pretrained_path is None — VAE-family encoder will be "
            "randomly initialized. If you're loading a saved policy this "
            "is fine (encoder weights come from the checkpoint). For fresh "
            "policy training, pretrain first and set vae_pretrained_path.",
            stacklevel=3,
        )
        return

    weights_path = Path(path)
    if not weights_path.exists():
        warnings.warn(
            f"vae_pretrained_path={path} does not exist on this machine. "
            f"Skipping pretrained-weight load — assuming weights will come "
            f"from the policy checkpoint state_dict.",
            stacklevel=3,
        )
        return

    from safetensors.torch import load_file

    state = load_file(str(weights_path))
    missing, unexpected = encoder.load_state_dict(state, strict=False)

    # Decoder.* keys are expected to be missing (pretraining saved encoder only).
    real_missing = [
        k for k in missing if not k.startswith(("decoder.", "_decoder."))
    ]
    if real_missing:
        warnings.warn(
            f"Missing keys when loading pretrained encoder: {real_missing[:5]}"
            f"{' ...' if len(real_missing) > 5 else ''}",
            stacklevel=3,
        )
    if unexpected:
        warnings.warn(
            f"Unexpected keys when loading pretrained encoder: "
            f"{unexpected[:5]}{' ...' if len(unexpected) > 5 else ''}. "
            f"This usually means architecture changed (e.g. spatial=False "
            f"checkpoint loaded into spatial=True encoder). Re-pretrain.",
            stacklevel=3,
        )


__all__ = [
    "VisualPriorEncoder",
    "ResNetBaselineEncoder",
    "VAEEncoder",
    "BetaVAEEncoder",
    "VQVAEEncoder",
    "build_encoder",
]
