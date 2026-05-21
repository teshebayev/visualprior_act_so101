"""Visual prior encoders.

Provides a factory function `build_encoder(config)` that returns the right
encoder class based on config.encoder string.
"""

from __future__ import annotations

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
        e = VAEEncoder(latent_dim=config.vae_latent_dim)
        _load_pretrained(e, config.vae_pretrained_path)
        return e

    elif enc == "beta_vae":
        e = BetaVAEEncoder(latent_dim=config.vae_latent_dim)
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
    """Load pretrained encoder weights from .safetensors file."""
    if path is None:
        raise ValueError("Pretrained path required for VAE-family encoders")

    from safetensors.torch import load_file

    state = load_file(path)
    # strict=False — decoder weights may be absent (we only need encoder)
    missing, unexpected = encoder.load_state_dict(state, strict=False)

    # Filter expected missing keys (e.g. decoder.* if pretraining script saved encoder only)
    real_missing = [k for k in missing if not k.startswith(("decoder.", "_decoder."))]
    if real_missing:
        import warnings

        warnings.warn(
            f"Missing keys when loading pretrained encoder: {real_missing[:5]}",
            stacklevel=2,
        )


__all__ = [
    "VisualPriorEncoder",
    "ResNetBaselineEncoder",
    "VAEEncoder",
    "BetaVAEEncoder",
    "VQVAEEncoder",
    "build_encoder",
]
