"""Configuration for VisualPriorACTPolicy.

Registers itself with LeRobot as policy.type='visualprior_act' so that the
standard CLI can be used:
    lerobot-train --policy.type=visualprior_act --policy.encoder=vqvae ...

This config dataclass holds:
- ACT-specific hyperparameters (transformer dims, chunk size, action-CVAE)
- Encoder selection ('encoder' field)
- Encoder-specific hyperparameters (one block per family)
- Optimizer / scheduler presets

The choice of encoder is validated in __post_init__.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

from lerobot.configs import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim import AdamWConfig


# Valid encoder identifiers (replaces the old Literal[...] type — see config).
VALID_ENCODERS = frozenset({
    "resnet18",
    "vae",
    "beta_vae",
    "vqvae",
    "yolo",
    "unet",
    "yolo_bbox",
    "sam2",
    "dinov2",
})


@PreTrainedConfig.register_subclass("visualprior_act")
@dataclass
class VisualPriorACTConfig(PreTrainedConfig):
    """Configuration for ACT with replaceable visual prior encoder."""

    # ---------- ACT-specific (mirrors lerobot.policies.act.ACTConfig) ----------
    n_obs_steps: int = 1
    chunk_size: int = 100
    n_action_steps: int = 100

    # Transformer
    dim_model: int = 512
    dim_feedforward: int = 3200
    n_encoder_layers: int = 4
    n_decoder_layers: int = 1  # ACT historically uses 1
    n_heads: int = 8
    dropout: float = 0.1
    pre_norm: bool = False

    # Internal action-CVAE (z_act). Kept unchanged from standard ACT.
    use_vae: bool = True
    latent_dim: int = 32  # size of z_act, NOT z_vis. See `vae_latent_dim` for z_vis.
    n_vae_encoder_layers: int = 4
    kl_weight: float = 10.0

    temporal_ensemble_coeff: Optional[float] = None

    # ---------- Visual encoder choice ----------
    encoder: str = "resnet18"

    # Unified output dimension — projector maps every encoder to this
    projector_dim: int = 256
    projector_hidden_dim: int = 256

    # IMPORTANT: For VAE-family encoders the recommended workflow is
    # pretrain -> freeze -> train policy. Default True now to avoid silently
    # destroying pretrained features during policy training.
    freeze_encoder: bool = True

    # ---------- Encoder-specific hyperparams ----------
    # ResNet baseline (Family A)
    use_linear_bottleneck: bool = False  # M1 control variant
    bottleneck_dim: int = 32

    # VAE family (Family B)
    vae_latent_dim: int = 32
    vae_pretrained_path: Optional[str] = None
    vae_beta: float = 1.0
    # NEW: spatial VAE keeps the 7x7 grid -> 49 tokens of latent_dim each.
    # Old single-token bottleneck (spatial=False) is preserved as ablation.
    vae_spatial: bool = True
    # VQ-VAE: grid_size=7 keeps the full backbone resolution (no extra pooling).
    # The old default of 4 was an unnecessary information bottleneck.
    vqvae_codebook_size: int = 512
    vqvae_grid_size: int = 7

    # YOLO (Family C)
    yolo_model_name: str = "yolov8n"
    yolo_feature_level: int = 4  # P3=3, P4=4, P5=5
    yolo_topk_boxes: int = 5  # for structured yolo_bbox variant

    # U-Net (Family C)
    unet_encoder_name: str = "resnet34"
    unet_pretrained: str = "imagenet"

    # SAM2 (Family D)
    sam2_model_name: str = "facebook/sam2-hiera-tiny"

    # DINOv2 (Family D)
    dinov2_model_name: str = "facebook/dinov2-small"

    # ---------- Spatial structure of visual tokens ----------
    use_spatial_tokens: bool = True

    # ---------- Optimizer / scheduler ----------
    optimizer_lr: float = 1e-5
    optimizer_lr_backbone: float = 1e-5
    optimizer_weight_decay: float = 1e-4

    # ---------- Normalization ----------
    # Match stock ACT: IDENTITY for VISUAL (images stay in [0, 1] as the
    # dataset stores them — same statistics our pretrained encoders saw).
    # The OLD default of MEAN_STD here caused a hard distribution shift
    # between pretraining and policy inference and was almost certainly
    # the main cause of "policy doesn't work at eval time".
    normalization_mapping: dict = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # ============================================================
    #               Validation
    # ============================================================

    def __post_init__(self):
        super().__post_init__()

        if self.encoder not in VALID_ENCODERS:
            raise ValueError(
                f"encoder='{self.encoder}' is invalid. "
                f"Choose one of: {sorted(VALID_ENCODERS)}"
            )

        # VAE family used to hard-require vae_pretrained_path here. We now
        # downgrade this to a warning, because:
        #   1. PreTrainedPolicy.from_pretrained() instantiates the config
        #      first, then loads state_dict — the pretrained .safetensors is
        #      not needed in that case, the encoder weights come from the
        #      policy checkpoint.
        #   2. On a different machine the original local path won't exist.
        # _load_pretrained in encoders/__init__.py handles None gracefully.
        if (
            self.encoder in ("vae", "beta_vae", "vqvae")
            and self.vae_pretrained_path is None
        ):
            warnings.warn(
                f"encoder='{self.encoder}' without vae_pretrained_path. "
                f"This is fine if you're reloading a saved policy. "
                f"For fresh training, pretrain first:\n"
                f"    pretrain-visual-encoder --encoder-type={self.encoder} ...",
                stacklevel=2,
            )

        if self.use_linear_bottleneck and self.encoder != "resnet18":
            raise ValueError(
                "use_linear_bottleneck=True is only valid for encoder='resnet18'"
            )

        if self.encoder in ("sam2", "dinov2") and not self.freeze_encoder:
            warnings.warn(
                f"encoder='{self.encoder}' is typically used frozen on small "
                "datasets. Setting freeze_encoder=True.",
                stacklevel=2,
            )
            self.freeze_encoder = True

    def validate_features(self) -> None:
        """Verify dataset features are compatible. Called from policy.__init__."""
        has_image = any("image" in k.lower() for k in self.input_features.keys())
        if not has_image:
            raise ValueError(
                "VisualPriorACT requires at least one image input. "
                f"Got: {list(self.input_features.keys())}"
            )

        if "observation.state" not in self.input_features:
            raise ValueError(
                "VisualPriorACT requires 'observation.state' in input_features."
            )

        if "action" not in self.output_features:
            raise ValueError(
                "VisualPriorACT requires 'action' in output_features."
            )

    # ============================================================
    #               Optimizer / scheduler presets
    # ============================================================

    def get_optimizer_preset(self):
        return AdamWConfig(
            lr=self.optimizer_lr,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self):
        return None

    # ============================================================
    #               Delta indices (required by PreTrainedConfig)
    # ============================================================

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
