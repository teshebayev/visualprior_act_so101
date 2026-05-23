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
    """Configuration for ACT with replaceable visual prior encoder.

    Encoder choices:
        - 'resnet18'   — baseline (M0) or with linear bottleneck (M1)
        - 'vae'        — M2/M3, requires vae_pretrained_path
        - 'beta_vae'   — M4/M5, requires vae_pretrained_path
        - 'vqvae'      — M6/M7, requires vae_pretrained_path
        - 'yolo'       — M8/M9, requires `ultralytics`
        - 'unet'       — M10/M11, requires `segmentation-models-pytorch`
        - 'yolo_bbox'  — M14 (structured features, optional)
        - 'sam2'       — M12, requires `transformers`, frozen only
        - 'dinov2'     — M13, requires `transformers`, frozen only
    """

    # ---------- ACT-specific (mirrors lerobot.policies.act.ACTConfig) ----------
    # If these defaults don't match your lerobot version, adjust here.
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
    latent_dim: int = 32  # size of z_act, NOT z_vis
    n_vae_encoder_layers: int = 4
    kl_weight: float = 10.0

    temporal_ensemble_coeff: Optional[float] = None

    # ---------- Visual encoder choice ----------
    # NOTE: draccus (lerobot's CLI parser) doesn't support Literal types
    # with many options, so we use plain str + runtime validation in
    # __post_init__ against VALID_ENCODERS below.
    encoder: str = "resnet18"

    # Unified output dimension — projector maps every encoder to this
    projector_dim: int = 256
    projector_hidden_dim: int = 256

    freeze_encoder: bool = False

    # ---------- Encoder-specific hyperparams ----------
    # ResNet baseline (Family A)
    use_linear_bottleneck: bool = False  # M1 control variant
    bottleneck_dim: int = 32

    # VAE family (Family B)
    vae_latent_dim: int = 32
    vae_pretrained_path: Optional[str] = None
    vae_beta: float = 1.0
    vqvae_codebook_size: int = 512
    vqvae_grid_size: int = 4

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
    # If True, encoder preserves spatial structure and passes a sequence of
    # tokens to the transformer (like standard ACT). If False, encoder
    # produces a single token vector. Default True for fair comparison.
    use_spatial_tokens: bool = True

    # ---------- Optimizer / scheduler ----------
    optimizer_lr: float = 1e-5
    optimizer_lr_backbone: float = 1e-5
    optimizer_weight_decay: float = 1e-4

    # ---------- Normalization ----------
    normalization_mapping: dict = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # ============================================================
    #               Validation
    # ============================================================

    def __post_init__(self):
        super().__post_init__()

        # Validate encoder choice (replaces the type system check we lost
        # when switching from Literal[...] to plain str for draccus compat).
        if self.encoder not in VALID_ENCODERS:
            raise ValueError(
                f"encoder='{self.encoder}' is invalid. "
                f"Choose one of: {sorted(VALID_ENCODERS)}"
            )

        # VAE family requires pretrained weights
        if self.encoder in ("vae", "beta_vae", "vqvae"):
            if self.vae_pretrained_path is None:
                raise ValueError(
                    f"encoder='{self.encoder}' requires vae_pretrained_path. "
                    "Pretrain first:\n"
                    f"    pretrain-visual-encoder --encoder-type={self.encoder} "
                    "--dataset-repo-id=... --output-path=..."
                )

        # Linear bottleneck only makes sense for resnet18
        if self.use_linear_bottleneck and self.encoder != "resnet18":
            raise ValueError(
                "use_linear_bottleneck=True is only valid for encoder='resnet18'"
            )

        # Foundation models are expensive — finetune is impractical with small datasets
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
        # Match standard lerobot ACTConfig — return None for a constant LR.
        # This ensures M0 (resnet18 baseline) is fairly comparable to stock ACT,
        # since the only architectural difference is then the visual encoder
        # path (not the optimization schedule).
        return None

    # ============================================================
    #               Delta indices (required by PreTrainedConfig)
    # ============================================================

    @property
    def observation_delta_indices(self) -> None:
        # Match stock ACT: return None for plain non-temporal observations.
        # This ensures batch[OBS_STATE] arrives as (B, state_dim) — flat 2D,
        # the same shape the standard ACT VAE encoder expects. If we returned
        # a list (even [0]) the dataloader would add a temporal dim and the
        # VAE encoder's cat over [cls, state, action] would fail with rank
        # mismatch (state 3D vs action 3D, but cls 3D — same shape needed).
        return None

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
