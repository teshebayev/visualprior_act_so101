"""VisualPriorACTPolicy — ACT with replaceable visual prior encoder.

Architecture:
    image  -> VisualEncoder -> z_vis
    z_vis  -> Projector -> visual_tokens (B, N, dim_model)
    state  -> linear -> state_token
    visual_tokens + state_token -> ACT-transformer -> action chunk

The ACT internal action-CVAE (z_act) is preserved unchanged. Our z_vis is a
separate latent that conditions the same ACT decoder.

⚠ INTEGRATION NOTE ⚠

The most fragile part of this file is `_run_act_head`. Standard LeRobot ACT
runs images through its own ResNet backbone internally. We need to BYPASS
that and feed our pre-computed visual tokens directly into the transformer.

There are 3 approaches, ordered from least-invasive to most-robust:

  (A) Monkey-patch: subclass ACT model, replace backbone with Identity, and
      override the part of forward that uses backbone output. Works but tied
      to current lerobot ACT internals.

  (B) Subclass: same as A but cleaner — inherit from `lerobot.policies.act.ACT`
      and override only forward. Better, but still tied to internals.

  (C) Self-contained: implement minimal transformer encoder/decoder ourselves,
      no dependency on lerobot ACT internals. Most robust to lerobot updates,
      ~300 extra lines of code.

This file uses approach (B) by default with explicit fallback to (C).
You MUST review `_build_act_head` and `_run_act_head` against your installed
lerobot/policies/act/modeling_act.py before training. See docs/INTEGRATION_NOTES.md
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.normalize import Normalize, Unnormalize
from lerobot.utils.constants import ACTION

from .configuration_visualprior_act import VisualPriorACTConfig
from .encoders import build_encoder
from .utils.projector import VisualProjector


class VisualPriorACTPolicy(PreTrainedPolicy):
    """ACT with swappable visual prior encoder."""

    config_class = VisualPriorACTConfig
    name = "visualprior_act"

    def __init__(
        self,
        config: VisualPriorACTConfig,
        dataset_stats: Optional[dict[str, dict[str, Tensor]]] = None,
    ):
        super().__init__(config)
        config.validate_features()
        self.config = config

        # ---- Normalization (standard lerobot pattern) ----
        self.normalize_inputs = Normalize(
            config.input_features, config.normalization_mapping, dataset_stats
        )
        self.normalize_targets = Normalize(
            config.output_features, config.normalization_mapping, dataset_stats
        )
        self.unnormalize_outputs = Unnormalize(
            config.output_features, config.normalization_mapping, dataset_stats
        )

        # ---- Visual encoder ----
        self.visual_encoder = build_encoder(config)
        if config.freeze_encoder:
            self.visual_encoder.freeze()

        # ---- Projector: encoder_dim -> dim_model ----
        # Unified across all variants so that ACT head sees same dimension
        self.projector = VisualProjector(
            input_dim=self.visual_encoder.output_dim,
            output_dim=config.dim_model,
            hidden_dim=config.projector_hidden_dim,
            num_spatial_tokens=self.visual_encoder.num_spatial_tokens,
        )

        # ---- ACT transformer head ----
        # TODO(integration): see docstring at top of file
        self.act_head = self._build_act_head(config)

        # ---- Inference state ----
        self._action_queue: list[Tensor] = []

    # ============================================================
    #               ACT head construction (integration point)
    # ============================================================

    def _build_act_head(self, config: VisualPriorACTConfig) -> nn.Module:
        """Construct the ACT transformer head.

        TODO(integration): This function builds a wrapper around standard
        lerobot ACT, with the backbone replaced. You MUST verify against your
        installed lerobot version that:

        1. The import path below is correct.
        2. The ACTConfig parameters we pass actually exist in your version.
        3. The internal attribute names (`self.backbone`, etc.) match.

        If any of these fail, switch to the fallback in `_build_fallback_head`.
        """
        try:
            from lerobot.policies.act.modeling_act import ACT
            from lerobot.policies.act.configuration_act import ACTConfig
        except ImportError as e:
            raise ImportError(
                "Could not import lerobot ACT internals. "
                "Your lerobot version may have different module paths. "
                "See docs/INTEGRATION_NOTES.md for fallback options."
            ) from e

        # Build ACTConfig with parameters mirroring our config
        # TODO(integration): if ACTConfig signature differs in your version,
        # adjust the kwargs below.
        act_cfg = ACTConfig(
            input_features=config.input_features,
            output_features=config.output_features,
            chunk_size=config.chunk_size,
            n_action_steps=config.n_action_steps,
            dim_model=config.dim_model,
            dim_feedforward=config.dim_feedforward,
            n_encoder_layers=config.n_encoder_layers,
            n_decoder_layers=config.n_decoder_layers,
            n_heads=config.n_heads,
            dropout=config.dropout,
            pre_norm=config.pre_norm,
            use_vae=config.use_vae,
            latent_dim=config.latent_dim,
            n_vae_encoder_layers=config.n_vae_encoder_layers,
            kl_weight=config.kl_weight,
            temporal_ensemble_coeff=config.temporal_ensemble_coeff,
        )

        act = ACT(act_cfg)

        # Disable internal backbone — we feed visual tokens externally.
        # TODO(integration): verify that `backbone` is the correct attribute
        # name in your lerobot version. Common alternatives: `vision_encoder`,
        # `image_encoder`, `cnn_backbone`.
        if hasattr(act, "backbone"):
            act.backbone = nn.Identity()
        elif hasattr(act, "vision_encoder"):
            act.vision_encoder = nn.Identity()
        else:
            raise RuntimeError(
                "Could not locate ACT's visual backbone attribute. "
                "Inspect your lerobot ACT class and add the correct name here."
            )

        return act

    # ============================================================
    #               Forward / training
    # ============================================================

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Training forward. Returns (loss, metrics_dict)."""
        batch = self.normalize_inputs(batch)
        if ACTION in batch:
            batch = self.normalize_targets(batch)

        # 1. Extract images and run visual encoder
        image_keys = sorted(
            k for k in batch if k.startswith("observation.images")
        )
        if not image_keys:
            raise ValueError("No observation.images.* found in batch")

        # For now, use first camera only — multi-camera support is a TODO
        image = batch[image_keys[0]]

        # If input has time dimension (B, T, C, H, W), take last frame
        if image.dim() == 5:
            image = image[:, -1]

        z_vis = self.visual_encoder(image)  # (B, encoder_dim) or (B, N, encoder_dim)

        # 2. Project to ACT's dim_model
        visual_tokens = self.projector(z_vis)  # (B, N, dim_model)

        # 3. Run ACT head with custom visual tokens
        actions_hat, cvae_dist = self._run_act_head(batch, visual_tokens)

        # 4. Compute loss
        loss, loss_dict = self._compute_loss(batch, actions_hat, cvae_dist)
        return loss, loss_dict

    def _run_act_head(
        self, batch: dict[str, Tensor], visual_tokens: Tensor
    ) -> tuple[Tensor, Optional[tuple[Tensor, Tensor]]]:
        """Run ACT transformer with pre-computed visual tokens.

        TODO(integration): This is the most version-sensitive function.
        You will likely need to adapt it to your lerobot version.

        The standard ACT forward does roughly this:
            1. Run images through self.backbone -> cam_features (B, C, H', W')
            2. Project cam_features to dim_model and flatten spatial -> tokens
            3. Concat with state token, run through transformer encoder
            4. Run decoder, output action chunk
            5. (training only) Compute action-CVAE forward, return (mu, logvar)

        We need to inject our `visual_tokens` at step 2-3, bypassing step 1.

        Options to do this:

        (A) Add `__custom_visual_tokens` to batch dict, modify ACT.forward
            to check for this key (requires patching).

        (B) Call ACT's lower-level methods directly (encoder + decoder),
            constructing the input tokens ourselves.

        (C) Replace self.act_head with a self-contained transformer (see
            `_build_fallback_head`).

        The current implementation uses option B — calling internals directly.
        Inspect modeling_act.py to find the relevant methods.
        """
        # TODO(integration): The following is a SKELETON. You must adapt the
        # exact attribute and method names to match your lerobot version.
        #
        # Pseudo-code of what ACT.forward typically does after the backbone:
        #
        #   pos_embed = self.position_embedding(visual_tokens)
        #   state_token = self.state_proj(batch["observation.state"])
        #   # Optional: action-CVAE encoder for training
        #   if self.training and self.use_vae:
        #       mu, logvar = self.vae_encoder(state, actions)
        #       z_act = reparametrize(mu, logvar)
        #   else:
        #       z_act = zeros(...)
        #   z_act_token = self.latent_proj(z_act)
        #   encoder_in = cat([z_act_token, state_token, visual_tokens])
        #   encoder_out = self.encoder(encoder_in + pos_embed)
        #   decoder_out = self.decoder(query_embed, encoder_out)
        #   actions = self.action_head(decoder_out)
        #   return actions, (mu, logvar)
        #
        # Replace this skeleton with actual calls.

        raise NotImplementedError(
            "_run_act_head must be adapted to your lerobot version.\n"
            "See docs/INTEGRATION_NOTES.md for guidance.\n"
            "Steps to fix:\n"
            "1. Open lerobot/src/lerobot/policies/act/modeling_act.py\n"
            "2. Locate ACT.forward method\n"
            "3. Replace this NotImplementedError with calls to ACT internals\n"
            "   that bypass the (disabled) backbone and use `visual_tokens`."
        )

    def _compute_loss(
        self,
        batch: dict[str, Tensor],
        actions_hat: Tensor,
        cvae_dist: Optional[tuple[Tensor, Tensor]],
    ) -> tuple[Tensor, dict]:
        """L1 reconstruction loss + KL divergence for action-CVAE."""
        l1_loss = F.l1_loss(actions_hat, batch[ACTION], reduction="none")
        if "action_is_pad" in batch:
            l1_loss = l1_loss * (~batch["action_is_pad"]).unsqueeze(-1)
        l1_loss = l1_loss.mean()

        loss_dict = {"l1_loss": l1_loss.item()}
        total_loss = l1_loss

        if self.config.use_vae and cvae_dist is not None:
            mu, logvar = cvae_dist
            kl_loss = (
                -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(-1).mean()
            )
            total_loss = total_loss + self.config.kl_weight * kl_loss
            loss_dict["kl_loss"] = kl_loss.item()

        return total_loss, loss_dict

    # ============================================================
    #               Inference
    # ============================================================

    @torch.no_grad()
    def predict_action_chunk(
        self, batch: dict[str, Tensor], **kwargs
    ) -> Tensor:
        """Inference: predict full chunk of actions."""
        batch = self.normalize_inputs(batch)

        image_keys = sorted(
            k for k in batch if k.startswith("observation.images")
        )
        image = batch[image_keys[0]]
        if image.dim() == 5:
            image = image[:, -1]

        z_vis = self.visual_encoder(image)
        visual_tokens = self.projector(z_vis)

        actions_hat, _ = self._run_act_head(batch, visual_tokens)

        actions = self.unnormalize_outputs({"action": actions_hat})["action"]
        return actions

    def select_action(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        """Return single action, caching chunks for efficiency."""
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)
            # actions: (B, chunk_size, action_dim)
            self._action_queue = list(actions[0])
        return self._action_queue.pop(0).unsqueeze(0)

    def reset(self):
        """Clear action queue. Call between episodes."""
        self._action_queue = []

    # ============================================================
    #               Optimizer parameter groups
    # ============================================================

    def get_optim_params(self) -> list[dict]:
        """Build parameter groups with separate lr for encoder vs head."""
        param_groups = []

        encoder_params = self.visual_encoder.get_optim_params(
            lr=self.config.optimizer_lr,
            lr_backbone=self.config.optimizer_lr_backbone,
        )
        param_groups.extend(encoder_params)

        # Projector + ACT head: standard lr
        other = list(self.projector.parameters()) + list(
            self.act_head.parameters()
        )
        param_groups.append({"params": other, "lr": self.config.optimizer_lr})

        return param_groups
