"""VisualPriorACTPolicy — ACT with replaceable visual prior encoder.

Architecture (multi-camera):
    for each camera in observation.images.*:
        image -> VisualEncoder -> z_vis -> Projector -> tokens (B, N, dim_model)
    concat tokens across cameras: (B, num_cameras * N, dim_model)
    state -> linear -> state_token
    [latent_token, state_token, *visual_tokens] -> ACT-transformer -> action chunk

The visual_encoder and projector weights are SHARED across cameras
(same convention as standard ACT shares its ResNet backbone). Each
(camera, position) pair gets a UNIQUE learnable positional embedding,
so the transformer can distinguish tokens from different ports/views.

The ACT internal action-CVAE (z_act) is preserved unchanged. Our z_vis is a
separate latent that conditions the same ACT decoder.

INTEGRATION STRATEGY (lerobot >= 0.4.0)
========================================

We use approach (B) from the original integration plan: build an ACT submodule
with no visual features in its ACTConfig — so the parent class never creates
`self.backbone`, `self.encoder_img_feat_input_proj`, or `self.encoder_cam_feat_pos_embed`.
We then call ACT's submodules (`vae_encoder`, `encoder`, `decoder`, `action_head`,
projection layers) directly from `_run_act_head`, injecting our pre-computed
`visual_tokens` where ACT would have inserted its own ResNet feature map.

For visual-token positional encoding we use a learnable 1D `nn.Embedding`
(`self.visual_token_pos_embed`), sized to `num_cameras * num_spatial_tokens`.
This avoids relying on ACT's 2D sinusoidal embedding which assumes a
CNN-style spatial grid — works for ResNet/U-Net but not for 1-token VAE
or top-K YOLO boxes.

Verified against lerobot main as of 2025-12 (commit `dfdc48a` and later).
"""

from __future__ import annotations

from typing import Any, Optional

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from lerobot.configs.types import FeatureType
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_STATE

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
        **kwargs: Any,
    ):
        """Args:
            config: Policy configuration.
            **kwargs: Ignored. `make_policy` in lerobot passes `dataset_stats`
                here, but normalization is now handled by the external
                processor pipeline (see processor_visualprior_act.py), not
                inside the policy. Accepting kwargs keeps us compatible with
                the factory's calling convention.
        """
        super().__init__(config)
        config.validate_features()
        self.config = config

        # ---- Normalization ----
        # In lerobot >= 0.4.0 normalization lives in the external
        # DataProcessorPipeline (NormalizerProcessorStep / UnnormalizerProcessorStep),
        # NOT inside the policy. See processor_visualprior_act.py.
        # The batch arriving at forward() is already normalized; the action
        # tensor we return is normalized and gets un-normalized by the
        # post-processor pipeline.

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
        # Built with VISUAL features stripped from ACTConfig, so the parent
        # ACT class doesn't create its own backbone or 2D image pos embedding.
        # We feed visual tokens into ACT's encoder ourselves in _run_act_head.
        self.act_head = self._build_act_head(config)

        # ---- Visual-token positional embedding ----
        # Learnable 1D positional encoding for our visual tokens. Sized to
        # num_cameras × num_spatial_tokens — every (camera, position) pair gets
        # a UNIQUE learnable embedding, so the transformer can distinguish
        # tokens from different cameras even when they share spatial position.
        # The visual_encoder + projector are SHARED across cameras (same as
        # standard ACT shares its ResNet backbone across cameras).
        self.num_cameras = max(len(config.image_features), 1)
        self.visual_token_pos_embed = nn.Embedding(
            self.num_cameras * self.visual_encoder.num_spatial_tokens,
            config.dim_model,
        )

        # ---- Inference state ----
        self._action_queue: list[Tensor] = []

    # ============================================================
    #               ACT head construction (integration point)
    # ============================================================

    def _build_act_head(self, config: VisualPriorACTConfig) -> nn.Module:
        """Construct the ACT transformer head.

        We build ACT with a *stripped* ACTConfig — input_features filtered to
        remove all VISUAL features. With no image features ACT.__init__ skips
        creating `self.backbone`, `self.encoder_img_feat_input_proj`, and
        `self.encoder_cam_feat_pos_embed`. The transformer encoder, decoder,
        VAE encoder, projection layers, and action head are all created
        normally — we reuse them from `_run_act_head`.

        Verified against ACT class in lerobot/policies/act/modeling_act.py
        (lerobot main, late 2025).
        """
        from lerobot.policies.act.configuration_act import ACTConfig
        from lerobot.policies.act.modeling_act import ACT

        # Strip VISUAL features so ACT doesn't build its own image processing path.
        input_features_no_visual = {
            k: v
            for k, v in config.input_features.items()
            if v.type is not FeatureType.VISUAL
        }

        act_cfg = ACTConfig(
            input_features=input_features_no_visual,
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

        return ACT(act_cfg)

    # ============================================================
    #               Forward / training
    # ============================================================

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Training forward. Returns (loss, metrics_dict).

        Note: batch has already been normalized by the pre-processor pipeline
        (NormalizerProcessorStep). The action tensor returned to the loss is
        still in the normalized space — same as standard ACT in lerobot >= 0.4.0.
        """

        # 1. Run visual encoder on every camera (shared weights across cameras)
        visual_tokens = self._encode_all_cameras(batch)

        # 2. Run ACT head with our visual tokens
        actions_hat, cvae_dist = self._run_act_head(batch, visual_tokens)

        # 3. Compute loss
        loss, loss_dict = self._compute_loss(batch, actions_hat, cvae_dist)
        return loss, loss_dict

    def _encode_all_cameras(self, batch: dict[str, Tensor]) -> Tensor:
        """Run visual encoder on every observation.images.* key in batch.

        visual_encoder and projector are SHARED across cameras (same weights),
        matching how standard ACT shares its ResNet backbone. Tokens from
        different cameras are concatenated along the sequence dim.

        Frames are resized to 224x224 here, on the same device as the input
        (GPU during training). This matches the input size every encoder in
        this plugin expects (VAE family backbone, ResNet, U-Net, SAM2, DINOv2,
        YOLO all assume 224x224).

        Camera keys are read from `config.image_features` — NOT from string
        prefixes — because the batch also contains `*_is_pad` masks that
        share the `observation.images.` prefix but are not images.

        Returns:
            (B, num_cameras * num_spatial_tokens, dim_model) — concatenated
            visual tokens, ordered by sorted camera key.
        """
        image_keys = sorted(self.config.image_features.keys())
        if not image_keys:
            raise ValueError("No image_features registered in config")

        all_tokens: list[Tensor] = []
        for image_key in image_keys:
            image = batch[image_key]
            original_shape = tuple(image.shape)

            # Normalize image to (B, C, H, W) regardless of input rank
            if image.dim() == 5:
                # (B, T, C, H, W) — temporal dim, take last frame
                image = image[:, -1]
            elif image.dim() == 3:
                # (C, H, W) — missing batch dim, add it
                image = image.unsqueeze(0)
            elif image.dim() != 4:
                raise ValueError(
                    f"Cannot interpret image at key '{image_key}' with "
                    f"shape {original_shape}. Expected one of: "
                    f"(C,H,W), (B,C,H,W), or (B,T,C,H,W). "
                    f"This often means the NormalizerProcessorStep or some "
                    f"other step in the pipeline has reshaped the tensor."
                )

            # Sanity: at this point image must be (B, C, H, W)
            if image.shape[-2] < 8 or image.shape[-1] < 8:
                raise ValueError(
                    f"Image at key '{image_key}' has tiny spatial dims "
                    f"{image.shape[-2:]} (original shape: {original_shape}). "
                    f"Something upstream broke the tensor."
                )

            # Resize to 224x224 if needed (lazily — skip if already correct)
            if image.shape[-2:] != (224, 224):
                image = F.interpolate(
                    image,
                    size=(224, 224),
                    mode="bilinear",
                    align_corners=False,
                )
            z_vis = self.visual_encoder(image)  # (B, D) or (B, N, D)
            tokens = self.projector(z_vis)  # (B, N, dim_model) — projector guarantees seq dim
            all_tokens.append(tokens)

        # Concatenate along sequence dim: (B, num_cameras * N, dim_model)
        return torch.cat(all_tokens, dim=1)

    def _run_act_head(
        self, batch: dict[str, Tensor], visual_tokens: Tensor
    ) -> tuple[Tensor, Optional[tuple[Tensor, Tensor]]]:
        """Run ACT transformer with pre-computed visual tokens.

        This mirrors `ACT.forward` from lerobot/policies/act/modeling_act.py
        but with the backbone+image-projection block replaced by our
        externally-computed `visual_tokens` (B, N, dim_model).

        Tensor convention: ACT uses sequence-first (S, B, D) for transformer
        inputs. We follow the same convention here.

        Args:
            batch: dict with at least observation.state, and (training only)
                action + action_is_pad.
            visual_tokens: (B, N, dim_model) — already projected to ACT's
                hidden dimension by VisualProjector.

        Returns:
            actions_hat: (B, chunk_size, action_dim)
            cvae_dist: (mu, log_sigma_x2) if action-CVAE is active during
                training, otherwise None. log_sigma_x2 == log(sigma^2) (= logvar)
                — name matches ACT's internal naming.
        """
        act = self.act_head
        cfg = act.config
        batch_size = visual_tokens.shape[0]
        num_visual_tokens = visual_tokens.shape[1]
        device = visual_tokens.device

        # ----------------------------------------------------------------
        # 1. Action-CVAE encoder (z_act) — training only, when use_vae=True
        #    Identical to ACT.forward lines ~407-451.
        # ----------------------------------------------------------------
        if cfg.use_vae and self.training and ACTION in batch:
            cls_embed = einops.repeat(
                act.vae_encoder_cls_embed.weight, "1 d -> b 1 d", b=batch_size
            )  # (B, 1, D)

            vae_encoder_input = [cls_embed]
            if cfg.robot_state_feature:
                robot_state_embed = act.vae_encoder_robot_state_input_proj(
                    batch[OBS_STATE]
                ).unsqueeze(1)  # (B, 1, D)
                vae_encoder_input.append(robot_state_embed)

            action_embed = act.vae_encoder_action_input_proj(
                batch[ACTION]
            )  # (B, S, D)
            vae_encoder_input.append(action_embed)

            vae_encoder_input = torch.cat(vae_encoder_input, dim=1)  # (B, S+{1,2}, D)

            # Fixed sinusoidal positional embedding registered on ACT as a buffer.
            pos_embed = act.vae_encoder_pos_enc.clone().detach()  # (1, S+{1,2}, D)

            # Key padding mask. cls + (robot_state) tokens are never padding.
            n_prefix = 2 if cfg.robot_state_feature else 1
            cls_joint_is_pad = torch.full(
                (batch_size, n_prefix), False, device=device
            )
            key_padding_mask = torch.cat(
                [cls_joint_is_pad, batch["action_is_pad"]], dim=1
            )  # (B, n_prefix + S)

            # ACTEncoder takes (S, B, D).
            cls_token_out = act.vae_encoder(
                vae_encoder_input.permute(1, 0, 2),
                pos_embed=pos_embed.permute(1, 0, 2),
                key_padding_mask=key_padding_mask,
            )[0]  # (B, D) — first (cls) token

            latent_pdf_params = act.vae_encoder_latent_output_proj(cls_token_out)
            mu = latent_pdf_params[:, : cfg.latent_dim]
            log_sigma_x2 = latent_pdf_params[:, cfg.latent_dim:]
            latent_sample = (
                mu + log_sigma_x2.div(2).exp() * torch.randn_like(mu)
            )
        else:
            mu = log_sigma_x2 = None
            latent_sample = torch.zeros(
                batch_size, cfg.latent_dim, dtype=torch.float32, device=device
            )

        # ----------------------------------------------------------------
        # 2. Build transformer-encoder input tokens in ACT order:
        #       [latent, (robot_state), (env_state), *visual_tokens]
        #    All accumulated as sequence-first tensors of shape (B, D)
        #    individually, then stacked into (S, B, D) at the end.
        # ----------------------------------------------------------------
        encoder_in_tokens: list[Tensor] = [
            act.encoder_latent_input_proj(latent_sample)  # (B, D)
        ]
        # ACT's encoder_1d_feature_pos_embed: nn.Embedding(n_1d_tokens, D)
        # where n_1d_tokens = 1 (latent) + (1 if robot_state) + (1 if env_state).
        # weight is (n_1d, D); unsqueeze(1) -> (n_1d, 1, D) for broadcast.
        encoder_in_pos_embed: list[Tensor] = list(
            act.encoder_1d_feature_pos_embed.weight.unsqueeze(1)
        )

        if cfg.robot_state_feature:
            encoder_in_tokens.append(
                act.encoder_robot_state_input_proj(batch[OBS_STATE])
            )
        if cfg.env_state_feature:
            encoder_in_tokens.append(
                act.encoder_env_state_input_proj(batch[OBS_ENV_STATE])
            )

        # Our visual tokens replace ACT's per-image backbone outputs.
        # visual_tokens: (B, N, D) -> (N, B, D)
        vis_tokens_seq_first = visual_tokens.permute(1, 0, 2)
        # Pos embed: (N, D) -> (N, 1, D), broadcasts to (N, B, D) when added.
        vis_pos_embed = self.visual_token_pos_embed.weight[
            :num_visual_tokens
        ].unsqueeze(1)

        encoder_in_tokens.extend(list(vis_tokens_seq_first))
        encoder_in_pos_embed.extend(list(vis_pos_embed))

        # Stack -> (S, B, D), (S, 1, D)
        encoder_in_tokens_t = torch.stack(encoder_in_tokens, dim=0)
        encoder_in_pos_embed_t = torch.stack(encoder_in_pos_embed, dim=0)

        # ----------------------------------------------------------------
        # 3. Transformer encoder + decoder + action head.
        # ----------------------------------------------------------------
        encoder_out = act.encoder(
            encoder_in_tokens_t, pos_embed=encoder_in_pos_embed_t
        )

        decoder_in = torch.zeros(
            (cfg.chunk_size, batch_size, cfg.dim_model),
            dtype=encoder_in_pos_embed_t.dtype,
            device=device,
        )
        decoder_out = act.decoder(
            decoder_in,
            encoder_out,
            encoder_pos_embed=encoder_in_pos_embed_t,
            decoder_pos_embed=act.decoder_pos_embed.weight.unsqueeze(1),
        )
        # Back to (B, chunk, D)
        decoder_out = decoder_out.transpose(0, 1)

        actions_hat = act.action_head(decoder_out)  # (B, chunk, action_dim)

        cvae_dist = (mu, log_sigma_x2) if mu is not None else None
        return actions_hat, cvae_dist

    def _compute_loss(
        self,
        batch: dict[str, Tensor],
        actions_hat: Tensor,
        cvae_dist: Optional[tuple[Tensor, Tensor]],
    ) -> tuple[Tensor, dict]:
        """L1 reconstruction loss + KL divergence for action-CVAE.

        Computed identically to lerobot ACTPolicy.forward: mask-aware average
        (sum over valid positions divided by valid-element count), so that
        M0 baseline reproduces standard ACT within numerical noise.
        """
        # Mask-aware L1, exactly as in ACTPolicy.forward.
        abs_err = F.l1_loss(batch[ACTION], actions_hat, reduction="none")
        valid_mask = ~batch["action_is_pad"].unsqueeze(-1)
        num_valid = valid_mask.sum() * abs_err.shape[-1]
        l1_loss = (abs_err * valid_mask).sum() / num_valid.clamp_min(1)

        loss_dict = {"l1_loss": l1_loss.item()}
        total_loss = l1_loss

        if self.config.use_vae and cvae_dist is not None:
            mu, log_sigma_x2 = cvae_dist
            # Sum over latent dim, mean over batch — identical to ACTPolicy.
            mean_kld = (
                -0.5
                * (1 + log_sigma_x2 - mu.pow(2) - log_sigma_x2.exp())
            ).sum(-1).mean()
            total_loss = total_loss + self.config.kl_weight * mean_kld
            loss_dict["kld_loss"] = mean_kld.item()

        return total_loss, loss_dict

    # ============================================================
    #               Inference
    # ============================================================

    @torch.no_grad()
    def predict_action_chunk(
        self, batch: dict[str, Tensor], **kwargs
    ) -> Tensor:
        """Inference: predict full chunk of actions.

        Note: batch is already normalized by the pre-processor pipeline.
        The returned actions are NORMALIZED — the post-processor pipeline
        (UnnormalizerProcessorStep) will un-normalize them.
        """
        visual_tokens = self._encode_all_cameras(batch)
        actions_hat, _ = self._run_act_head(batch, visual_tokens)
        return actions_hat

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

        # Projector + visual token pos embed + ACT head: standard lr
        other = (
            list(self.projector.parameters())
            + list(self.visual_token_pos_embed.parameters())
            + list(self.act_head.parameters())
        )
        param_groups.append({"params": other, "lr": self.config.optimizer_lr})

        return param_groups
