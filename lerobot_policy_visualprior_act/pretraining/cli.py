"""Standalone pretraining script for VAE-family encoders.

Trains VAE / β-VAE / VQ-VAE on frames from a LeRobotDataset, saves encoder
weights to a .safetensors file. Optionally pushes the result to the
Hugging Face Hub.

Usage:
    pretrain-visual-encoder \\
        --dataset-repo-id=your_org/so101_pickplace \\
        --encoder-type=vqvae \\
        --latent-dim=32 \\
        --codebook-size=512 \\
        --grid-size=7 \\
        --output-path=./pretrained/vqvae_c512_g7_d32.safetensors \\
        --num-epochs=50 \\
        --push-to-hub \\
        --hub-repo-id=your_hf_user/so101-vqvae-c512-d32

This script does NOT use lerobot-train — it's a separate pretraining stage
that must be done BEFORE training any policy variant.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from rich.console import Console
from safetensors.torch import save_file
from torch.utils.data import DataLoader, Dataset

from ..encoders.vae_family import VAEEncoder, VQVAEEncoder


# ============================================================
#               Dataset
# ============================================================


class FramesOnlyDataset(Dataset):
    """Extract individual frames from a LeRobotDataset for unsupervised pretraining."""

    def __init__(
        self,
        repo_id: str,
        image_keys: str | list[str] = "observation.images.front",
        image_size: int = 224,
    ):
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        except ImportError as e:
            raise ImportError(
                "LeRobotDataset import failed. Install lerobot:\n"
                "    cd ~/projects/lerobot && pip install -e ."
            ) from e

        self.ds = LeRobotDataset(repo_id)
        self.image_keys: list[str] = (
            [image_keys] if isinstance(image_keys, str) else list(image_keys)
        )
        self.image_size = image_size

        missing = [k for k in self.image_keys if k not in self.ds.features]
        if missing:
            available = [k for k in self.ds.features if "image" in k.lower()]
            raise ValueError(
                f"image_keys {missing} not in dataset. "
                f"Available image keys: {available}"
            )

    def __len__(self):
        return len(self.ds) * len(self.image_keys)

    def __getitem__(self, idx):
        n_keys = len(self.image_keys)
        ds_idx = idx // n_keys
        key = self.image_keys[idx % n_keys]

        sample = self.ds[ds_idx]
        img = sample[key]
        if img.dim() == 4:
            img = img[-1]  # take last frame if temporal
        if img.shape[-1] != self.image_size:
            img = F.interpolate(
                img.unsqueeze(0),
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        # Images from LeRobotDataset are in [0, 1] float. No normalization here,
        # matching the policy preprocessor convention (VISUAL=IDENTITY).
        return img


# ============================================================
#               Decoders (discarded after pretraining)
# ============================================================


def build_flat_vae_decoder(input_dim: int, output_channels: int = 3) -> nn.Module:
    """Decoder for spatial=False (flat) VAE. Input is (B, input_dim)."""
    return nn.Sequential(
        nn.Linear(input_dim, 256 * 7 * 7),
        nn.ReLU(inplace=True),
        nn.Unflatten(1, (256, 7, 7)),
        nn.ConvTranspose2d(256, 256, 4, 2, 1),  # 7 -> 14
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 14 -> 28
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(128, 64, 4, 2, 1),   # 28 -> 56
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(64, 32, 4, 2, 1),    # 56 -> 112
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(32, output_channels, 4, 2, 1),  # 112 -> 224
        nn.Sigmoid(),
    )


class SpatialVAEDecoder(nn.Module):
    """Decoder for spatial=True VAE. Input is (B, 49, latent_dim)."""

    def __init__(self, latent_dim: int, output_channels: int = 3):
        super().__init__()
        # 1x1 conv unprojects latent_dim -> 256
        self.unproject = nn.Conv2d(latent_dim, 256, 1)
        # Inverse of the backbone: 7 -> 14 -> 28 -> 56 -> 112 -> 224
        self.upsample = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, 4, 2, 1),  # 7 -> 14
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 14 -> 28
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),   # 28 -> 56
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),    # 56 -> 112
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, output_channels, 4, 2, 1),  # 112 -> 224
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B, 49, latent_dim) -> (B, latent_dim, 7, 7)
        b, n, c = z.shape
        g = int(round(n ** 0.5))
        assert g * g == n, f"Expected square spatial grid, got n={n}"
        z = z.permute(0, 2, 1).reshape(b, c, g, g)
        x = self.unproject(z)
        return self.upsample(x)


class VQVAEDecoder(nn.Module):
    """Decoder for VQ-VAE — input is (B, grid*grid, latent_dim)."""

    def __init__(self, latent_dim: int, grid_size: int, output_channels: int = 3):
        super().__init__()
        self.grid_size = grid_size
        self.unproject = nn.Conv2d(latent_dim, 256, 1)
        self.upsample = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, output_channels, 4, 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        b, n, c = z.shape
        g = int(round(n ** 0.5))
        z = z.permute(0, 2, 1).reshape(b, c, g, g)
        x = self.unproject(z)
        x = self.upsample(x)
        if x.shape[-1] != 224:
            x = F.interpolate(
                x, size=(224, 224), mode="bilinear", align_corners=False
            )
        return x


# ============================================================
#               Training
# ============================================================


def _build_encoder_and_decoder(args):
    """Construct encoder + paired decoder from CLI args."""
    if args.encoder_type in ("vae", "beta_vae"):
        encoder = VAEEncoder(
            latent_dim=args.latent_dim,
            spatial=not args.no_spatial,
            return_mode="sample",
        )
        if encoder.spatial:
            decoder = SpatialVAEDecoder(args.latent_dim)
        else:
            decoder = build_flat_vae_decoder(args.latent_dim)
    elif args.encoder_type == "vqvae":
        encoder = VQVAEEncoder(
            latent_dim=args.latent_dim,
            codebook_size=args.codebook_size,
            grid_size=args.grid_size,
        )
        decoder = VQVAEDecoder(args.latent_dim, args.grid_size)
    else:
        raise ValueError(args.encoder_type)
    return encoder, decoder


def _compute_loss(args, encoder, decoder, images):
    """Single-batch loss. Returns (loss, metrics_dict)."""
    if args.encoder_type in ("vae", "beta_vae"):
        mu, logvar = encoder.encode_with_dist(images)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        recon = decoder(z)
        recon_loss = F.mse_loss(recon, images)
        # KL: sum over feature dims, mean over all leading dims (batch + spatial).
        # `.sum(-1)` collapses latent_dim; `.mean()` averages the rest. For
        # spatial=True, this is mean over (B * 49); for spatial=False, mean over B.
        kl_loss = (
            -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(-1).mean()
        )
        loss = recon_loss + args.beta * kl_loss
        return loss, {"recon": recon_loss.item(), "kl": kl_loss.item()}

    elif args.encoder_type == "vqvae":
        z = encoder(images)  # (B, grid*grid, latent_dim)
        recon = decoder(z)
        loss = F.mse_loss(recon, images)
        return loss, {"recon": loss.item()}

    raise ValueError(args.encoder_type)


def _resolve_image_keys(args, console) -> list[str]:
    if args.all_cameras:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        peek = LeRobotDataset(args.dataset_repo_id)
        image_keys = sorted(
            k for k in peek.features if k.startswith("observation.images")
        )
        del peek
        if not image_keys:
            raise ValueError("--all-cameras: no observation.images.* in dataset")
        console.print(f"[cyan]--all-cameras → using {image_keys}[/cyan]")
        return image_keys
    return [k.strip() for k in args.image_key.split(",") if k.strip()]


def train(args):
    console = Console()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"[bold]Pretraining {args.encoder_type}[/bold] on {device}")

    image_keys = _resolve_image_keys(args, console)

    console.print(f"Loading dataset: {args.dataset_repo_id}")
    ds = FramesOnlyDataset(args.dataset_repo_id, image_keys=image_keys)
    console.print(
        f"  {len(ds)} frames available "
        f"({len(ds) // len(image_keys)} timesteps × {len(image_keys)} cameras)"
    )

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    encoder, decoder = _build_encoder_and_decoder(args)
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    optim = torch.optim.AdamW(
        list(encoder.parameters()) + list(decoder.parameters()), lr=args.lr
    )

    encoder.train()
    decoder.train()

    for epoch in range(args.num_epochs):
        batch_loss = 0.0
        n_batches = 0
        last_metrics: dict = {}

        for batch in loader:
            images = batch.to(device, non_blocking=True)
            loss, metrics = _compute_loss(args, encoder, decoder, images)

            optim.zero_grad()
            loss.backward()
            optim.step()

            batch_loss += loss.item()
            n_batches += 1
            last_metrics = metrics

        avg_loss = batch_loss / max(n_batches, 1)
        log_msg = (
            f"Epoch {epoch + 1}/{args.num_epochs}: "
            f"loss={avg_loss:.4f}  "
            + "  ".join(f"{k}={v:.4f}" for k, v in last_metrics.items())
        )
        if args.encoder_type == "vqvae":
            usage = encoder.get_codebook_usage()
            log_msg += f"  codebook_usage={usage:.1%}"
            if 0 < usage < 0.30:
                log_msg += " [yellow](collapse risk)[/yellow]"
        console.print(log_msg)

    # ----- Save -----
    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(encoder.state_dict(), str(out_path))
    console.print(f"\n[green]Saved encoder to {out_path}[/green]")

    # Write a small JSON sidecar so the encoder arch is self-describing.
    cfg_path = out_path.with_suffix(".config.json")
    cfg = {
        "encoder_type": args.encoder_type,
        "latent_dim": args.latent_dim,
        "spatial": (not args.no_spatial)
        if args.encoder_type in ("vae", "beta_vae")
        else None,
        "codebook_size": args.codebook_size
        if args.encoder_type == "vqvae"
        else None,
        "grid_size": args.grid_size if args.encoder_type == "vqvae" else None,
        "beta": args.beta if args.encoder_type in ("vae", "beta_vae") else None,
        "dataset_repo_id": args.dataset_repo_id,
        "image_keys": image_keys,
        "image_size": 224,
        "num_epochs": args.num_epochs,
    }
    cfg_path.write_text(json.dumps(cfg, indent=2))
    console.print(f"[green]Wrote config sidecar to {cfg_path}[/green]")

    # ----- Optional: push to Hugging Face Hub -----
    if args.push_to_hub:
        from .hub_utils import push_encoder_to_hub  # local import

        push_encoder_to_hub(
            weights_path=out_path,
            config_path=cfg_path,
            repo_id=args.hub_repo_id,
            private=args.hub_private,
            console=console,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Pretrain a VAE-family visual encoder on LeRobotDataset frames"
    )
    parser.add_argument("--dataset-repo-id", required=True)
    parser.add_argument(
        "--image-key",
        default="observation.images.front",
        help="Camera key, or comma-separated list of keys.",
    )
    parser.add_argument(
        "--all-cameras",
        action="store_true",
        help="Auto-discover all observation.images.* keys.",
    )
    parser.add_argument(
        "--encoder-type",
        choices=["vae", "beta_vae", "vqvae"],
        required=True,
    )
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--codebook-size", type=int, default=512)
    parser.add_argument(
        "--grid-size",
        type=int,
        default=7,
        help="VQ-VAE spatial grid. Default 7 matches backbone resolution.",
    )
    parser.add_argument(
        "--no-spatial",
        action="store_true",
        help=(
            "Use legacy flat (single-token) VAE bottleneck. "
            "Default is spatial=True (49 tokens). Only affects VAE / β-VAE."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output-path", required=True)

    # Hugging Face Hub upload
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Upload the trained encoder to the Hugging Face Hub.",
    )
    parser.add_argument(
        "--hub-repo-id",
        default=None,
        help="HF repo id, e.g. your_user/so101-vae-d32. Required if --push-to-hub.",
    )
    parser.add_argument(
        "--hub-private",
        action="store_true",
        help="Create the HF repo as private.",
    )

    args = parser.parse_args()

    if args.push_to_hub and not args.hub_repo_id:
        parser.error("--push-to-hub requires --hub-repo-id")

    train(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
