"""Standalone pretraining script for VAE-family encoders.

Trains VAE / β-VAE / VQ-VAE on frames from a LeRobotDataset, saves encoder
weights to a .safetensors file for later loading in policy training.

Usage:
    pretrain-visual-encoder \\
        --dataset-repo-id=your_org/so101_pickplace \\
        --encoder-type=vqvae \\
        --latent-dim=32 \\
        --codebook-size=512 \\
        --output-path=./pretrained/vqvae_c512_d32.safetensors \\
        --num-epochs=50

This script does NOT use lerobot-train — it's a separate pretraining stage
that must be done BEFORE training any policy variant.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from rich.console import Console
from rich.progress import Progress
from safetensors.torch import save_file
from torch.utils.data import DataLoader, Dataset

from ..encoders.vae_family import VAEEncoder, VQVAEEncoder


class FramesOnlyDataset(Dataset):
    """Extract individual frames from a LeRobotDataset for unsupervised pretraining.

    Supports multiple camera keys — if more than one key is given, each
    timestep produces num_keys items (one per camera), effectively
    multiplying training data without any architecture changes.
    """

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
        # Normalize to a list internally.
        self.image_keys: list[str] = (
            [image_keys] if isinstance(image_keys, str) else list(image_keys)
        )
        self.image_size = image_size

        # Validate every requested key exists in the dataset features.
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
        # Stretch dataset by num_cameras: every frame contributes once per camera.
        n_keys = len(self.image_keys)
        ds_idx = idx // n_keys
        key = self.image_keys[idx % n_keys]

        sample = self.ds[ds_idx]
        img = sample[key]
        # img typically (C, H, W) float [0, 1]
        if img.dim() == 4:
            img = img[-1]  # take last frame if temporal
        # Resize if needed
        if img.shape[-1] != self.image_size:
            img = F.interpolate(
                img.unsqueeze(0),
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        return img


def build_simple_decoder(input_dim: int, output_channels: int = 3) -> nn.Module:
    """Symmetric decoder for VAE pretraining. Discarded after."""
    return nn.Sequential(
        nn.Linear(input_dim, 256 * 7 * 7),
        nn.ReLU(inplace=True),
        nn.Unflatten(1, (256, 7, 7)),
        nn.ConvTranspose2d(256, 256, 4, 2, 1),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(256, 128, 4, 2, 1),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(128, 64, 4, 2, 1),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(64, 32, 4, 2, 1),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(32, output_channels, 4, 2, 1),
        nn.Sigmoid(),
    )


def build_vqvae_decoder(latent_dim: int, grid_size: int) -> nn.Module:
    """Spatial decoder for VQ-VAE — input is (B, grid*grid, latent_dim)."""

    class VQDecoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.unproject = nn.Conv2d(latent_dim, 256, 1)
            self.upsample = nn.Sequential(
                # Upsample from grid_size to 224
                # Approximate ratios depending on grid_size
                nn.ConvTranspose2d(256, 128, 4, 2, 1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(128, 64, 4, 2, 1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(64, 32, 4, 2, 1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(32, 16, 4, 2, 1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(16, 3, 4, 2, 1),
                nn.Sigmoid(),
            )

        def forward(self, z):
            # z: (B, grid*grid, latent_dim) -> (B, latent_dim, grid, grid)
            b, n, c = z.shape
            g = int(n**0.5)
            z = z.permute(0, 2, 1).reshape(b, c, g, g)
            x = self.unproject(z)
            x = self.upsample(x)
            # Resize to exactly 224x224 (may not match perfectly via ConvT)
            if x.shape[-1] != 224:
                x = F.interpolate(
                    x, size=(224, 224), mode="bilinear", align_corners=False
                )
            return x

    return VQDecoder()


def train(args):
    console = Console()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"[bold]Pretraining {args.encoder_type}[/bold] on {device}")

    # Resolve image keys: --all-cameras flag overrides --image-key; otherwise
    # treat --image-key as comma-separated list (single key still works).
    if args.all_cameras:
        # Auto-discover all observation.images.* keys from dataset metadata.
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        peek = LeRobotDataset(args.dataset_repo_id)
        image_keys = sorted(
            k for k in peek.features if k.startswith("observation.images")
        )
        del peek
        if not image_keys:
            raise ValueError("--all-cameras: no observation.images.* in dataset")
        console.print(f"[cyan]--all-cameras → using {image_keys}[/cyan]")
    else:
        image_keys = [k.strip() for k in args.image_key.split(",") if k.strip()]

    # Dataset
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

    # Model
    if args.encoder_type in ("vae", "beta_vae"):
        encoder = VAEEncoder(latent_dim=args.latent_dim, return_mode="sample")
        decoder = build_simple_decoder(args.latent_dim)
    elif args.encoder_type == "vqvae":
        encoder = VQVAEEncoder(
            latent_dim=args.latent_dim,
            codebook_size=args.codebook_size,
            grid_size=args.grid_size,
        )
        decoder = build_vqvae_decoder(args.latent_dim, args.grid_size)
    else:
        raise ValueError(args.encoder_type)

    encoder = encoder.to(device)
    decoder = decoder.to(device)

    # Optimizer
    optim = torch.optim.AdamW(
        list(encoder.parameters()) + list(decoder.parameters()), lr=args.lr
    )

    # Training loop
    encoder.train()
    decoder.train()

    with Progress() as progress:
        epoch_task = progress.add_task(
            "[cyan]Epochs", total=args.num_epochs
        )

        for epoch in range(args.num_epochs):
            batch_loss = 0.0
            n_batches = 0

            for batch in loader:
                images = batch.to(device, non_blocking=True)

                if args.encoder_type in ("vae", "beta_vae"):
                    mu, logvar = encoder.encode_with_dist(images)
                    std = torch.exp(0.5 * logvar)
                    z = mu + std * torch.randn_like(std)
                    recon = decoder(z)
                    recon_loss = F.mse_loss(recon, images)
                    kl_loss = (
                        -0.5
                        * (1 + logvar - mu.pow(2) - logvar.exp()).sum(-1).mean()
                    )
                    loss = recon_loss + args.beta * kl_loss
                    metrics = {
                        "recon": recon_loss.item(),
                        "kl": kl_loss.item(),
                    }

                elif args.encoder_type == "vqvae":
                    z = encoder(images)  # (B, grid*grid, latent_dim)
                    recon = decoder(z)
                    loss = F.mse_loss(recon, images)
                    metrics = {"recon": loss.item()}

                optim.zero_grad()
                loss.backward()
                optim.step()

                batch_loss += loss.item()
                n_batches += 1

            avg_loss = batch_loss / max(n_batches, 1)
            log_msg = f"Epoch {epoch + 1}/{args.num_epochs}: loss={avg_loss:.4f}"
            if args.encoder_type == "vqvae":
                usage = encoder.get_codebook_usage()
                log_msg += f"  codebook_usage={usage:.1%}"
                if usage > 0 and usage < 0.30:
                    log_msg += " [yellow](collapse risk)[/yellow]"
            console.print(log_msg)
            progress.update(epoch_task, advance=1)

    # Save encoder weights only (decoder discarded)
    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(encoder.state_dict(), str(out_path))
    console.print(f"\n[green]Saved encoder to {out_path}[/green]")


def main():
    parser = argparse.ArgumentParser(
        description="Pretrain a VAE-family visual encoder on LeRobotDataset frames"
    )
    parser.add_argument("--dataset-repo-id", required=True)
    parser.add_argument(
        "--image-key",
        default="observation.images.front",
        help=(
            "Camera key to use. Pass a single key (e.g. observation.images.camera1) "
            "or a comma-separated list to use multiple cameras: "
            "observation.images.camera1,observation.images.camera2"
        ),
    )
    parser.add_argument(
        "--all-cameras",
        action="store_true",
        help=(
            "Auto-discover all observation.images.* keys in the dataset and "
            "train on frames from every camera. Overrides --image-key. "
            "Recommended for datasets with multiple camera views — multiplies "
            "effective training data by num_cameras with no architecture changes."
        ),
    )
    parser.add_argument(
        "--encoder-type",
        choices=["vae", "beta_vae", "vqvae"],
        required=True,
    )
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--codebook-size", type=int, default=512)
    parser.add_argument("--grid-size", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output-path", required=True)
    args = parser.parse_args()

    train(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
