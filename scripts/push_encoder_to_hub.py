#!/usr/bin/env python
"""Push an already-trained visual encoder to the Hugging Face Hub.

For weights produced by an OLD pretraining run that doesn't have the
.config.json sidecar, you can pass --encoder-type / --latent-dim / etc.
manually and the script will synthesize the config for you.

Usage:
    python scripts/push_encoder_to_hub.py \\
        --weights=./pretrained/vqvae_c512_g7_d32.safetensors \\
        --repo-id=your_user/so101-vqvae-c512-d32

    # If you don't have the .config.json sidecar, provide arch info:
    python scripts/push_encoder_to_hub.py \\
        --weights=./pretrained/vae_d32.safetensors \\
        --repo-id=your_user/so101-vae-d32 \\
        --encoder-type=vae --latent-dim=32 --spatial=true
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from lerobot_policy_visualprior_act.pretraining.hub_utils import (
    push_encoder_to_hub,
)


def _truthy(s: str) -> bool:
    return s.lower() in ("1", "true", "yes", "y", "t")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True, help="Path to .safetensors weights")
    p.add_argument("--repo-id", required=True, help="HF repo id (user/repo)")
    p.add_argument("--private", action="store_true")
    p.add_argument(
        "--config",
        default=None,
        help=(
            "Optional path to .config.json sidecar. If omitted, the script "
            "looks for <weights>.config.json next to the weights file."
        ),
    )
    # Manual override fields — useful if .config.json doesn't exist (legacy runs).
    p.add_argument("--encoder-type", choices=["vae", "beta_vae", "vqvae"])
    p.add_argument("--latent-dim", type=int)
    p.add_argument("--spatial", type=_truthy, default=None)
    p.add_argument("--codebook-size", type=int)
    p.add_argument("--grid-size", type=int)
    p.add_argument("--dataset-repo-id", default="<unspecified>")
    p.add_argument("--image-keys", default="observation.images.front")
    p.add_argument("--num-epochs", type=int, default=-1)
    p.add_argument("--beta", type=float, default=1.0)

    args = p.parse_args()
    weights = Path(args.weights)
    if not weights.exists():
        p.error(f"weights not found: {weights}")

    # Resolve config: either explicit, or sidecar next to weights, or synthesize.
    if args.config:
        cfg_path = Path(args.config)
        if not cfg_path.exists():
            p.error(f"--config path does not exist: {cfg_path}")
    else:
        cfg_path = weights.with_suffix(".config.json")

    if not cfg_path.exists():
        # Synthesize from CLI flags.
        if not args.encoder_type or not args.latent_dim:
            p.error(
                f"No config sidecar at {cfg_path}. "
                f"Pass --encoder-type and --latent-dim (at minimum) so a "
                f"config can be synthesized."
            )
        synth = {
            "encoder_type": args.encoder_type,
            "latent_dim": args.latent_dim,
            "spatial": args.spatial
            if args.encoder_type in ("vae", "beta_vae")
            else None,
            "codebook_size": args.codebook_size
            if args.encoder_type == "vqvae"
            else None,
            "grid_size": args.grid_size
            if args.encoder_type == "vqvae"
            else None,
            "beta": args.beta
            if args.encoder_type in ("vae", "beta_vae")
            else None,
            "dataset_repo_id": args.dataset_repo_id,
            "image_keys": [k.strip() for k in args.image_keys.split(",")],
            "image_size": 224,
            "num_epochs": args.num_epochs,
        }
        cfg_path = weights.with_suffix(".config.json")
        cfg_path.write_text(json.dumps(synth, indent=2))
        print(f"Synthesized config sidecar at {cfg_path}")

    push_encoder_to_hub(
        weights_path=weights,
        config_path=cfg_path,
        repo_id=args.repo_id,
        private=args.private,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
