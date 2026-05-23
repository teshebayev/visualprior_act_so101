"""Helpers for pushing trained visual encoders to the Hugging Face Hub.

The encoder is uploaded as a small repo containing:
    encoder.safetensors        — the actual weights
    encoder.config.json        — arch metadata (latent_dim, spatial, etc.)
    README.md                  — minimal model card

Anyone can then re-train a policy that uses this encoder by passing the
HF repo id to a downloader (or by manually downloading the .safetensors
into ./pretrained/ and pointing vae_pretrained_path at it).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional


_MODEL_CARD_TEMPLATE = """---
library_name: lerobot_policy_visualprior_act
tags:
  - robotics
  - so-101
  - lerobot
  - visual-encoder
  - {encoder_type}
license: apache-2.0
---

# Visual encoder ({encoder_type})

Pretrained visual encoder for the
[`lerobot_policy_visualprior_act`](https://github.com/teshebayev/visualprior_act_so101)
plugin. Used as a frozen vision frontend for ACT policies on the SO-101 arm.

## Architecture

| Field | Value |
|---|---|
| encoder_type | `{encoder_type}` |
| latent_dim | `{latent_dim}` |
| spatial | `{spatial}` |
| codebook_size | `{codebook_size}` |
| grid_size | `{grid_size}` |
| num_spatial_tokens | `{num_spatial_tokens}` |
| input_size | 224 × 224 |
| input_range | `[0, 1]` (no ImageNet normalization) |

Pretrained on dataset `{dataset_repo_id}` for `{num_epochs}` epochs across
image keys: `{image_keys}`.

## Usage

```bash
# Download the safetensors locally
huggingface-cli download {repo_id} encoder.safetensors --local-dir ./pretrained/

# Train a policy that uses it (frozen by default)
lerobot-train \\
    --policy.type=visualprior_act \\
    --policy.encoder={encoder_type} \\
    --policy.vae_pretrained_path=./pretrained/encoder.safetensors \\
    --policy.vae_latent_dim={latent_dim} \\
    --dataset.repo_id=your_org/your_so101_dataset \\
    --output_dir=outputs/policy_run
```

## Notes

- The encoder consumes images in `[0, 1]` range. The matching policy uses
  `NormalizationMode.IDENTITY` for `VISUAL` — do not apply ImageNet mean/std
  in your preprocessor or features will be out of distribution.
- For VAE / β-VAE, `forward()` returns the latent mean μ at inference (no
  sampling). For VQ-VAE, it returns quantized codes from the codebook.
"""


def _format_model_card(config: dict, repo_id: str, num_spatial_tokens: int) -> str:
    return _MODEL_CARD_TEMPLATE.format(
        encoder_type=config.get("encoder_type"),
        latent_dim=config.get("latent_dim"),
        spatial=config.get("spatial"),
        codebook_size=config.get("codebook_size"),
        grid_size=config.get("grid_size"),
        num_spatial_tokens=num_spatial_tokens,
        dataset_repo_id=config.get("dataset_repo_id"),
        image_keys=", ".join(config.get("image_keys", [])),
        num_epochs=config.get("num_epochs"),
        repo_id=repo_id,
    )


def _infer_num_spatial_tokens(config: dict) -> int:
    et = config.get("encoder_type")
    if et in ("vae", "beta_vae"):
        return 49 if config.get("spatial") else 1
    if et == "vqvae":
        g = int(config.get("grid_size") or 7)
        return g * g
    return -1


def push_encoder_to_hub(
    weights_path: Path,
    config_path: Path,
    repo_id: str,
    private: bool = False,
    console: Optional[object] = None,
) -> None:
    """Push encoder weights + config + auto-generated model card to HF Hub.

    Args:
        weights_path: path to encoder .safetensors file.
        config_path: path to .config.json sidecar produced by pretraining.
        repo_id: Hugging Face repo id (e.g. "your_user/so101-vqvae-d32").
        private: whether to create the repo private.
        console: optional rich Console for nicer logging.

    Requires `huggingface_hub` installed and the user logged in
    (`huggingface-cli login` or HF_TOKEN env var).
    """
    try:
        from huggingface_hub import HfApi
    except ImportError as e:
        raise ImportError(
            "Pushing to the Hub requires huggingface_hub. Install with:\n"
            "    pip install huggingface_hub"
        ) from e

    weights_path = Path(weights_path)
    config_path = Path(config_path)

    if not weights_path.exists():
        raise FileNotFoundError(f"weights_path does not exist: {weights_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"config_path does not exist: {config_path}")

    config = json.loads(config_path.read_text())
    n_tokens = _infer_num_spatial_tokens(config)
    card = _format_model_card(config, repo_id, n_tokens)

    api = HfApi()

    def _log(msg: str):
        if console is not None:
            console.print(msg)
        else:
            print(msg)

    _log(f"[cyan]Creating HF repo {repo_id} (private={private})[/cyan]")
    api.create_repo(repo_id=repo_id, private=private, exist_ok=True)

    _log(f"[cyan]Uploading {weights_path.name}[/cyan]")
    api.upload_file(
        path_or_fileobj=str(weights_path),
        path_in_repo="encoder.safetensors",
        repo_id=repo_id,
    )

    _log(f"[cyan]Uploading {config_path.name}[/cyan]")
    api.upload_file(
        path_or_fileobj=str(config_path),
        path_in_repo="encoder.config.json",
        repo_id=repo_id,
    )

    # Model card
    api.upload_file(
        path_or_fileobj=card.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
    )

    _log(f"[green]✔ Pushed to https://huggingface.co/{repo_id}[/green]")
