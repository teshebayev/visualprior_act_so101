#!/usr/bin/env python
"""Pre-fetch & smoke-test Family C/D encoders (YOLO / U-Net / SAM2 / DINOv2).

Unlike the VAE family, these encoders ship with pretrained weights from
their original authors:

    YOLO backbone / bbox  →  COCO detection (Ultralytics)
    U-Net                 →  ImageNet (segmentation-models-pytorch)
    SAM2                  →  SA-1B (Meta, via HF transformers)
    DINOv2                →  LVD-142M self-supervised (Meta, via HF)

You don't "train" them on SO-101 data — they're loaded as frozen feature
extractors. What you DO want before policy training is:

    1. Download the weights once (avoids hanging mid-training)
    2. Verify they instantiate and produce sane output shapes on YOUR machine
       (some encoders fail at import time on certain CUDA/transformers combos)
    3. Cache them in HF / Ultralytics cache so all subsequent runs are instant
    4. Optionally push a snapshot to your HF Hub for reproducibility on
       other machines / collaborators

This script does all four. Run it once per machine before launching policy
training. Total time: ~5-10 min depending on bandwidth (SAM2 is the big one).

Usage:
    # Prefetch all encoders, smoke-test, report shapes
    python scripts/prefetch_pretrained_encoders.py --all

    # Specific encoders only
    python scripts/prefetch_pretrained_encoders.py --encoders yolo unet dinov2

    # Snapshot a foundation model to your HF Hub (mirror for repro)
    python scripts/prefetch_pretrained_encoders.py --encoders dinov2 \\
        --mirror-to-hub --hub-user=your_hf_username
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

import torch

# ============================================================
#               Smoke-test runner: shared infrastructure
# ============================================================


def _print_header(label: str):
    print()
    print("=" * 78)
    print(f"  {label}")
    print("=" * 78)


def _try_smoke_test(name: str, build_fn, batch_size: int = 2) -> dict:
    """Build encoder, push a random [0, 1] batch through, report shape.

    Returns a dict with keys: name, ok, output_dim, num_spatial_tokens,
    output_shape, elapsed_s, error (if any).
    """
    info = {"name": name, "ok": False, "error": None}
    t0 = time.time()
    try:
        enc = build_fn()
        enc.eval()
        x = torch.rand(batch_size, 3, 224, 224)
        with torch.no_grad():
            y = enc(x)
        info.update(
            ok=True,
            output_dim=int(enc.output_dim),
            num_spatial_tokens=int(enc.num_spatial_tokens),
            output_shape=tuple(y.shape),
            output_min=float(y.min()),
            output_max=float(y.max()),
            n_params=sum(p.numel() for p in enc.parameters()),
        )
        # Try to find where it was cached on disk
        info["cache_hint"] = _guess_cache_path(name)
    except Exception as e:
        info["error"] = f"{type(e).__name__}: {e}"
        info["traceback"] = traceback.format_exc()
    info["elapsed_s"] = round(time.time() - t0, 1)
    return info


def _guess_cache_path(name: str) -> str:
    """Best-effort path hint where weights got cached."""
    home = Path.home()
    if "yolo" in name:
        # Ultralytics caches in CWD or ~/.config/Ultralytics — varies by version
        for candidate in [Path("yolov8n.pt"), home / ".config" / "Ultralytics"]:
            if candidate.exists():
                return str(candidate)
        return "current working directory (yolov8n.pt) or ~/.config/Ultralytics"
    if "unet" in name:
        return str(home / ".cache" / "torch" / "hub" / "checkpoints")
    if "sam2" in name or "dinov2" in name:
        return str(home / ".cache" / "huggingface" / "hub")
    return "unknown"


# ============================================================
#               Per-encoder build functions
# ============================================================


def build_yolo_backbone(model_name: str = "yolov8n", feature_level: int = 4):
    from lerobot_policy_visualprior_act.encoders.yolo_encoder import (
        YOLOBackboneEncoder,
    )
    return YOLOBackboneEncoder(
        model_name=model_name, feature_level=feature_level
    )


def build_yolo_bbox(model_name: str = "yolov8n", topk: int = 5):
    from lerobot_policy_visualprior_act.encoders.yolo_encoder import (
        YOLOBBoxEncoder,
    )
    return YOLOBBoxEncoder(model_name=model_name, topk=topk)


def build_unet(encoder_name: str = "resnet34", pretrained: str = "imagenet"):
    from lerobot_policy_visualprior_act.encoders.unet_encoder import UNetEncoder
    return UNetEncoder(encoder_name=encoder_name, pretrained=pretrained)


def build_sam2(model_name: str = "facebook/sam2-hiera-tiny"):
    from lerobot_policy_visualprior_act.encoders.sam2_encoder import SAM2Encoder
    return SAM2Encoder(model_name=model_name)


def build_dinov2(model_name: str = "facebook/dinov2-small"):
    from lerobot_policy_visualprior_act.encoders.dinov2_encoder import (
        DINOv2Encoder,
    )
    return DINOv2Encoder(model_name=model_name)


ENCODER_BUILDERS = {
    "yolo": ("YOLO backbone (yolov8n, P4)", build_yolo_backbone),
    "yolo_bbox": ("YOLO bbox detector (yolov8n, topk=5)", build_yolo_bbox),
    "unet": ("U-Net encoder (resnet34, ImageNet)", build_unet),
    "sam2": ("SAM2 (hiera-tiny)", build_sam2),
    "dinov2": ("DINOv2 (ViT-S/14)", build_dinov2),
}


# ============================================================
#               Optional: mirror foundation models to your hub
# ============================================================


def _mirror_hf_repo(source_repo: str, target_repo: str, private: bool = False):
    """Clone source_repo from HF and push as target_repo under your account.

    This is useful for reproducibility — if Meta deprecates a model, your
    mirror still works. Only mirrors transformers-hosted models (SAM2 /
    DINOv2), not Ultralytics weights (those have their own URL).
    """
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as e:
        raise ImportError(
            "Mirroring requires huggingface_hub. Install:\n"
            "    pip install huggingface_hub"
        ) from e

    print(f"  Downloading snapshot of {source_repo}...")
    local_dir = snapshot_download(repo_id=source_repo)
    print(f"    cached at {local_dir}")

    api = HfApi()
    print(f"  Creating mirror {target_repo} (private={private})")
    api.create_repo(repo_id=target_repo, private=private, exist_ok=True)

    print(f"  Uploading folder to {target_repo}...")
    api.upload_folder(
        folder_path=local_dir,
        repo_id=target_repo,
        commit_message=f"Mirror of {source_repo}",
    )
    print(f"  ✔ Mirrored to https://huggingface.co/{target_repo}")


# ============================================================
#               Main
# ============================================================


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--all", action="store_true",
        help="Prefetch all Family C/D encoders.",
    )
    p.add_argument(
        "--encoders",
        nargs="+",
        choices=list(ENCODER_BUILDERS.keys()),
        default=[],
        help="Subset of encoders to prefetch.",
    )
    p.add_argument(
        "--report",
        default="./pretrained/family_cd_report.json",
        help="Path to write JSON report with shapes / cache locations / errors.",
    )

    # Mirror options (only meaningful for HF-hosted models)
    p.add_argument(
        "--mirror-to-hub", action="store_true",
        help="After prefetch, push a snapshot of HF models to your account.",
    )
    p.add_argument(
        "--hub-user", default=None,
        help="HF username for mirroring (required if --mirror-to-hub).",
    )
    p.add_argument(
        "--hub-private", action="store_true",
        help="Create mirror repos as private.",
    )

    # Custom model names (advanced)
    p.add_argument("--yolo-model", default="yolov8n",
                   help="Ultralytics model name. Other options: yolov8s, yolov8m, yolov8l")
    p.add_argument("--sam2-model", default="facebook/sam2-hiera-tiny",
                   help="HF repo for SAM2. Other options: facebook/sam2-hiera-small, ...-base, ...-large")
    p.add_argument("--dinov2-model", default="facebook/dinov2-small",
                   help="HF repo for DINOv2. Other options: facebook/dinov2-base, ...-large, ...-giant")

    args = p.parse_args()

    if not args.all and not args.encoders:
        p.error("Specify --all or --encoders <name> [<name> ...]")
    if args.mirror_to_hub and not args.hub_user:
        p.error("--mirror-to-hub requires --hub-user")

    to_run = list(ENCODER_BUILDERS.keys()) if args.all else args.encoders

    print(f"Prefetching {len(to_run)} encoder(s): {to_run}")
    print(f"This downloads weights and runs a smoke-test forward pass.\n")

    reports = []
    for name in to_run:
        label, base_builder = ENCODER_BUILDERS[name]
        _print_header(f"[{name}] {label}")

        # Inject custom model name if provided
        if name == "yolo":
            builder = lambda: build_yolo_backbone(model_name=args.yolo_model)
        elif name == "yolo_bbox":
            builder = lambda: build_yolo_bbox(model_name=args.yolo_model)
        elif name == "unet":
            builder = build_unet
        elif name == "sam2":
            builder = lambda: build_sam2(model_name=args.sam2_model)
        elif name == "dinov2":
            builder = lambda: build_dinov2(model_name=args.dinov2_model)
        else:
            builder = base_builder

        info = _try_smoke_test(name, builder)

        if info["ok"]:
            print(f"  ✔ output shape : {info['output_shape']}")
            print(f"    output_dim   : {info['output_dim']}")
            print(f"    num_tokens   : {info['num_spatial_tokens']}")
            print(f"    val range    : [{info['output_min']:.3f}, {info['output_max']:.3f}]")
            print(f"    params       : {info['n_params']:,}")
            print(f"    cached       : {info['cache_hint']}")
            print(f"    elapsed      : {info['elapsed_s']}s")
        else:
            print(f"  ✗ FAILED: {info['error']}")
            print(f"    elapsed     : {info['elapsed_s']}s")
            print(f"    traceback (first 10 lines):")
            for line in info["traceback"].splitlines()[:10]:
                print(f"      {line}")

        reports.append(info)

    # ----- Optional mirroring to HF -----
    if args.mirror_to_hub:
        _print_header("Mirroring foundation models to your HF Hub")
        mirrorable = {
            "sam2": (args.sam2_model, f"{args.hub_user}/so101-mirror-sam2-hiera-tiny"),
            "dinov2": (args.dinov2_model, f"{args.hub_user}/so101-mirror-dinov2-small"),
        }
        for name, (src, dst) in mirrorable.items():
            if name not in to_run:
                continue
            ok_entry = next((r for r in reports if r["name"] == name and r["ok"]), None)
            if not ok_entry:
                print(f"  [{name}] skipped (prefetch failed, see above)")
                continue
            print(f"\n  [{name}] {src}  →  {dst}")
            try:
                _mirror_hf_repo(src, dst, private=args.hub_private)
            except Exception as e:
                print(f"  ✗ mirror failed: {type(e).__name__}: {e}")

    # ----- Write JSON report -----
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    # Strip traceback from JSON (keep it readable)
    clean = []
    for r in reports:
        c = {k: v for k, v in r.items() if k != "traceback"}
        # Tuple -> list for JSON
        if "output_shape" in c:
            c["output_shape"] = list(c["output_shape"])
        clean.append(c)
    report_path.write_text(json.dumps(clean, indent=2))

    print()
    print("=" * 78)
    print(f"  Report written to {report_path}")
    print("=" * 78)

    n_ok = sum(1 for r in reports if r["ok"])
    n_fail = len(reports) - n_ok
    print(f"\n  {n_ok} encoder(s) OK, {n_fail} failed")
    if n_fail:
        print("  → Check the failed encoders above. Common causes:")
        print("    - Missing optional deps (pip install -e \".[yolo,unet,foundation]\")")
        print("    - transformers/torch version mismatch (esp. for SAM2)")
        print("    - No internet on the worker / firewall blocking HF or Ultralytics")
        return 1

    print("\n  Next step: launch policy training")
    print("    bash scripts/train_family_cd.sh your_org/so101_pickplace")
    return 0


if __name__ == "__main__":
    sys.exit(main())
