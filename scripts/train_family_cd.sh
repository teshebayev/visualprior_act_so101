#!/usr/bin/env bash
# train_family_cd.sh — train Family C (YOLO/U-Net) and Family D (SAM2/DINOv2) policies.
#
# These encoders ship pretrained from their authors; no separate pretrain stage.
# Run scripts/prefetch_pretrained_encoders.py first to download & cache weights.
#
# Usage:
#   ./train_family_cd.sh <dataset_repo_id> [output_dir]
#
# Optional env vars:
#   STEPS=100000              # training steps per config
#   BATCH_SIZE=8              # adjust based on VRAM (see notes below)
#   SEEDS="42 123 7"          # space-separated seeds (default: 42)
#   MODELS="M8 M10 M12 M13"   # space-separated subset to train
#   PUSH_TO_HUB=1
#   HUB_USER=your_hf_username # required if PUSH_TO_HUB=1
#   HUB_PRIVATE=1
#
# VRAM notes (rough, RTX 3090 / A100 reference, batch 8, chunk_size=100):
#   M8  yolo backbone (frozen)    ~  6 GB
#   M9  yolo backbone (finetune)  ~  9 GB
#   M10 unet (frozen)             ~  5 GB
#   M11 unet (finetune)           ~  8 GB
#   M12 sam2 hiera-tiny (frozen)  ~ 11 GB   ← can't finetune at this batch
#   M13 dinov2-small (frozen)     ~  7 GB
#
# Sequence vs scenes:
#   Family C/D pretrained encoders see ImageNet/COCO/SA-1B/LVD images. None
#   of them have seen SO-101 cube manipulation. Expect M0 (ResNet-18) to be
#   competitive or beat them unless your scene closely matches their domain.
#   That said, M12/M13 carry strong general priors and tend to generalize
#   better OOD (different lighting / background).

set -euo pipefail

DATASET_REPO_ID="${1:-}"
if [ -z "$DATASET_REPO_ID" ]; then
    echo "Usage: $0 <dataset_repo_id> [output_dir]"
    echo "Example: $0 your_org/so101_pickplace_v1"
    exit 1
fi

OUTPUT_BASE="${2:-./outputs}"

STEPS="${STEPS:-100000}"
BATCH_SIZE="${BATCH_SIZE:-8}"
SEEDS="${SEEDS:-42}"
MODELS="${MODELS:-M8 M9 M10 M11 M12 M13}"

mkdir -p "$OUTPUT_BASE"

# Push-to-hub helpers
HUB_FLAGS=()
if [ "${PUSH_TO_HUB:-0}" = "1" ]; then
    if [ -z "${HUB_USER:-}" ]; then
        echo "PUSH_TO_HUB=1 requires HUB_USER=your_hf_username"
        exit 1
    fi
    HUB_FLAGS+=(--policy.push_to_hub=true)
    if [ "${HUB_PRIVATE:-0}" = "1" ]; then
        HUB_FLAGS+=(--policy.private=true)
    fi
fi

echo "================================================================"
echo "  Family C/D training matrix"
echo "  Dataset    : $DATASET_REPO_ID"
echo "  Output dir : $OUTPUT_BASE"
echo "  Steps      : $STEPS"
echo "  Batch size : $BATCH_SIZE"
echo "  Seeds      : $SEEDS"
echo "  Models     : $MODELS"
echo "  Push HF    : ${PUSH_TO_HUB:-0}"
echo "================================================================"

# Sanity: did the user run prefetch_pretrained_encoders.py?
if [ ! -f "./pretrained/family_cd_report.json" ]; then
    echo
    echo "WARNING: ./pretrained/family_cd_report.json not found."
    echo "Recommended: run prefetch first to download weights and avoid"
    echo "stalling mid-training:"
    echo "    python scripts/prefetch_pretrained_encoders.py --all"
    echo
    echo "Continuing anyway in 5 seconds (Ctrl-C to abort)..."
    sleep 5
fi

# Helper: should we run this model?
should_run() {
    local id="$1"
    # Match whole-word against MODELS env var
    [[ " $MODELS " == *" $id "* ]]
}

# Helper: train a single config across all seeds
train_config() {
    local model_id="$1"
    shift
    local extra_args=("$@")

    for seed in $SEEDS; do
        local run_name="${model_id}_seed${seed}"
        local out_dir="$OUTPUT_BASE/$run_name"
        if [ -d "$out_dir" ]; then
            echo "  Skipping ${run_name} (output dir exists)"
            continue
        fi

        # Build push-hub repo id per-run if hub is on
        local hub_args=()
        if [ "${PUSH_TO_HUB:-0}" = "1" ]; then
            hub_args=(--policy.repo_id="${HUB_USER}/so101-policy-${model_id,,}-seed${seed}")
        fi

        echo
        echo ">>> Training ${run_name}"
        lerobot-train \
            --policy.type=visualprior_act \
            --dataset.repo_id="$DATASET_REPO_ID" \
            --wandb.enable=true \
            --output_dir="$out_dir" \
            --seed="$seed" \
            --steps="$STEPS" \
            --batch_size="$BATCH_SIZE" \
            "${extra_args[@]}" \
            "${HUB_FLAGS[@]}" "${hub_args[@]}"
    done
}

# ============================================================
#               Family C: YOLO backbone
# ============================================================

if should_run "M8"; then
    echo
    echo "=== M8: YOLO backbone (frozen) ==="
    train_config "M8_yolo_frozen" \
        --policy.encoder=yolo \
        --policy.yolo_model_name=yolov8n \
        --policy.yolo_feature_level=4 \
        --policy.freeze_encoder=true
fi

if should_run "M9"; then
    echo
    echo "=== M9: YOLO backbone (finetuned, lower lr_backbone) ==="
    # Backbone LR 10x lower than head — standard recipe for finetuning
    # ImageNet-style backbones.
    train_config "M9_yolo_finetune" \
        --policy.encoder=yolo \
        --policy.yolo_model_name=yolov8n \
        --policy.yolo_feature_level=4 \
        --policy.freeze_encoder=false \
        --policy.optimizer_lr=1e-4 \
        --policy.optimizer_lr_backbone=1e-5
fi

# ============================================================
#               Family C: U-Net
# ============================================================

if should_run "M10"; then
    echo
    echo "=== M10: U-Net encoder (frozen) ==="
    train_config "M10_unet_frozen" \
        --policy.encoder=unet \
        --policy.unet_encoder_name=resnet34 \
        --policy.freeze_encoder=true
fi

if should_run "M11"; then
    echo
    echo "=== M11: U-Net encoder (finetuned) ==="
    train_config "M11_unet_finetune" \
        --policy.encoder=unet \
        --policy.unet_encoder_name=resnet34 \
        --policy.freeze_encoder=false \
        --policy.optimizer_lr=1e-4 \
        --policy.optimizer_lr_backbone=1e-5
fi

# ============================================================
#               Family D: Foundation models (always frozen)
# ============================================================

if should_run "M12"; then
    echo
    echo "=== M12: SAM2 hiera-tiny (frozen, no finetune) ==="
    # freeze_encoder=true is enforced automatically in config __post_init__
    # for sam2/dinov2; passing it explicitly here for clarity.
    train_config "M12_sam2" \
        --policy.encoder=sam2 \
        --policy.sam2_model_name=facebook/sam2-hiera-tiny \
        --policy.freeze_encoder=true
fi

if should_run "M13"; then
    echo
    echo "=== M13: DINOv2 ViT-S/14 (frozen) ==="
    train_config "M13_dinov2" \
        --policy.encoder=dinov2 \
        --policy.dinov2_model_name=facebook/dinov2-small \
        --policy.freeze_encoder=true
fi

echo
echo "================================================================"
echo "  All Family C/D training complete. Outputs in $OUTPUT_BASE/"
echo "================================================================"
ls -la "$OUTPUT_BASE/" | head -30
