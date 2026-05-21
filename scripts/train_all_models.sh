#!/usr/bin/env bash
# train_all_models.sh
# Тренировка всей experimental matrix Phase 2 (M0-M13)
#
# Usage: ./train_all_models.sh your_org/so101_pickplace_v1
#
# Перед запуском убедись что:
#   1) Pretrained VAE-family weights существуют (запускал pretrain_all_vae_variants.sh)
#   2) docs/INTEGRATION_NOTES.md прочитан и _run_act_head реализован
#   3) Baseline ACT работает с твоими данными

set -euo pipefail

DATASET_REPO_ID="${1:-}"
if [ -z "$DATASET_REPO_ID" ]; then
    echo "Usage: $0 <dataset_repo_id>"
    exit 1
fi

OUTPUT_BASE="${2:-./outputs}"
PRETRAINED_DIR="${3:-./pretrained}"
SEEDS=(42 123 7)
STEPS="${STEPS:-100000}"
BATCH_SIZE="${BATCH_SIZE:-8}"

mkdir -p "$OUTPUT_BASE"

# Helper: train a single config across all seeds
train_config() {
    local model_id="$1"
    shift
    local extra_args=("$@")
    
    for seed in "${SEEDS[@]}"; do
        local out_dir="$OUTPUT_BASE/${model_id}_seed${seed}"
        if [ -d "$out_dir" ]; then
            echo "  Skipping ${model_id}_seed${seed} (already exists)"
            continue
        fi
        echo
        echo ">>> Training ${model_id} seed=${seed}"
        lerobot-train \
            --policy.type=visualprior_act \
            --dataset.repo_id="$DATASET_REPO_ID" \
            --output_dir="$out_dir" \
            --seed="$seed" \
            --steps="$STEPS" \
            --batch_size="$BATCH_SIZE" \
            "${extra_args[@]}"
    done
}

# ============================================================
#               Family A: Baseline + linear bottleneck
# ============================================================

echo "=== Family A: baselines ==="
train_config "M0_resnet_baseline" \
    --policy.encoder=resnet18

train_config "M1_linear_bottleneck" \
    --policy.encoder=resnet18 \
    --policy.use_linear_bottleneck=true \
    --policy.bottleneck_dim=32

# ============================================================
#               Family B: VAE / β-VAE / VQ-VAE
# ============================================================

echo "=== Family B: VAE-family ==="

# M2/M3: VAE frozen/finetuned
train_config "M2_vae_frozen" \
    --policy.encoder=vae \
    --policy.vae_pretrained_path="$PRETRAINED_DIR/vae_d32.safetensors" \
    --policy.freeze_encoder=true

train_config "M3_vae_finetune" \
    --policy.encoder=vae \
    --policy.vae_pretrained_path="$PRETRAINED_DIR/vae_d32.safetensors" \
    --policy.freeze_encoder=false

# M4/M5: β-VAE frozen/finetuned (best β based on Phase 1)
# Adjust path if your best β differs
train_config "M4_betavae_frozen" \
    --policy.encoder=beta_vae \
    --policy.vae_pretrained_path="$PRETRAINED_DIR/beta_vae_b4_d32.safetensors" \
    --policy.freeze_encoder=true

train_config "M5_betavae_finetune" \
    --policy.encoder=beta_vae \
    --policy.vae_pretrained_path="$PRETRAINED_DIR/beta_vae_b4_d32.safetensors" \
    --policy.freeze_encoder=false

# M6/M7: VQ-VAE frozen/finetuned
train_config "M6_vqvae_frozen" \
    --policy.encoder=vqvae \
    --policy.vae_pretrained_path="$PRETRAINED_DIR/vqvae_c512_g4_d32.safetensors" \
    --policy.freeze_encoder=true

train_config "M7_vqvae_finetune" \
    --policy.encoder=vqvae \
    --policy.vae_pretrained_path="$PRETRAINED_DIR/vqvae_c512_g4_d32.safetensors" \
    --policy.freeze_encoder=false

# ============================================================
#               Family C: Task-supervised
# ============================================================

echo "=== Family C: task-supervised priors ==="

train_config "M8_yolo_frozen" \
    --policy.encoder=yolo \
    --policy.freeze_encoder=true

train_config "M9_yolo_finetune" \
    --policy.encoder=yolo \
    --policy.freeze_encoder=false

train_config "M10_unet_frozen" \
    --policy.encoder=unet \
    --policy.freeze_encoder=true

train_config "M11_unet_finetune" \
    --policy.encoder=unet \
    --policy.freeze_encoder=false

# ============================================================
#               Family D: Foundation models
# ============================================================

echo "=== Family D: foundation models (always frozen) ==="

train_config "M12_sam2" \
    --policy.encoder=sam2

train_config "M13_dinov2" \
    --policy.encoder=dinov2

echo
echo "================================================================"
echo "  All training complete. Models in $OUTPUT_BASE/"
echo "================================================================"
ls -la "$OUTPUT_BASE/"
