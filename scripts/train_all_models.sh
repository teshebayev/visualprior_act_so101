#!/usr/bin/env bash
# train_all_models.sh
#
# Тренировка experimental matrix Phase 2 (M0-M13)
#
# Usage:
#   ./scripts/train_all_models.sh DATASET_REPO_ID [OUTPUT_BASE] [PRETRAINED_DIR]
#
# Example:
#   ./scripts/train_all_models.sh rtx409011/ep_120 ./outputs ./pretrained
#
# Optional env vars:
#   STEPS=100000
#   BATCH_SIZE=8
#   NUM_WORKERS=4
#   SEEDS="42 123 7"
#   FORCE=1              # перезапускать даже если output_dir уже существует
#
# Важно:
#   Перед запуском VAE-family моделей нужно сначала выполнить:
#   ./scripts/pretrain_all_vae_variants.sh rtx409011/ep_120 ./pretrained

set -euo pipefail

DATASET_REPO_ID="${1:-}"

if [ -z "$DATASET_REPO_ID" ]; then
  echo "Usage: $0 DATASET_REPO_ID [OUTPUT_BASE] [PRETRAINED_DIR]"
  echo "Example: $0 rtx409011/ep_120 ./outputs ./pretrained"
  exit 1
fi

OUTPUT_BASE="${2:-./outputs}"
PRETRAINED_DIR="${3:-./pretrained}"

STEPS="${STEPS:-100000}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEEDS_STR="${SEEDS:-42 123 7}"
FORCE="${FORCE:-0}"

read -r -a SEEDS <<< "$SEEDS_STR"

mkdir -p "$OUTPUT_BASE"

echo "================================================================"
echo " Training VisualPrior ACT experimental matrix"
echo " Dataset:        $DATASET_REPO_ID"
echo " Output base:    $OUTPUT_BASE"
echo " Pretrained dir: $PRETRAINED_DIR"
echo " Steps:          $STEPS"
echo " Batch size:     $BATCH_SIZE"
echo " Num workers:    $NUM_WORKERS"
echo " Seeds:          ${SEEDS[*]}"
echo " Force:          $FORCE"
echo "================================================================"


# ------------------------------------------------------------
# Safety checks
# ------------------------------------------------------------

require_command() {
  local cmd="$1"

  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "ERROR: command not found: $cmd"
    echo "Check that your conda/venv environment is activated."
    exit 1
  fi
}

require_file() {
  local f="$1"

  if [ ! -f "$f" ]; then
    echo
    echo "ERROR: pretrained file not found:"
    echo "  $f"
    echo
    echo "Run VAE pretraining first:"
    echo "  ./scripts/pretrain_all_vae_variants.sh $DATASET_REPO_ID $PRETRAINED_DIR"
    echo
    echo "Or check PRETRAINED_DIR:"
    echo "  ls -lh $PRETRAINED_DIR"
    exit 1
  fi
}

require_command "lerobot-train"

# Эти файлы создаёт scripts/pretrain_all_vae_variants.sh
VAE_WEIGHTS="$PRETRAINED_DIR/vae_spatial_d32.safetensors"
BETA_VAE_B4_WEIGHTS="$PRETRAINED_DIR/beta_vae_spatial_b4_d32.safetensors"
VQVAE_WEIGHTS="$PRETRAINED_DIR/vqvae_c512_g7_d32.safetensors"

require_file "$VAE_WEIGHTS"
require_file "$BETA_VAE_B4_WEIGHTS"
require_file "$VQVAE_WEIGHTS"


# ------------------------------------------------------------
# Helper: train a single config across all seeds
# ------------------------------------------------------------

train_config() {
  local model_id="$1"
  shift

  local extra_args=("$@")

  for seed in "${SEEDS[@]}"; do
    local out_dir="$OUTPUT_BASE/${model_id}_seed${seed}"

    if [ -d "$out_dir" ] && [ "$FORCE" != "1" ]; then
      echo
      echo ">>> Skipping ${model_id}_seed${seed}"
      echo "    Output already exists: $out_dir"
      echo "    To rerun, use FORCE=1"
      continue
    fi

    if [ -d "$out_dir" ] && [ "$FORCE" = "1" ]; then
      echo
      echo ">>> FORCE=1: removing existing output:"
      echo "    $out_dir"
      rm -rf "$out_dir"
    fi

    echo
    echo "================================================================"
    echo " Training: $model_id"
    echo " Seed:     $seed"
    echo " Output:   $out_dir"
    echo "================================================================"

    PYTHONUNBUFFERED=1 lerobot-train \
      --policy.type=visualprior_act \
      --dataset.repo_id="$DATASET_REPO_ID" \
      --output_dir="$out_dir" \
      --seed="$seed" \
      --steps="$STEPS" \
      --batch_size="$BATCH_SIZE" \
      --num_workers="$NUM_WORKERS" \
      "${extra_args[@]}"
  done
}


# ============================================================
# Family A: Baseline + linear bottleneck
# ============================================================

echo
echo "=== Family A: baselines ==="

train_config "M0_resnet_baseline" \
  --policy.encoder=resnet18

train_config "M1_linear_bottleneck" \
  --policy.encoder=resnet18 \
  --policy.use_linear_bottleneck=true \
  --policy.bottleneck_dim=32


# ============================================================
# Family B: VAE / β-VAE / VQ-VAE
# ============================================================

echo
echo "=== Family B: VAE-family ==="

# M2/M3: Spatial VAE frozen / finetuned
train_config "M2_vae_frozen" \
  --policy.encoder=vae \
  --policy.vae_pretrained_path="$VAE_WEIGHTS" \
  --policy.vae_spatial=true \
  --policy.vae_latent_dim=32 \
  --policy.freeze_encoder=true

train_config "M3_vae_finetune" \
  --policy.encoder=vae \
  --policy.vae_pretrained_path="$VAE_WEIGHTS" \
  --policy.vae_spatial=true \
  --policy.vae_latent_dim=32 \
  --policy.freeze_encoder=false


# M4/M5: Spatial β-VAE β=4 frozen / finetuned
train_config "M4_betavae_frozen" \
  --policy.encoder=beta_vae \
  --policy.vae_pretrained_path="$BETA_VAE_B4_WEIGHTS" \
  --policy.vae_spatial=true \
  --policy.vae_latent_dim=32 \
  --policy.vae_beta=4.0 \
  --policy.freeze_encoder=true

train_config "M5_betavae_finetune" \
  --policy.encoder=beta_vae \
  --policy.vae_pretrained_path="$BETA_VAE_B4_WEIGHTS" \
  --policy.vae_spatial=true \
  --policy.vae_latent_dim=32 \
  --policy.vae_beta=4.0 \
  --policy.freeze_encoder=false


# M6/M7: VQ-VAE grid=7 frozen / finetuned
train_config "M6_vqvae_frozen" \
  --policy.encoder=vqvae \
  --policy.vae_pretrained_path="$VQVAE_WEIGHTS" \
  --policy.vae_latent_dim=32 \
  --policy.vqvae_codebook_size=512 \
  --policy.vqvae_grid_size=7 \
  --policy.freeze_encoder=true

train_config "M7_vqvae_finetune" \
  --policy.encoder=vqvae \
  --policy.vae_pretrained_path="$VQVAE_WEIGHTS" \
  --policy.vae_latent_dim=32 \
  --policy.vqvae_codebook_size=512 \
  --policy.vqvae_grid_size=7 \
  --policy.freeze_encoder=false


# ============================================================
# Family C: Task-supervised priors
# ============================================================

echo
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
# Family D: Foundation models
# ============================================================

echo
echo "=== Family D: foundation models ==="

train_config "M12_sam2" \
  --policy.encoder=sam2 \
  --policy.freeze_encoder=true

train_config "M13_dinov2" \
  --policy.encoder=dinov2 \
  --policy.freeze_encoder=true


echo
echo "================================================================"
echo " All training complete."
echo " Models in: $OUTPUT_BASE"
echo "================================================================"

ls -la "$OUTPUT_BASE"