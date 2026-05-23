#!/usr/bin/env bash
# pretrain_all_vae_variants.sh
# Pretrain all VAE-family encoders for Phase 1 experiment.
#
# Usage:
#   ./pretrain_all_vae_variants.sh <dataset_repo_id> [out_dir]
#
# Optional env vars:
#   NUM_EPOCHS=50  BATCH_SIZE=64  NUM_WORKERS=4
#   PUSH_TO_HUB=1                # enable HF Hub push
#   HUB_USER=your_hf_username    # required if PUSH_TO_HUB=1
#   HUB_PRIVATE=1                # create private repos

set -euo pipefail

DATASET_REPO_ID="${1:-}"
if [ -z "$DATASET_REPO_ID" ]; then
    echo "Usage: $0 <dataset_repo_id> [out_dir]"
    echo "Example: $0 your_org/so101_pickplace_v1"
    exit 1
fi

OUT_DIR="${2:-./pretrained}"
mkdir -p "$OUT_DIR"

NUM_EPOCHS="${NUM_EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-4}"

# HF Hub push setup
HUB_FLAGS=()
if [ "${PUSH_TO_HUB:-0}" = "1" ]; then
    if [ -z "${HUB_USER:-}" ]; then
        echo "PUSH_TO_HUB=1 requires HUB_USER=your_hf_username"
        exit 1
    fi
    HUB_FLAGS+=(--push-to-hub)
    if [ "${HUB_PRIVATE:-0}" = "1" ]; then
        HUB_FLAGS+=(--hub-private)
    fi
fi

echo "================================================================"
echo "  Pretraining VAE-family encoders"
echo "  Dataset:  $DATASET_REPO_ID"
echo "  Out dir:  $OUT_DIR"
echo "  Epochs:   $NUM_EPOCHS"
echo "  Push HF:  ${PUSH_TO_HUB:-0}"
echo "================================================================"

run_pretrain() {
    local label="$1"
    local encoder_type="$2"
    local out_file="$3"
    local hub_suffix="$4"
    shift 4

    echo
    echo "[$label] Training $encoder_type → $out_file"

    local hub_args=()
    if [ "${PUSH_TO_HUB:-0}" = "1" ]; then
        hub_args=(--hub-repo-id="${HUB_USER}/so101-${hub_suffix}")
    fi

    pretrain-visual-encoder \
        --dataset-repo-id="$DATASET_REPO_ID" \
        --all-cameras \
        --encoder-type="$encoder_type" \
        --batch-size="$BATCH_SIZE" \
        --num-workers="$NUM_WORKERS" \
        --num-epochs="$NUM_EPOCHS" \
        --output-path="$OUT_DIR/$out_file" \
        "${HUB_FLAGS[@]}" "${hub_args[@]}" \
        "$@"
}

# Spatial VAE (49 tokens × 32-dim each)
run_pretrain "1/5" "vae" "vae_spatial_d32.safetensors" "vae-spatial-d32" \
    --latent-dim=32 --beta=1.0

# Spatial β-VAE with β=2
run_pretrain "2/5" "beta_vae" "beta_vae_spatial_b2_d32.safetensors" "beta-vae-b2-d32" \
    --latent-dim=32 --beta=2.0

# Spatial β-VAE with β=4
run_pretrain "3/5" "beta_vae" "beta_vae_spatial_b4_d32.safetensors" "beta-vae-b4-d32" \
    --latent-dim=32 --beta=4.0

# Spatial β-VAE with β=8
run_pretrain "4/5" "beta_vae" "beta_vae_spatial_b8_d32.safetensors" "beta-vae-b8-d32" \
    --latent-dim=32 --beta=8.0

# VQ-VAE with grid_size=7 (full backbone resolution)
run_pretrain "5/5" "vqvae" "vqvae_c512_g7_d32.safetensors" "vqvae-c512-g7-d32" \
    --latent-dim=32 --codebook-size=512 --grid-size=7

echo
echo "================================================================"
echo "  All pretraining complete. Weights in $OUT_DIR/"
echo "================================================================"
ls -lh "$OUT_DIR/"
