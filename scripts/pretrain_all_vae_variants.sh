#!/usr/bin/env bash
# pretrain_all_vae_variants.sh
# Pretrain все VAE-family encoders для Phase 1 эксперимента
#
# Usage: ./pretrain_all_vae_variants.sh your_org/so101_pickplace_v1

set -euo pipefail

DATASET_REPO_ID="${1:-}"
if [ -z "$DATASET_REPO_ID" ]; then
    echo "Usage: $0 <dataset_repo_id>"
    echo "Example: $0 your_org/so101_pickplace_v1"
    exit 1
fi

OUT_DIR="${2:-./pretrained}"
mkdir -p "$OUT_DIR"

NUM_EPOCHS="${NUM_EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-4}"

echo "================================================================"
echo "  Pretraining VAE-family encoders"
echo "  Dataset:  $DATASET_REPO_ID"
echo "  Out dir:  $OUT_DIR"
echo "  Epochs:   $NUM_EPOCHS"
echo "================================================================"

# Standard VAE
echo
echo "[1/5] Training VAE (d=32, β=1.0)"
pretrain-visual-encoder \
    --dataset-repo-id="$DATASET_REPO_ID" \
    --encoder-type=vae \
    --latent-dim=32 \
    --beta=1.0 \
    --batch-size="$BATCH_SIZE" \
    --num-workers="$NUM_WORKERS" \
    --num-epochs="$NUM_EPOCHS" \
    --output-path="$OUT_DIR/vae_d32.safetensors"

# β-VAE with β=2
echo
echo "[2/5] Training β-VAE (d=32, β=2.0)"
pretrain-visual-encoder \
    --dataset-repo-id="$DATASET_REPO_ID" \
    --encoder-type=beta_vae \
    --latent-dim=32 \
    --beta=2.0 \
    --batch-size="$BATCH_SIZE" \
    --num-workers="$NUM_WORKERS" \
    --num-epochs="$NUM_EPOCHS" \
    --output-path="$OUT_DIR/beta_vae_b2_d32.safetensors"

# β-VAE with β=4
echo
echo "[3/5] Training β-VAE (d=32, β=4.0)"
pretrain-visual-encoder \
    --dataset-repo-id="$DATASET_REPO_ID" \
    --encoder-type=beta_vae \
    --latent-dim=32 \
    --beta=4.0 \
    --batch-size="$BATCH_SIZE" \
    --num-workers="$NUM_WORKERS" \
    --num-epochs="$NUM_EPOCHS" \
    --output-path="$OUT_DIR/beta_vae_b4_d32.safetensors"

# β-VAE with β=8
echo
echo "[4/5] Training β-VAE (d=32, β=8.0)"
pretrain-visual-encoder \
    --dataset-repo-id="$DATASET_REPO_ID" \
    --encoder-type=beta_vae \
    --latent-dim=32 \
    --beta=8.0 \
    --batch-size="$BATCH_SIZE" \
    --num-workers="$NUM_WORKERS" \
    --num-epochs="$NUM_EPOCHS" \
    --output-path="$OUT_DIR/beta_vae_b8_d32.safetensors"

# VQ-VAE
echo
echo "[5/5] Training VQ-VAE (d=32, codebook=512, grid=4)"
pretrain-visual-encoder \
    --dataset-repo-id="$DATASET_REPO_ID" \
    --encoder-type=vqvae \
    --latent-dim=32 \
    --codebook-size=512 \
    --grid-size=4 \
    --batch-size="$BATCH_SIZE" \
    --num-workers="$NUM_WORKERS" \
    --num-epochs="$NUM_EPOCHS" \
    --output-path="$OUT_DIR/vqvae_c512_g4_d32.safetensors"

echo
echo "================================================================"
echo "  All pretraining complete. Weights in $OUT_DIR/"
echo "================================================================"
ls -lh "$OUT_DIR/"
