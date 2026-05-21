#!/usr/bin/env bash
# eval_robot.sh
# Real-robot evaluation policy на SO-101
#
# Usage: ./eval_robot.sh outputs/M7_vqvae_finetune_seed42 [num_episodes]

set -euo pipefail

POLICY_PATH="${1:-}"
NUM_EPISODES="${2:-30}"
ROBOT_ID="${ROBOT_ID:-so101_main}"

if [ -z "$POLICY_PATH" ]; then
    echo "Usage: $0 <policy_path> [num_episodes]"
    echo "Example: $0 outputs/M7_vqvae_finetune_seed42 30"
    exit 1
fi

if [ ! -d "$POLICY_PATH" ]; then
    echo "ERROR: policy path not found: $POLICY_PATH"
    exit 1
fi

POLICY_NAME=$(basename "$POLICY_PATH")
EVAL_DIR="eval/${POLICY_NAME}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$EVAL_DIR"

echo "================================================================"
echo "  Real-robot evaluation"
echo "  Policy:    $POLICY_PATH"
echo "  Robot ID:  $ROBOT_ID"
echo "  Episodes:  $NUM_EPISODES"
echo "  Output:    $EVAL_DIR"
echo "================================================================"
echo

# Pre-flight checks
echo "Pre-flight checklist:"
echo "  [ ] Robot powered on and calibrated"
echo "  [ ] Camera in fixed position (run session-precheck if available)"
echo "  [ ] Cube placed on starting position P1"
echo "  [ ] Workspace clear"
echo
read -p "Press Enter when ready..."

lerobot-record \
    --robot.type=so101 \
    --robot.id="$ROBOT_ID" \
    --policy.path="$POLICY_PATH" \
    --num_episodes="$NUM_EPISODES" \
    --eval_dir="$EVAL_DIR"

echo
echo "================================================================"
echo "  Evaluation complete."
echo "================================================================"
echo "  Review video and score success/failure for each episode."
echo "  Then enter results into eval_results.csv with columns:"
echo "    policy, seed, condition, trial, success, failure_mode, notes"
