#!/usr/bin/env bash
set -euo pipefail

# Minimal runner for Stage 2.5 rollout (single sample smoke test).
# Edit the variables below, then run:
#   bash run_stage25_simple.sh

# ===== Required inputs =====
RGB_IMAGE="/path/to/rgb_frame.jpg"                    # RGB image file
DEPTH_IMAGE="/path/to/depth_frame.png"                # Depth image file
INSTRUCTION="pick up the cup"                         # Task instruction text

# LAPA checkpoints
VQGAN_CHECKPOINT="/path/to/lapa_checkpoints/vqgan"
VOCAB_FILE="/path/to/lapa_checkpoints/tokenizer.model"
LOAD_CHECKPOINT="params::/path/to/lapa_checkpoints/streaming_params_22485"

# Model4 checkpoint
MODEL4_CHECKPOINT="/path/to/model4.65000.pt"

# Optional output
OUTPUT_PT="/tmp/stage25_step_output.pt"

# ===== Basic validation =====
[[ -f "$RGB_IMAGE" ]] || { echo "Missing RGB_IMAGE: $RGB_IMAGE"; exit 1; }
[[ -f "$DEPTH_IMAGE" ]] || { echo "Missing DEPTH_IMAGE: $DEPTH_IMAGE"; exit 1; }
[[ -e "$VQGAN_CHECKPOINT" ]] || { echo "Missing VQGAN_CHECKPOINT: $VQGAN_CHECKPOINT"; exit 1; }
[[ -f "$VOCAB_FILE" ]] || { echo "Missing VOCAB_FILE: $VOCAB_FILE"; exit 1; }
[[ "$LOAD_CHECKPOINT" == params::* ]] || {
  echo "LOAD_CHECKPOINT must start with params::"
  exit 1
}
[[ -f "$MODEL4_CHECKPOINT" ]] || { echo "Missing MODEL4_CHECKPOINT: $MODEL4_CHECKPOINT"; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python "$SCRIPT_DIR/laq/rollout_stage25_model4.py" \
  --rgb_image "$RGB_IMAGE" \
  --depth_image "$DEPTH_IMAGE" \
  --instruction "$INSTRUCTION" \
  --vqgan_checkpoint "$VQGAN_CHECKPOINT" \
  --vocab_file "$VOCAB_FILE" \
  --load_checkpoint "$LOAD_CHECKPOINT" \
  --model4_checkpoint "$MODEL4_CHECKPOINT" \
  --output_pt "$OUTPUT_PT"

echo "Done. Output saved to: $OUTPUT_PT"
