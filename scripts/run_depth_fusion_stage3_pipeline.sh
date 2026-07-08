#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_DIR="$( cd -- "$( dirname -- "$SCRIPT_DIR" )" &> /dev/null && pwd )"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"

# Fill these paths on the server before running.
: "${DEPTH_DATA_DIR:?Set DEPTH_DATA_DIR to the Stage-2.5 depth feature shard directory.}"
: "${DEPTH_MANIFEST:?Set DEPTH_MANIFEST to the Stage-2.5 manifest JSON.}"

# Required when depth shards do not contain z_rgb_feature_input. Model4 inspect
# output showed that RGB features must be supplied separately.
: "${RGB_DATA_DIR:?Set RGB_DATA_DIR to the LAPA RGB feature shard directory.}"
: "${RGB_MANIFEST:?Set RGB_MANIFEST to the LAPA RGB feature manifest JSON.}"

# Required because the inspected Stage-2.5 shards do not contain action labels.
# Use either ACTION_JSONL or ACTION_DATA_DIR/ACTION_MANIFEST.
ACTION_JSONL="${ACTION_JSONL:-}"
ACTION_DATA_DIR="${ACTION_DATA_DIR:-}"
ACTION_MANIFEST="${ACTION_MANIFEST:-}"
if [ -z "$ACTION_JSONL" ] && [ -z "$ACTION_DATA_DIR" ]; then
  echo "Set ACTION_JSONL or ACTION_DATA_DIR/ACTION_MANIFEST for action labels." >&2
  exit 1
fi

OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/outputs/depth_fusion_stage3}"
SMOKE_OUTPUT_DIR="${SMOKE_OUTPUT_DIR:-$OUTPUT_ROOT/smoke}"
FULL_OUTPUT_DIR="${FULL_OUTPUT_DIR:-$OUTPUT_ROOT/full}"

ACTION_ARGS=()
if [ -n "$ACTION_JSONL" ]; then
  ACTION_ARGS+=(--action_jsonl "$ACTION_JSONL")
fi
if [ -n "$ACTION_DATA_DIR" ]; then
  ACTION_ARGS+=(--action_data_dir "$ACTION_DATA_DIR")
fi
if [ -n "$ACTION_MANIFEST" ]; then
  ACTION_ARGS+=(--action_manifest "$ACTION_MANIFEST")
fi

echo "[1/5] Inspect depth shard schema"
python3 -m latent_pretraining.depth_fusion.inspect_pt_shard \
  --data_dir "$DEPTH_DATA_DIR" \
  --manifest "$DEPTH_MANIFEST"

echo "[2/5] Inspect RGB shard schema"
python3 -m latent_pretraining.depth_fusion.inspect_pt_shard \
  --data_dir "$RGB_DATA_DIR" \
  --manifest "$RGB_MANIFEST" \
  --action_key __skip_action_for_inspection__

echo "[3/5] Smoke fine-tune one epoch / few batches"
python3 -u -m latent_pretraining.depth_fusion.train_depth_fusion \
  --data_dir "$DEPTH_DATA_DIR" \
  --manifest "$DEPTH_MANIFEST" \
  --rgb_data_dir "$RGB_DATA_DIR" \
  --rgb_manifest "$RGB_MANIFEST" \
  "${ACTION_ARGS[@]}" \
  --output_dir "$SMOKE_OUTPUT_DIR" \
  --rgb_feature_key auto \
  --depth_feature_key auto \
  --action_key auto \
  --image_key auto \
  --epochs 1 \
  --batch_size "${SMOKE_BATCH_SIZE:-64}" \
  --lr "${SMOKE_LR:-1e-4}" \
  --weight_decay "${WEIGHT_DECAY:-1e-4}" \
  --val_fraction "${VAL_FRACTION:-0.05}" \
  --max_samples "${SMOKE_MAX_SAMPLES:-2048}" \
  --max_train_batches "${SMOKE_MAX_TRAIN_BATCHES:-8}" \
  --max_val_batches "${SMOKE_MAX_VAL_BATCHES:-2}" \
  --num_workers "${NUM_WORKERS:-4}"

echo "[4/5] Check smoke checkpoint can predict actions"
python3 -m latent_pretraining.depth_fusion.predict_depth_fusion \
  --checkpoint "$SMOKE_OUTPUT_DIR/best.pt" \
  --data_dir "$DEPTH_DATA_DIR" \
  --manifest "$DEPTH_MANIFEST" \
  --rgb_data_dir "$RGB_DATA_DIR" \
  --rgb_manifest "$RGB_MANIFEST" \
  --output_jsonl "$SMOKE_OUTPUT_DIR/predictions.jsonl" \
  --max_samples "${PREDICT_MAX_SAMPLES:-32}"

echo "[5/5] Full offline Stage-3 depth-fusion fine-tune"
python3 -u -m latent_pretraining.depth_fusion.train_depth_fusion \
  --data_dir "$DEPTH_DATA_DIR" \
  --manifest "$DEPTH_MANIFEST" \
  --rgb_data_dir "$RGB_DATA_DIR" \
  --rgb_manifest "$RGB_MANIFEST" \
  "${ACTION_ARGS[@]}" \
  --output_dir "$FULL_OUTPUT_DIR" \
  --rgb_feature_key auto \
  --depth_feature_key auto \
  --action_key auto \
  --image_key auto \
  --epochs "${EPOCHS:-20}" \
  --batch_size "${BATCH_SIZE:-256}" \
  --lr "${LR:-1e-4}" \
  --weight_decay "${WEIGHT_DECAY:-1e-4}" \
  --val_fraction "${VAL_FRACTION:-0.05}" \
  --num_workers "${NUM_WORKERS:-4}"

echo "Done. Smoke checkpoint: $SMOKE_OUTPUT_DIR/best.pt"
echo "Done. Full checkpoint:  $FULL_OUTPUT_DIR/best.pt"
