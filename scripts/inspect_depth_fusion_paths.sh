#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_DIR="$( cd -- "$( dirname -- "$SCRIPT_DIR" )" &> /dev/null && pwd )"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"

: "${DEPTH_DATA_DIR:?Set DEPTH_DATA_DIR to the Stage-2.5 depth feature shard directory.}"
: "${DEPTH_MANIFEST:?Set DEPTH_MANIFEST to the Stage-2.5 manifest JSON.}"

RGB_DATA_DIR="${RGB_DATA_DIR:-}"
RGB_MANIFEST="${RGB_MANIFEST:-}"
ACTION_JSONL="${ACTION_JSONL:-}"
ACTION_DATA_DIR="${ACTION_DATA_DIR:-}"
ACTION_MANIFEST="${ACTION_MANIFEST:-}"

echo "[1/3] Inspect depth feature shard"
python3 -m latent_pretraining.depth_fusion.inspect_pt_shard \
  --data_dir "$DEPTH_DATA_DIR" \
  --manifest "$DEPTH_MANIFEST"

if [ -n "$RGB_DATA_DIR" ]; then
  echo "[2/3] Inspect RGB feature shard"
  RGB_ARGS=(--data_dir "$RGB_DATA_DIR" --action_key __skip_action_for_inspection__)
  if [ -n "$RGB_MANIFEST" ]; then
    RGB_ARGS+=(--manifest "$RGB_MANIFEST")
  fi
  python3 -m latent_pretraining.depth_fusion.inspect_pt_shard "${RGB_ARGS[@]}"
else
  echo "[2/3] Skip RGB inspect: RGB_DATA_DIR is not set"
fi

echo "[3/3] Optional tiny alignment check"
if [ -z "$RGB_DATA_DIR" ]; then
  echo "Cannot build aligned dataset without RGB_DATA_DIR when depth shards lack z_rgb_feature_input."
  exit 0
fi
if [ -z "$ACTION_JSONL" ] && [ -z "$ACTION_DATA_DIR" ]; then
  echo "Cannot build aligned dataset without ACTION_JSONL or ACTION_DATA_DIR."
  exit 0
fi

TRAIN_ARGS=(
  --data_dir "$DEPTH_DATA_DIR"
  --manifest "$DEPTH_MANIFEST"
  --rgb_data_dir "$RGB_DATA_DIR"
  --output_dir "${ALIGN_CHECK_OUTPUT_DIR:-$PROJECT_DIR/outputs/depth_fusion_align_check}"
  --epochs 1
  --batch_size 2
  --max_samples 8
  --max_train_batches 1
  --max_val_batches 1
  --num_workers 0
  --device cpu
)
if [ -n "$RGB_MANIFEST" ]; then
  TRAIN_ARGS+=(--rgb_manifest "$RGB_MANIFEST")
fi
if [ -n "$ACTION_JSONL" ]; then
  TRAIN_ARGS+=(--action_jsonl "$ACTION_JSONL")
fi
if [ -n "$ACTION_DATA_DIR" ]; then
  TRAIN_ARGS+=(--action_data_dir "$ACTION_DATA_DIR")
fi
if [ -n "$ACTION_MANIFEST" ]; then
  TRAIN_ARGS+=(--action_manifest "$ACTION_MANIFEST")
fi

python3 -u -m latent_pretraining.depth_fusion.train_depth_fusion "${TRAIN_ARGS[@]}"
