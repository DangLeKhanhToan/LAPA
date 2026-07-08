export SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
export PROJECT_DIR="$( cd -- "$( dirname -- "$SCRIPT_DIR" )" &> /dev/null && pwd )"
cd "$PROJECT_DIR"
export PYTHONPATH="$PYTHONPATH:$PROJECT_DIR"

# Primary source: Stage-2.5 depth feature shards.
export DEPTH_DATA_DIR="${DEPTH_DATA_DIR:-$PROJECT_DIR/data/libero_depth_fusion}"
export DEPTH_MANIFEST="${DEPTH_MANIFEST:-$DEPTH_DATA_DIR/z_depth_train_shard0_model4_manifest.json}"

# Optional source: RGB feature shards, required when DEPTH_DATA_DIR does not
# contain z_rgb_feature_input. Model4 inspect output showed this is required.
export RGB_DATA_DIR="${RGB_DATA_DIR:-}"
export RGB_MANIFEST="${RGB_MANIFEST:-}"

# Action labels are required because inspected Stage-2.5 shards do not contain
# action_vector/raw_actions. Provide either ACTION_JSONL or ACTION_DATA_DIR.
export ACTION_JSONL="${ACTION_JSONL:-}"
export ACTION_DATA_DIR="${ACTION_DATA_DIR:-}"
export ACTION_MANIFEST="${ACTION_MANIFEST:-}"

export OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/outputs/depth_fusion_smoke}"
export MAX_SAMPLES="${MAX_SAMPLES:-2048}"
export MAX_TRAIN_BATCHES="${MAX_TRAIN_BATCHES:-8}"
export MAX_VAL_BATCHES="${MAX_VAL_BATCHES:-2}"
export BATCH_SIZE="${BATCH_SIZE:-64}"

EXTRA_ARGS=()
if [ -n "$RGB_DATA_DIR" ]; then
  EXTRA_ARGS+=(--rgb_data_dir "$RGB_DATA_DIR")
fi
if [ -n "$RGB_MANIFEST" ]; then
  EXTRA_ARGS+=(--rgb_manifest "$RGB_MANIFEST")
fi
if [ -n "$ACTION_JSONL" ]; then
  EXTRA_ARGS+=(--action_jsonl "$ACTION_JSONL")
fi
if [ -n "$ACTION_DATA_DIR" ]; then
  EXTRA_ARGS+=(--action_data_dir "$ACTION_DATA_DIR")
fi
if [ -n "$ACTION_MANIFEST" ]; then
  EXTRA_ARGS+=(--action_manifest "$ACTION_MANIFEST")
fi

python3 -u -m latent_pretraining.depth_fusion.train_depth_fusion \
    --data_dir "$DEPTH_DATA_DIR" \
    --manifest "$DEPTH_MANIFEST" \
    --output_dir "$OUTPUT_DIR" \
    --rgb_feature_key "auto" \
    --depth_feature_key "auto" \
    --action_key "auto" \
    --image_key "auto" \
    --epochs 1 \
    --batch_size "$BATCH_SIZE" \
    --lr 1e-4 \
    --weight_decay 1e-4 \
    --val_fraction 0.05 \
    --hidden_dim 2048 \
    --dropout 0.1 \
    --max_samples "$MAX_SAMPLES" \
    --max_train_batches "$MAX_TRAIN_BATCHES" \
    --max_val_batches "$MAX_VAL_BATCHES" \
    "${EXTRA_ARGS[@]}"
