#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_DIR="$( cd -- "$( dirname -- "$SCRIPT_DIR" )" &> /dev/null && pwd )"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"

: "${SMOKE_JSONL:?Set SMOKE_JSONL to the one-task JSONL.}"
: "${DEPTH_DATA_DIR:?Set DEPTH_DATA_DIR to the directory containing depth .pt/.pth parts.}"

MODEL_PY="${MODEL_PY:-python3}"
CLIENT_PY="${CLIENT_PY:-python3}"
LAPA_ROOT="${LAPA_ROOT:-$PROJECT_DIR}"
SUITE="${SUITE:-libero_90}"
DATA_ROOT="${DATA_ROOT:-$LAPA_ROOT/datasets/lapa_libero_v2}"
FINETUNED_CHECKPOINT="${FINETUNED_CHECKPOINT:-params::$LAPA_ROOT/outputs/smoke_overfit_lapa_depth_one_task/streaming_params}"
ACTION_SCALE_FILE="${ACTION_SCALE_FILE:-$DATA_ROOT/action_bins_${SUITE}.csv}"
VQGAN_CHECKPOINT="${VQGAN_CHECKPOINT:-$LAPA_ROOT/lapa_checkpoints/vqgan}"
VOCAB_FILE="${VOCAB_FILE:-$LAPA_ROOT/lapa_checkpoints/tokenizer.model}"
DEPTH_MANIFEST="${DEPTH_MANIFEST:-}"
DEPTH_FEATURE_KEY="${DEPTH_FEATURE_KEY:-auto}"
DEPTH_ID_KEY="${DEPTH_ID_KEY:-auto}"
IMAGE_ROOT="${IMAGE_ROOT:-$DATA_ROOT}"
OUTPUT_JSON="${OUTPUT_JSON:-$LAPA_ROOT/outputs/smoke_lapa_depth_replay/results.json}"
PORT="${PORT:-32820}"
TOKENS_PER_DELTA="${TOKENS_PER_DELTA:-4}"
TOKENS_PER_ACTION="${TOKENS_PER_ACTION:-7}"
if [[ -f "$ACTION_SCALE_FILE" ]]; then
  ACTION_VOCAB_SIZE="${ACTION_VOCAB_SIZE:-$(head -1 "$ACTION_SCALE_FILE" | awk -F, '{print NF}')}"
else
  ACTION_VOCAB_SIZE="${ACTION_VOCAB_SIZE:-256}"
fi
MAX_ROWS="${MAX_ROWS:-64}"

UPDATE_LLAMA_CONFIG="${UPDATE_LLAMA_CONFIG:-dict(action_vocab_size=${ACTION_VOCAB_SIZE},delta_vocab_size=8,sample_mode='text',theta=50000000,max_sequence_length=32768,scan_attention=False,scan_query_chunk_size=128,scan_key_chunk_size=128,scan_mlp=False,scan_mlp_chunk_size=8192,scan_layers=True)}"

server_args=(
  -m latent_pretraining.deploy
  --load_checkpoint "$FINETUNED_CHECKPOINT"
  --action_scale_file "$ACTION_SCALE_FILE"
  --vqgan_checkpoint "$VQGAN_CHECKPOINT"
  --vocab_file "$VOCAB_FILE"
  --update_llama_config "$UPDATE_LLAMA_CONFIG"
  --port "$PORT"
  --mesh_dim "1,-1,1,1"
  --tokens_per_delta "$TOKENS_PER_DELTA"
  --tokens_per_action "$TOKENS_PER_ACTION"
  --depth_feature_data_dir "$DEPTH_DATA_DIR"
  --depth_feature_key "$DEPTH_FEATURE_KEY"
  --depth_feature_id_key "$DEPTH_ID_KEY"
)
if [[ -n "$DEPTH_MANIFEST" ]]; then
  server_args+=(--depth_feature_manifest "$DEPTH_MANIFEST")
fi

echo "[smoke-replay] starting LAPA-Depth server on port $PORT"
echo "[smoke-replay] image root: $IMAGE_ROOT"
echo "[smoke-replay] action bins: $ACTION_SCALE_FILE"
echo "[smoke-replay] action_vocab_size: $ACTION_VOCAB_SIZE"
echo "[smoke-replay] output json: $OUTPUT_JSON"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$MODEL_PY" "${server_args[@]}" &
SERVER_PID=$!

cleanup() {
  echo "[smoke-replay] stopping server pid $SERVER_PID"
  kill "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT

"$CLIENT_PY" eval/smoke_replay_lapa_depth.py \
  --jsonl "$SMOKE_JSONL" \
  --image_root "$IMAGE_ROOT" \
  --server_url "http://127.0.0.1:${PORT}/act" \
  --output_json "$OUTPUT_JSON" \
  --max_rows "$MAX_ROWS"

echo "[smoke-replay] wrote $OUTPUT_JSON"
