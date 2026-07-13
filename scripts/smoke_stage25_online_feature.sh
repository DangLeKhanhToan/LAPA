#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_DIR="$( cd -- "$( dirname -- "$SCRIPT_DIR" )" &> /dev/null && pwd )"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"

: "${RGB_IMAGE:?Set RGB_IMAGE to one RGB frame path.}"
: "${DEPTH_IMAGE:?Set DEPTH_IMAGE to one depth frame path or .npy.}"
: "${INSTRUCTION:?Set INSTRUCTION to the task instruction.}"

MODEL_PY="${MODEL_PY:-/scratch/users/create/smrvmdo/venvs/lapa-depth/bin/python}"
LAPA_ROOT="${LAPA_ROOT:-$PROJECT_DIR}"
DEPTH_BRANCH_ROOT="${DEPTH_BRANCH_ROOT:-$LAPA_ROOT/../Depth_branch}"
STAGE25_MODEL_NAME="${STAGE25_MODEL_NAME:-model4}"
STAGE25_MODEL_CHECKPOINT="${STAGE25_MODEL_CHECKPOINT:-$DEPTH_BRANCH_ROOT/${STAGE25_MODEL_NAME}.65000.pt}"
ORIGINAL_LAPA_CHECKPOINT="${ORIGINAL_LAPA_CHECKPOINT:-params::$LAPA_ROOT/lapa_checkpoints/lapa_7b_sth/params}"
VQGAN_CHECKPOINT="${VQGAN_CHECKPOINT:-$LAPA_ROOT/lapa_checkpoints/vqgan}"
VOCAB_FILE="${VOCAB_FILE:-$LAPA_ROOT/lapa_checkpoints/tokenizer.model}"
PORT="${PORT:-32821}"
OUTPUT_JSON="${OUTPUT_JSON:-$LAPA_ROOT/outputs/stage25_${STAGE25_MODEL_NAME}_smoke.json}"

mkdir -p "$(dirname "$OUTPUT_JSON")"

server_args=(
  -m eval.stage25_feature_server
  --stage25_bundle_dir "$DEPTH_BRANCH_ROOT"
  --model_name "$STAGE25_MODEL_NAME"
  --model_checkpoint "$STAGE25_MODEL_CHECKPOINT"
  --original_lapa_checkpoint "$ORIGINAL_LAPA_CHECKPOINT"
  --vqgan_checkpoint "$VQGAN_CHECKPOINT"
  --vocab_file "$VOCAB_FILE"
  --mesh_dim "1,1,1,1"
  --host "127.0.0.1"
  --port "$PORT"
)

echo "[stage25-smoke] starting server on $PORT"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$MODEL_PY" "${server_args[@]}" &
SERVER_PID=$!

cleanup() {
  kill "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT

"$MODEL_PY" - <<PY
import json
import time
import requests

url = "http://127.0.0.1:${PORT}/feature"
payload = {
    "image": "${RGB_IMAGE}",
    "depth_image": "${DEPTH_IMAGE}",
    "instruction": "${INSTRUCTION}",
    "return_debug": True,
}
last = None
for _ in range(60):
    try:
        r = requests.post(url, json=payload, timeout=180)
        r.raise_for_status()
        data = r.json()
        with open("${OUTPUT_JSON}", "w") as f:
            json.dump(data, f, indent=2)
        print(json.dumps({k: v for k, v in data.items() if k != "z_depth_feature_pred"}, indent=2))
        break
    except Exception as exc:
        last = exc
        time.sleep(10)
else:
    raise RuntimeError(f"stage25 server failed: {last}")
PY

echo "[stage25-smoke] wrote $OUTPUT_JSON"
