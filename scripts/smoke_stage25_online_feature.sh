#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_DIR="$( cd -- "$( dirname -- "$SCRIPT_DIR" )" &> /dev/null && pwd )"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --rgb_image)
      RGB_IMAGE="$2"
      shift 2
      ;;
    --depth_image)
      DEPTH_IMAGE="$2"
      shift 2
      ;;
    --instruction)
      INSTRUCTION="$2"
      shift 2
      ;;
    --output_json)
      OUTPUT_JSON="$2"
      shift 2
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    --model_py)
      MODEL_PY="$2"
      shift 2
      ;;
    --lapa_root)
      LAPA_ROOT="$2"
      shift 2
      ;;
    --depth_branch_root)
      DEPTH_BRANCH_ROOT="$2"
      shift 2
      ;;
    --stage25_model_name)
      STAGE25_MODEL_NAME="$2"
      shift 2
      ;;
    --stage25_model_checkpoint)
      STAGE25_MODEL_CHECKPOINT="$2"
      shift 2
      ;;
    --original_lapa_checkpoint)
      ORIGINAL_LAPA_CHECKPOINT="$2"
      shift 2
      ;;
    --vqgan_checkpoint)
      VQGAN_CHECKPOINT="$2"
      shift 2
      ;;
    --vocab_file)
      VOCAB_FILE="$2"
      shift 2
      ;;
    --depth_anything_repo_dir)
      DEPTH_ANYTHING_REPO_DIR="$2"
      shift 2
      ;;
    --depth_anything_checkpoint)
      DEPTH_ANYTHING_CHECKPOINT="$2"
      shift 2
      ;;
    --depth_anything_encoder)
      DEPTH_ANYTHING_ENCODER="$2"
      shift 2
      ;;
    --depth_anything_input_size)
      DEPTH_ANYTHING_INPUT_SIZE="$2"
      shift 2
      ;;
    --depth_anything_device)
      DEPTH_ANYTHING_DEVICE="$2"
      shift 2
      ;;
    --mesh_dim)
      STAGE25_MESH_DIM="$2"
      shift 2
      ;;
    *)
      echo "ERROR: unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

: "${RGB_IMAGE:?Set RGB_IMAGE to one RGB frame path.}"
: "${INSTRUCTION:?Set INSTRUCTION to the task instruction.}"

MODEL_PY="${MODEL_PY:-/mnt/hdd/linh/long/conda_envs/lapa-depth/bin/python}"
LAPA_ROOT="${LAPA_ROOT:-$PROJECT_DIR}"
DEPTH_BRANCH_ROOT="${DEPTH_BRANCH_ROOT:-$LAPA_ROOT/Depth_branch}"
STAGE25_MODEL_NAME="${STAGE25_MODEL_NAME:-model4}"
STAGE25_MODEL_CHECKPOINT="${STAGE25_MODEL_CHECKPOINT:-/home/linhkastner/lapa/LAPA-depth/lapa_checkpoints/depth_model/${STAGE25_MODEL_NAME}.65000.pt}"
ORIGINAL_LAPA_CHECKPOINT="${ORIGINAL_LAPA_CHECKPOINT:-params::$LAPA_ROOT/lapa_checkpoints/pretraining_LAPA_Sth2Sth}"
VQGAN_CHECKPOINT="${VQGAN_CHECKPOINT:-$LAPA_ROOT/lapa_checkpoints/vqgan}"
VOCAB_FILE="${VOCAB_FILE:-$LAPA_ROOT/lapa_checkpoints/tokenizer.model}"
DEPTH_ANYTHING_REPO_DIR="${DEPTH_ANYTHING_REPO_DIR:-$LAPA_ROOT/third_party/depth_anything_v2}"
DEPTH_ANYTHING_CHECKPOINT="${DEPTH_ANYTHING_CHECKPOINT:-$LAPA_ROOT/checkpoints/depth_anything_v2_sth2sth/depth_anything_v2_sth2sth.pth}"
DEPTH_ANYTHING_ENCODER="${DEPTH_ANYTHING_ENCODER:-vitl}"
DEPTH_ANYTHING_INPUT_SIZE="${DEPTH_ANYTHING_INPUT_SIZE:-518}"
DEPTH_ANYTHING_DEVICE="${DEPTH_ANYTHING_DEVICE:-auto}"
STAGE25_MESH_DIM="${STAGE25_MESH_DIM:-1,2,1,1}"
PORT="${PORT:-32823}"
OUTPUT_JSON="${OUTPUT_JSON:-$LAPA_ROOT/outputs/stage25_${STAGE25_MODEL_NAME}_smoke.json}"

mkdir -p "$(dirname "$OUTPUT_JSON")"

export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.55}"
export TF_FORCE_GPU_ALLOW_GROWTH="${TF_FORCE_GPU_ALLOW_GROWTH:-true}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda,cpu}"

server_args=(
  -m eval.stage25_feature_server
  --stage25_bundle_dir "$DEPTH_BRANCH_ROOT"
  --model_name "$STAGE25_MODEL_NAME"
  --model_checkpoint "$STAGE25_MODEL_CHECKPOINT"
  --original_lapa_checkpoint "$ORIGINAL_LAPA_CHECKPOINT"
  --vqgan_checkpoint "$VQGAN_CHECKPOINT"
  --vocab_file "$VOCAB_FILE"
  --mesh_dim "$STAGE25_MESH_DIM"
  --host "127.0.0.1"
  --port "$PORT"
)

if [[ -n "${DEPTH_IMAGE:-}" ]]; then
  echo "[stage25-smoke] using provided DEPTH_IMAGE=$DEPTH_IMAGE"
else
  [[ -d "$DEPTH_ANYTHING_REPO_DIR/depth_anything_v2" ]] || { echo "ERROR: DEPTH_ANYTHING_REPO_DIR is not a DepthAnythingV2 repo: $DEPTH_ANYTHING_REPO_DIR" >&2; exit 1; }
  [[ -f "$DEPTH_ANYTHING_CHECKPOINT" ]] || { echo "ERROR: DEPTH_ANYTHING_CHECKPOINT not found: $DEPTH_ANYTHING_CHECKPOINT" >&2; exit 1; }
  server_args+=(
    --depth_anything_repo_dir "$DEPTH_ANYTHING_REPO_DIR"
    --depth_anything_checkpoint "$DEPTH_ANYTHING_CHECKPOINT"
    --depth_anything_encoder "$DEPTH_ANYTHING_ENCODER"
    --depth_anything_input_size "$DEPTH_ANYTHING_INPUT_SIZE"
    --depth_anything_device "$DEPTH_ANYTHING_DEVICE"
  )
fi

echo "[stage25-smoke] stage25 visible GPUs: ${CUDA_VISIBLE_DEVICES:-${STAGE25_CUDA_VISIBLE_DEVICES:-2,3}}"
echo "[stage25-smoke] stage25 mesh_dim: $STAGE25_MESH_DIM"
echo "[stage25-smoke] starting server on $PORT"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${STAGE25_CUDA_VISIBLE_DEVICES:-2,3}}" "$MODEL_PY" "${server_args[@]}" &
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
    "instruction": "${INSTRUCTION}",
    "return_debug": True,
}
depth_image = "${DEPTH_IMAGE:-}"
if depth_image:
    payload["depth_image"] = depth_image
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
