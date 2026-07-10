#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_DIR="$( cd -- "$( dirname -- "$SCRIPT_DIR" )" &> /dev/null && pwd )"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"

: "${DEPTH_VIDEO_ID:?Set DEPTH_VIDEO_ID to the offline depth video id used for rollout lookup, e.g. libero_90_TASK_demo_0.}"

MODEL_PY="${MODEL_PY:-/scratch/users/create/smrvmdo/venvs/lapa-depth/bin/python}"
LIBERO_PY="${LIBERO_PY:-/scratch/users/create/smrvmdo/venvs/LIBERO/bin/python}"
LIBERO_REPO="${LIBERO_REPO:-$PROJECT_DIR/datasets/LIBERO}"
LAPA_ROOT="${LAPA_ROOT:-$PROJECT_DIR}"
SUITE="${SUITE:-libero_90}"
DATA_ROOT="${DATA_ROOT:-$LAPA_ROOT/datasets/lapa_libero_v2}"
FINETUNED_CHECKPOINT="${FINETUNED_CHECKPOINT:-params::$LAPA_ROOT/outputs/lapa_depth_stage3_${SUITE}/streaming_params}"
ACTION_SCALE_FILE="${ACTION_SCALE_FILE:-$DATA_ROOT/action_bins_${SUITE}.csv}"
VQGAN_CHECKPOINT="${VQGAN_CHECKPOINT:-$LAPA_ROOT/lapa_checkpoints/vqgan}"
VOCAB_FILE="${VOCAB_FILE:-$LAPA_ROOT/lapa_checkpoints/tokenizer.model}"
DEPTH_BASE_DIR="${DEPTH_BASE_DIR:-$LAPA_ROOT/datasets/features_depth_branch/stage25_libero_features_model4/${SUITE}/stage25_model4}"
DEPTH_DATA_DIR="${DEPTH_DATA_DIR:-}"
DEPTH_MANIFEST="${DEPTH_MANIFEST:-}"
DEPTH_FEATURE_KEY="${DEPTH_FEATURE_KEY:-auto}"
DEPTH_ID_KEY="${DEPTH_ID_KEY:-auto}"
OUTPUT_DIR="${OUTPUT_DIR:-$LAPA_ROOT/outputs/eval_lapa_depth_${SUITE}}"
PORT="${PORT:-32820}"
TASK_IDS="${TASK_IDS:-0}"
N_EVAL_PER_TASK="${N_EVAL_PER_TASK:-1}"
MAX_STEPS="${MAX_STEPS:-80}"
INIT_OFFSET="${INIT_OFFSET:-0}"
DEPTH_START_STEP="${DEPTH_START_STEP:-0}"
USE_SINGULARITY="${USE_SINGULARITY:-1}"
RENDER_SIF_URL="${RENDER_SIF_URL:-docker://nvidia/opengl:1.2-glvnd-runtime-ubuntu22.04}"
RENDER_SIF="${RENDER_SIF:-/scratch/users/create/smrvmdo/venvs/opengl_glvnd.sif}"
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-/scratch/users/create/smrvmdo/.singularity_cache}"
export SINGULARITY_CACHEDIR="$APPTAINER_CACHEDIR"
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-/scratch/users/create/smrvmdo/.singularity_tmp}"
export SINGULARITY_TMPDIR="$APPTAINER_TMPDIR"

if [[ -z "$DEPTH_DATA_DIR" ]]; then
  if compgen -G "$DEPTH_BASE_DIR/*_part*.pt" >/dev/null || compgen -G "$DEPTH_BASE_DIR/*_part*.pth" >/dev/null; then
    DEPTH_DATA_DIR="$DEPTH_BASE_DIR"
  elif [[ -d "$DEPTH_BASE_DIR/z_depth_train_shard0" ]]; then
    DEPTH_DATA_DIR="$DEPTH_BASE_DIR/z_depth_train_shard0"
  else
    DEPTH_DATA_DIR="$DEPTH_BASE_DIR"
  fi
fi

if [[ -z "$DEPTH_MANIFEST" ]]; then
  for candidate in \
    "$DEPTH_DATA_DIR/z_depth_train_model4_manifest.json" \
    "$DEPTH_DATA_DIR/z_depth_train_shard0_model4_manifest.json" \
    "$DEPTH_DATA_DIR"/*_manifest.json; do
    if [[ -f "$candidate" ]]; then
      DEPTH_MANIFEST="$candidate"
      break
    fi
  done
fi

if [[ ! -f "$ACTION_SCALE_FILE" ]]; then
  echo "ERROR: action bins CSV not found: $ACTION_SCALE_FILE" >&2
  exit 1
fi
if [[ ! -d "$DEPTH_DATA_DIR" ]]; then
  echo "ERROR: depth feature directory not found: $DEPTH_DATA_DIR" >&2
  exit 1
fi

ACTION_VOCAB_SIZE="${ACTION_VOCAB_SIZE:-$(head -1 "$ACTION_SCALE_FILE" | awk -F, '{print NF}')}"
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
  --tokens_per_delta 4
  --tokens_per_action 7
  --depth_feature_data_dir "$DEPTH_DATA_DIR"
  --depth_feature_key "$DEPTH_FEATURE_KEY"
  --depth_feature_id_key "$DEPTH_ID_KEY"
)
if [[ -n "$DEPTH_MANIFEST" ]]; then
  server_args+=(--depth_feature_manifest "$DEPTH_MANIFEST")
fi

echo "[eval-depth-suite] suite: $SUITE"
echo "[eval-depth-suite] checkpoint: $FINETUNED_CHECKPOINT"
echo "[eval-depth-suite] action bins: $ACTION_SCALE_FILE"
echo "[eval-depth-suite] action_vocab_size: $ACTION_VOCAB_SIZE"
echo "[eval-depth-suite] depth dir: $DEPTH_DATA_DIR"
echo "[eval-depth-suite] depth manifest: ${DEPTH_MANIFEST:-<none>}"
echo "[eval-depth-suite] depth video id: $DEPTH_VIDEO_ID"
echo "[eval-depth-suite] starting server on port $PORT"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$MODEL_PY" "${server_args[@]}" &
SERVER_PID=$!

cleanup() {
  echo "[eval-depth-suite] stopping server pid $SERVER_PID"
  kill "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT

client_args=(
  "$PROJECT_DIR/eval/eval_libero_rollout_depth.py"
  --server_url "http://127.0.0.1:${PORT}/act"
  --output_dir "$OUTPUT_DIR"
  --suites "$SUITE"
  --task_ids $TASK_IDS
  --n_eval_per_task "$N_EVAL_PER_TASK"
  --max_steps "$MAX_STEPS"
  --init_offset "$INIT_OFFSET"
  --depth_video_id "$DEPTH_VIDEO_ID"
  --depth_start_step "$DEPTH_START_STEP"
)

MUJOCO_GL="${MUJOCO_GL:-egl}"
DEV="${MUJOCO_EGL_DEVICE_ID:-0}"
if [[ "$USE_SINGULARITY" == "1" ]]; then
  command -v singularity >/dev/null 2>&1 || module load singularity/4.1.5 2>/dev/null || true
  if [[ ! -f "$RENDER_SIF" ]]; then
    mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR" "$(dirname "$RENDER_SIF")"
    singularity pull "$RENDER_SIF" "$RENDER_SIF_URL"
  fi
  singularity exec --nv --bind /scratch \
    --env "MUJOCO_GL=${MUJOCO_GL},PYOPENGL_PLATFORM=${MUJOCO_GL},MUJOCO_EGL_DEVICE_ID=${DEV},PYTHONPATH=${LIBERO_REPO}" \
    "$RENDER_SIF" \
    "$LIBERO_PY" "${client_args[@]}"
else
  MUJOCO_GL="$MUJOCO_GL" PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-$MUJOCO_GL}" \
  MUJOCO_EGL_DEVICE_ID="$DEV" PYTHONPATH="$LIBERO_REPO:${PYTHONPATH:-}" \
  "$LIBERO_PY" "${client_args[@]}"
fi

echo "[eval-depth-suite] wrote results/videos to $OUTPUT_DIR"
