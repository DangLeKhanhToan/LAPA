#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_DIR="$( cd -- "$( dirname -- "$SCRIPT_DIR" )" &> /dev/null && pwd )"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"

: "${DEPTH_DATA_DIR:?Set DEPTH_DATA_DIR to the directory containing depth .pt/.pth parts.}"
: "${DEPTH_VIDEO_ID:?Set DEPTH_VIDEO_ID, e.g. libero_90_TASK_demo_0, for offline rollout depth IDs.}"

MODEL_PY="${MODEL_PY:-/scratch/users/create/smrvmdo/venvs/lapa-depth/bin/python}"
LIBERO_PY="${LIBERO_PY:-/scratch/users/create/smrvmdo/venvs/LIBERO/bin/python}"
LIBERO_REPO="${LIBERO_REPO:-$PROJECT_DIR/datasets/LIBERO}"
LAPA_ROOT="${LAPA_ROOT:-$PROJECT_DIR}"
FINETUNED_CHECKPOINT="${FINETUNED_CHECKPOINT:-params::$LAPA_ROOT/outputs/smoke_overfit_lapa_depth_one_task/streaming_params}"
ACTION_SCALE_FILE="${ACTION_SCALE_FILE:-$LAPA_ROOT/datasets/libero_data/action_bins.csv}"
VQGAN_CHECKPOINT="${VQGAN_CHECKPOINT:-$LAPA_ROOT/lapa_checkpoints/vqgan}"
VOCAB_FILE="${VOCAB_FILE:-$LAPA_ROOT/lapa_checkpoints/tokenizer.model}"
DEPTH_MANIFEST="${DEPTH_MANIFEST:-}"
DEPTH_FEATURE_KEY="${DEPTH_FEATURE_KEY:-auto}"
DEPTH_ID_KEY="${DEPTH_ID_KEY:-auto}"
OUTPUT_DIR="${OUTPUT_DIR:-$LAPA_ROOT/outputs/eval_libero_depth_smoke}"
PORT="${PORT:-32820}"
SUITES="${SUITES:-libero_90}"
TASK_IDS="${TASK_IDS:-0}"
N_EVAL_PER_TASK="${N_EVAL_PER_TASK:-1}"
MAX_STEPS="${MAX_STEPS:-80}"
INIT_OFFSET="${INIT_OFFSET:-0}"
DEPTH_START_STEP="${DEPTH_START_STEP:-0}"
ACTION_VOCAB_SIZE="${ACTION_VOCAB_SIZE:-256}"
USE_SINGULARITY="${USE_SINGULARITY:-1}"
RENDER_SIF_URL="${RENDER_SIF_URL:-docker://nvidia/opengl:1.2-glvnd-runtime-ubuntu22.04}"
RENDER_SIF="${RENDER_SIF:-/scratch/users/create/smrvmdo/venvs/opengl_glvnd.sif}"
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-/scratch/users/create/smrvmdo/.singularity_cache}"
export SINGULARITY_CACHEDIR="$APPTAINER_CACHEDIR"
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-/scratch/users/create/smrvmdo/.singularity_tmp}"
export SINGULARITY_TMPDIR="$APPTAINER_TMPDIR"

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

echo "[eval-depth] starting LAPA-Depth server on port $PORT"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$MODEL_PY" "${server_args[@]}" &
SERVER_PID=$!

cleanup() {
  echo "[eval-depth] stopping server pid $SERVER_PID"
  kill "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT

client_args=(
  "$PROJECT_DIR/eval/eval_libero_rollout_depth.py"
  --server_url "http://127.0.0.1:${PORT}/act"
  --output_dir "$OUTPUT_DIR"
  --suites $SUITES
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

echo "[eval-depth] wrote results/videos to $OUTPUT_DIR"
