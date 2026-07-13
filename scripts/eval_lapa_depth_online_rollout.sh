#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_DIR="$( cd -- "$( dirname -- "$SCRIPT_DIR" )" &> /dev/null && pwd )"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"

MODEL_PY="${MODEL_PY:-/mnt/hdd/linh/long/conda_envs/lapa-depth/bin/python}"
LIBERO_PY="${LIBERO_PY:-$MODEL_PY}"
LIBERO_REPO="${LIBERO_REPO:-$PROJECT_DIR/datasets/LIBERO}"
LAPA_ROOT="${LAPA_ROOT:-$PROJECT_DIR}"
DEPTH_BRANCH_ROOT="${DEPTH_BRANCH_ROOT:-$LAPA_ROOT/Depth_branch}"
SUITE="${SUITE:-libero_spatial}"
DATA_ROOT="${DATA_ROOT:-$LAPA_ROOT/datasets/lapa_libero_v2}"
DEPTH_ANYTHING_REPO_DIR="${DEPTH_ANYTHING_REPO_DIR:-$LAPA_ROOT/third_party/depth_anything_v2}"
DEPTH_ANYTHING_CHECKPOINT="${DEPTH_ANYTHING_CHECKPOINT:-$LAPA_ROOT/checkpoints/depth_anything_v2_sth2sth/depth_anything_v2_sth2sth.pth}"
DEPTH_ANYTHING_ENCODER="${DEPTH_ANYTHING_ENCODER:-vitb}"
DEPTH_ANYTHING_INPUT_SIZE="${DEPTH_ANYTHING_INPUT_SIZE:-384}"
DEPTH_ANYTHING_DEVICE="${DEPTH_ANYTHING_DEVICE:-auto}"

FINETUNED_CHECKPOINT="${FINETUNED_CHECKPOINT:-params::$LAPA_ROOT/outputs/lapa_depth_stage3_${SUITE}/streaming_params}"
ORIGINAL_LAPA_CHECKPOINT="${ORIGINAL_LAPA_CHECKPOINT:-params::$LAPA_ROOT/lapa_checkpoints/pretraining_LAPA_Sth2Sth}"
ACTION_SCALE_FILE="${ACTION_SCALE_FILE:-$DATA_ROOT/action_bins_${SUITE}.csv}"
VQGAN_CHECKPOINT="${VQGAN_CHECKPOINT:-$LAPA_ROOT/lapa_checkpoints/vqgan}"
VOCAB_FILE="${VOCAB_FILE:-$LAPA_ROOT/lapa_checkpoints/tokenizer.model}"
STAGE25_MODEL_NAME="${STAGE25_MODEL_NAME:-model4}"
STAGE25_MODEL_CHECKPOINT="${STAGE25_MODEL_CHECKPOINT:-$LAPA_ROOT/lapa_checkpoints/depth_model/${STAGE25_MODEL_NAME}.65000.pt}"

POLICY_PORT="${POLICY_PORT:-32820}"
STAGE25_PORT="${STAGE25_PORT:-32821}"
OUTPUT_DIR="${OUTPUT_DIR:-$LAPA_ROOT/outputs/eval_lapa_depth_online_${SUITE}_${STAGE25_MODEL_NAME}}"
TASK_IDS="${TASK_IDS:-0}"
N_EVAL_PER_TASK="${N_EVAL_PER_TASK:-1}"
MAX_STEPS="${MAX_STEPS:-80}"
INIT_OFFSET="${INIT_OFFSET:-0}"
ACTION_VOCAB_SIZE="${ACTION_VOCAB_SIZE:-$(head -1 "$ACTION_SCALE_FILE" | awk -F, '{print NF}')}"
UPDATE_LLAMA_CONFIG="${UPDATE_LLAMA_CONFIG:-dict(action_vocab_size=${ACTION_VOCAB_SIZE},delta_vocab_size=8,sample_mode='text',theta=50000000,max_sequence_length=32768,scan_attention=False,scan_query_chunk_size=128,scan_key_chunk_size=128,scan_mlp=False,scan_mlp_chunk_size=8192,scan_layers=True)}"

USE_SINGULARITY="${USE_SINGULARITY:-0}"
RENDER_SIF_URL="${RENDER_SIF_URL:-docker://nvidia/opengl:1.2-glvnd-runtime-ubuntu22.04}"
RENDER_SIF="${RENDER_SIF:-$LAPA_ROOT/.apptainer/opengl_glvnd.sif}"
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-$LAPA_ROOT/.apptainer/cache}"
export SINGULARITY_CACHEDIR="$APPTAINER_CACHEDIR"
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-$LAPA_ROOT/.apptainer/tmp}"
export SINGULARITY_TMPDIR="$APPTAINER_TMPDIR"

export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.55}"
export TF_FORCE_GPU_ALLOW_GROWTH="${TF_FORCE_GPU_ALLOW_GROWTH:-true}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"

[[ -d "$DEPTH_BRANCH_ROOT" ]] || { echo "ERROR: DEPTH_BRANCH_ROOT not found: $DEPTH_BRANCH_ROOT" >&2; exit 1; }
[[ -f "$STAGE25_MODEL_CHECKPOINT" ]] || { echo "ERROR: STAGE25_MODEL_CHECKPOINT not found: $STAGE25_MODEL_CHECKPOINT" >&2; exit 1; }
[[ -f "$ACTION_SCALE_FILE" ]] || { echo "ERROR: ACTION_SCALE_FILE not found: $ACTION_SCALE_FILE" >&2; exit 1; }
[[ -d "$DEPTH_ANYTHING_REPO_DIR/depth_anything_v2" ]] || { echo "ERROR: DEPTH_ANYTHING_REPO_DIR is not a DepthAnythingV2 repo: $DEPTH_ANYTHING_REPO_DIR" >&2; exit 1; }
[[ -f "$DEPTH_ANYTHING_CHECKPOINT" ]] || { echo "ERROR: DEPTH_ANYTHING_CHECKPOINT not found: $DEPTH_ANYTHING_CHECKPOINT" >&2; exit 1; }

stage25_args=(
  -m eval.stage25_feature_server
  --stage25_bundle_dir "$DEPTH_BRANCH_ROOT"
  --model_name "$STAGE25_MODEL_NAME"
  --model_checkpoint "$STAGE25_MODEL_CHECKPOINT"
  --original_lapa_checkpoint "$ORIGINAL_LAPA_CHECKPOINT"
  --vqgan_checkpoint "$VQGAN_CHECKPOINT"
  --vocab_file "$VOCAB_FILE"
  --mesh_dim "1,1,1,1"
  --host "127.0.0.1"
  --port "$STAGE25_PORT"
  --depth_anything_repo_dir "$DEPTH_ANYTHING_REPO_DIR"
  --depth_anything_checkpoint "$DEPTH_ANYTHING_CHECKPOINT"
  --depth_anything_encoder "$DEPTH_ANYTHING_ENCODER"
  --depth_anything_input_size "$DEPTH_ANYTHING_INPUT_SIZE"
  --depth_anything_device "$DEPTH_ANYTHING_DEVICE"
)

policy_args=(
  -m latent_pretraining.deploy
  --load_checkpoint "$FINETUNED_CHECKPOINT"
  --action_scale_file "$ACTION_SCALE_FILE"
  --vqgan_checkpoint "$VQGAN_CHECKPOINT"
  --vocab_file "$VOCAB_FILE"
  --update_llama_config "$UPDATE_LLAMA_CONFIG"
  --port "$POLICY_PORT"
  --mesh_dim "1,-1,1,1"
  --tokens_per_delta 4
  --tokens_per_action 7
  --stage25_feature_server_url "http://127.0.0.1:${STAGE25_PORT}"
)

echo "[eval-online] suite: $SUITE"
echo "[eval-online] stage25 model: $STAGE25_MODEL_NAME $STAGE25_MODEL_CHECKPOINT"
echo "[eval-online] original LAPA: $ORIGINAL_LAPA_CHECKPOINT"
echo "[eval-online] finetuned policy: $FINETUNED_CHECKPOINT"
echo "[eval-online] depthanything: $DEPTH_ANYTHING_ENCODER $DEPTH_ANYTHING_CHECKPOINT"
echo "[eval-online] action bins: $ACTION_SCALE_FILE"
echo "[eval-online] starting Stage2.5 server on $STAGE25_PORT"
CUDA_VISIBLE_DEVICES="${STAGE25_CUDA_VISIBLE_DEVICES:-0}" "$MODEL_PY" "${stage25_args[@]}" &
STAGE25_PID=$!

echo "[eval-online] starting policy server on $POLICY_PORT"
CUDA_VISIBLE_DEVICES="${POLICY_CUDA_VISIBLE_DEVICES:-0}" "$MODEL_PY" "${policy_args[@]}" &
POLICY_PID=$!

cleanup() {
  echo "[eval-online] stopping servers"
  kill "$POLICY_PID" "$STAGE25_PID" 2>/dev/null || true
  wait "$POLICY_PID" 2>/dev/null || true
  wait "$STAGE25_PID" 2>/dev/null || true
}
trap cleanup EXIT

client_args=(
  "$PROJECT_DIR/eval/eval_libero_rollout_depth.py"
  --server_url "http://127.0.0.1:${POLICY_PORT}/act"
  --output_dir "$OUTPUT_DIR"
  --suites "$SUITE"
  --task_ids $TASK_IDS
  --n_eval_per_task "$N_EVAL_PER_TASK"
  --max_steps "$MAX_STEPS"
  --init_offset "$INIT_OFFSET"
)

MUJOCO_GL="${MUJOCO_GL:-egl}"
DEV="${MUJOCO_EGL_DEVICE_ID:-0}"
if [[ "$USE_SINGULARITY" == "1" ]]; then
  command -v singularity >/dev/null 2>&1 || module load singularity/4.1.5 2>/dev/null || true
  if [[ ! -f "$RENDER_SIF" ]]; then
    mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR" "$(dirname "$RENDER_SIF")"
    singularity pull "$RENDER_SIF" "$RENDER_SIF_URL"
  fi
  singularity exec --nv --bind "$LAPA_ROOT:$LAPA_ROOT" \
    --env "MUJOCO_GL=${MUJOCO_GL},PYOPENGL_PLATFORM=${MUJOCO_GL},MUJOCO_EGL_DEVICE_ID=${DEV},PYTHONPATH=${LIBERO_REPO}" \
    "$RENDER_SIF" \
    "$LIBERO_PY" "${client_args[@]}"
else
  MUJOCO_GL="$MUJOCO_GL" PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-$MUJOCO_GL}" \
  MUJOCO_EGL_DEVICE_ID="$DEV" PYTHONPATH="$LIBERO_REPO:${PYTHONPATH:-}" \
  "$LIBERO_PY" "${client_args[@]}"
fi

echo "[eval-online] wrote results/videos to $OUTPUT_DIR"
