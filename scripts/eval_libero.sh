#!/bin/bash
# Closed-loop LIBERO rollout evaluation for a fine-tuned LAPA model.
#
# Two processes, two venvs (they have conflicting deps):
#   1) LAPA action server  -> lapa-depth venv (JAX), latent_pretraining.deploy
#   2) LIBERO rollout client -> LIBERO venv (mujoco/robosuite), eval/eval_libero_rollout.py
# This script starts the server, runs the client (which retries until the server is up),
# then tears the server down.

set -e
export SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
export PROJECT_DIR="$( cd -- "$( dirname -- "$SCRIPT_DIR" )" &> /dev/null && pwd )"
cd "$PROJECT_DIR"

export absolute_path="$PROJECT_DIR"

# --- Interpreters / paths -------------------------------------------------
# On this machine the conda `lapa` env has both the JAX stack (server) and
# mujoco/robosuite (client), so one interpreter serves both roles.
MODEL_PY="${MODEL_PY:-/home/linhkastner/miniconda3/envs/lapa/bin/python}"
LIBERO_PY="${LIBERO_PY:-/home/linhkastner/miniconda3/envs/lapa/bin/python}"
LIBERO_REPO="$absolute_path/datasets/LIBERO"   # editable install is broken; put repo on PYTHONPATH

# --- Config (override via env) --------------------------------------------
# REQUIRED: path to the fine-tuned params saved by finetune_libero_full.sh.
FINETUNED_CHECKPOINT="${FINETUNED_CHECKPOINT:-params::$absolute_path/outputs/finetune_libero_full/streaming_params}"
ACTION_SCALE_FILE="${ACTION_SCALE_FILE:-$absolute_path/datasets/lapa_libero/action_bins.csv}"
VQGAN_CHECKPOINT="${VQGAN_CHECKPOINT:-$absolute_path/lapa_checkpoints/vqgan}"
VOCAB_FILE="${VOCAB_FILE:-$absolute_path/lapa_checkpoints/tokenizer.model}"
OUTPUT_DIR="${OUTPUT_DIR:-$absolute_path/outputs/eval_libero}"
PORT="${PORT:-32822}"
SUITES="${SUITES:-libero_spatial libero_object libero_goal libero_10}"
N_EVAL_PER_TASK="${N_EVAL_PER_TASK:-5}"

# --- Rendering: singularity is only needed on nodes WITHOUT a GL/EGL stack ---
# This machine has the NVIDIA EGL vendor ICD installed (native offscreen GPU
# rendering works), so default off. Set USE_SINGULARITY=1 on bare HPC nodes to
# run the client in a glvnd container with --nv instead.
USE_SINGULARITY="${USE_SINGULARITY:-0}"
# glvnd runtime image; ubuntu22.04 matches the venv's glibc (python built on 22.04).
RENDER_SIF_URL="${RENDER_SIF_URL:-docker://nvidia/opengl:1.2-glvnd-runtime-ubuntu22.04}"
RENDER_SIF="${RENDER_SIF:-/scratch/users/create/smrvmdo/venvs/opengl_glvnd.sif}"
# Home is small/full on this cluster -> keep singularity caches on scratch.
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-/scratch/users/create/smrvmdo/.singularity_cache}"
export SINGULARITY_CACHEDIR="$APPTAINER_CACHEDIR"
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-/scratch/users/create/smrvmdo/.singularity_tmp}"
export SINGULARITY_TMPDIR="$APPTAINER_TMPDIR"

# --- GPU selection ---------------------------------------------------------
# GPU_MODE=utilize -> pick free GPUs automatically (sorted by free memory):
# the freest for the JAX server, the next-freest for the client's EGL renderer,
# so the two heavy processes land on different idle GPUs. Explicitly set
# CUDA_VISIBLE_DEVICES / MUJOCO_EGL_DEVICE_ID still take precedence.
# Any other value (default: unset) -> original behavior, both default to GPU 0.
GPU_MODE="${GPU_MODE:-}"
# Space-separated GPU indices utilize must never pick (e.g. "0" — GPU 0 hosts
# the Xorg display on this machine, so its watchdog kills long JAX kernels
# with CUDA_ERROR_LAUNCH_TIMEOUT). Explicit CUDA_VISIBLE_DEVICES still wins.
GPU_EXCLUDE="${GPU_EXCLUDE:-}"
SERVER_GPU="${CUDA_VISIBLE_DEVICES:-0}"
CLIENT_GPU="${MUJOCO_EGL_DEVICE_ID:-0}"
if [ "${GPU_MODE}" = "utilize" ]; then
    mapfile -t GPUS_BY_FREE < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
        | sort -t, -k2,2 -rn | awk -F, '{gsub(/ /,"",$1); print $1}')
    if [ -n "${GPU_EXCLUDE}" ]; then
        FILTERED=()
        for g in "${GPUS_BY_FREE[@]}"; do
            keep=1
            for x in ${GPU_EXCLUDE}; do
                if [ "$g" = "$x" ]; then keep=0; break; fi
            done
            if [ "$keep" = "1" ]; then FILTERED+=("$g"); fi
        done
        GPUS_BY_FREE=("${FILTERED[@]}")
    fi
    if [ "${#GPUS_BY_FREE[@]}" -eq 0 ]; then
        echo "ERROR: GPU_MODE=utilize found no usable GPUs (GPU_EXCLUDE='${GPU_EXCLUDE}')"; exit 1
    fi
    SERVER_GPU="${CUDA_VISIBLE_DEVICES:-${GPUS_BY_FREE[0]}}"
    CLIENT_GPU="${MUJOCO_EGL_DEVICE_ID:-${GPUS_BY_FREE[1]:-${GPUS_BY_FREE[0]}}}"
    # If the auto-picked client GPU collides with the server GPU (e.g. the user
    # pinned CUDA_VISIBLE_DEVICES), move the client to the next free GPU.
    if [ -z "${MUJOCO_EGL_DEVICE_ID:-}" ] && [ "${CLIENT_GPU}" = "${SERVER_GPU}" ]; then
        for g in "${GPUS_BY_FREE[@]}"; do
            if [ "$g" != "${SERVER_GPU}" ]; then CLIENT_GPU="$g"; break; fi
        done
    fi
    SERVER_FREE_MIB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${SERVER_GPU}")
    if [ "${SERVER_FREE_MIB}" -lt 16000 ]; then
        echo "[eval] WARNING: server GPU ${SERVER_GPU} has only ${SERVER_FREE_MIB} MiB free (7B bf16 needs ~15 GiB)"
    fi
    echo "[eval] GPU_MODE=utilize: server GPU=${SERVER_GPU}, client EGL GPU=${CLIENT_GPU}"
fi

if [ ! -f "${ACTION_SCALE_FILE}" ]; then
    echo "ERROR: action scale file not found: ${ACTION_SCALE_FILE}"
    echo "       Run data/process_libero.py first (it writes action_bins.csv)."
    exit 1
fi

# action_vocab_size MUST equal what finetune_libero_full.sh trained with (219 = # action bins).
# deploy.py defaults to 256 (correct for finetune_real/simpler, wrong for LIBERO) -> the 219-shaped
# checkpoint would fail to load. Derive from the action_bins.csv header (column count) so it can't drift.
ACTION_VOCAB_SIZE="${ACTION_VOCAB_SIZE:-$(head -1 "$ACTION_SCALE_FILE" | awk -F, '{print NF}')}"
UPDATE_LLAMA_CONFIG="${UPDATE_LLAMA_CONFIG:-dict(action_vocab_size=${ACTION_VOCAB_SIZE},delta_vocab_size=8,sample_mode='text',theta=50000000,max_sequence_length=32768,scan_attention=False,scan_query_chunk_size=128,scan_key_chunk_size=128,scan_mlp=False,scan_mlp_chunk_size=8192,scan_layers=True)}"
echo "[eval] action_vocab_size=${ACTION_VOCAB_SIZE} (from ${ACTION_SCALE_FILE})"

if [ "${USE_SINGULARITY}" = "1" ]; then
    command -v singularity >/dev/null 2>&1 || module load singularity/4.1.5 2>/dev/null || true
    if ! command -v singularity >/dev/null 2>&1; then
        echo "ERROR: singularity not found (try: module load singularity/4.1.5)"; exit 1
    fi
    if [ ! -f "${RENDER_SIF}" ]; then
        echo "[eval] pulling render image -> ${RENDER_SIF}"
        mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR" "$(dirname "$RENDER_SIF")"
        singularity pull "${RENDER_SIF}" "${RENDER_SIF_URL}"
    fi
fi

# --- 1) Start the LAPA action server (background) -------------------------
echo "[eval] starting LAPA server on port ${PORT} (checkpoint: ${FINETUNED_CHECKPOINT})"
CUDA_VISIBLE_DEVICES="${SERVER_GPU}" \
"$MODEL_PY" -m latent_pretraining.deploy \
    --load_checkpoint "${FINETUNED_CHECKPOINT}" \
    --action_scale_file "${ACTION_SCALE_FILE}" \
    --vqgan_checkpoint "${VQGAN_CHECKPOINT}" \
    --vocab_file "${VOCAB_FILE}" \
    --update_llama_config "${UPDATE_LLAMA_CONFIG}" \
    --port "${PORT}" \
    --mesh_dim "1,-1,1,1" \
    --tokens_per_delta 4 \
    --tokens_per_action 7 &
SERVER_PID=$!

# Always kill the server on exit.
cleanup() {
    echo "[eval] stopping LAPA server (pid ${SERVER_PID})"
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
}
trap cleanup EXIT

# --- 2) Run the LIBERO rollout client -------------------------------------
# Client retries the connection (default 60 x 10s = 10 min) while the 7B model loads.
MUJOCO_GL="${MUJOCO_GL:-egl}"
DEV="${CLIENT_GPU}"
# Orientation of frames sent to the model — must match the training images:
#   ROT180_FOR_MODEL=1 -> 180° rotation (OpenVLA-style img[::-1, ::-1]; wins over flip)
#   FLIP_FOR_MODEL=1   -> vertical flip only (default, preserves previous behavior)
#   both 0             -> raw env frame
FLIP_FOR_MODEL="${FLIP_FOR_MODEL:-1}"
ROT180_FOR_MODEL="${ROT180_FOR_MODEL:-0}"
CLIENT_ARGS=(
    "$absolute_path/eval/eval_libero_rollout.py"
    --server_url "http://127.0.0.1:${PORT}/act"
    --output_dir "${OUTPUT_DIR}"
    --suites ${SUITES}
    --n_eval_per_task "${N_EVAL_PER_TASK}"
)
[ "${FLIP_FOR_MODEL}" = "0" ] && CLIENT_ARGS+=(--flip_for_model)
[ "${ROT180_FOR_MODEL}" = "1" ] && CLIENT_ARGS+=(--rot180_for_model)

if [ "${USE_SINGULARITY}" = "1" ]; then
    echo "[eval] launching LIBERO rollout client in singularity --nv (${MUJOCO_GL})"
    # --nv mounts host NVIDIA GL libs; --bind /scratch exposes venv/repo/outputs.
    singularity exec --nv --bind /scratch \
        --env "MUJOCO_GL=${MUJOCO_GL},PYOPENGL_PLATFORM=${MUJOCO_GL},MUJOCO_EGL_DEVICE_ID=${DEV},PYTHONPATH=${LIBERO_REPO}" \
        "${RENDER_SIF}" \
        "$LIBERO_PY" "${CLIENT_ARGS[@]}"
else
    echo "[eval] launching LIBERO rollout client directly (${MUJOCO_GL})"
    # Direct path (only works if the node already has the GL/EGL/OSMesa stack).
    MUJOCO_GL="${MUJOCO_GL}" PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-${MUJOCO_GL}}" \
    MUJOCO_EGL_DEVICE_ID="${DEV}" \
    LD_LIBRARY_PATH="${RENDER_LD_LIBRARY_PATH:+${RENDER_LD_LIBRARY_PATH}:}${LD_LIBRARY_PATH}" \
    PYTHONPATH="$LIBERO_REPO:$PYTHONPATH" \
    "$LIBERO_PY" "${CLIENT_ARGS[@]}"
fi

echo "[eval] done. Results + videos in ${OUTPUT_DIR}"
