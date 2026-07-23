#!/bin/bash
# Config-file wrapper around scripts/eval_libero.sh.
#
# Edit the CONFIG block below (or override any value via env when invoking),
# then run:  bash scripts/eval_libero_with_config.sh
# Values left empty ("") fall through to eval_libero.sh's own defaults.

set -e
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_DIR="$( cd -- "$( dirname -- "$SCRIPT_DIR" )" &> /dev/null && pwd )"

# ============================== CONFIG =====================================

# Checkpoint to evaluate (params::<path to streaming_params file>).
CHECKPOINT="${CHECKPOINT:-params::$PROJECT_DIR/lapa_checkpoints/params_v6_spatial}"

# Where results.json + per-episode videos go.
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/outputs/eval_libero_spatial_$(date +%Y%m%d_%H%M%S)}"

# Benchmark suites: all 4 eval suites, libero_90 excluded (train-only suite).
SUITES="${SUITES:-libero_spatial}" # libero_goal libero_object libero_spatial}"

# Rollouts per task.
N_EVAL_PER_TASK="${N_EVAL_PER_TASK:-10}"

# GPU selection: "utilize" = auto-pick the freest GPUs (server + client on
# different idle GPUs); "" = original behavior (GPU 0 for both).
GPU_MODE="${GPU_MODE:-utilize}"
# GPUs that utilize mode must never pick (space-separated indices). GPU 0
# hosts the Xorg display on this machine -> its watchdog kills long JAX
# kernels (CUDA_ERROR_LAUNCH_TIMEOUT), so exclude it by default.
GPU_EXCLUDE="${GPU_EXCLUDE:-0}"
# Or pin GPUs manually (these win over GPU_MODE=utilize when non-empty):
SERVER_CUDA_DEVICE="${SERVER_CUDA_DEVICE:-}"   # -> CUDA_VISIBLE_DEVICES of the JAX server
CLIENT_EGL_DEVICE="${CLIENT_EGL_DEVICE:-}"     # -> MUJOCO_EGL_DEVICE_ID of the renderer

# Action-bin table used to decode tokens -> continuous actions. MUST be the
# same action_bins.csv the checkpoint was fine-tuned with.
ACTION_SCALE_FILE="${ACTION_SCALE_FILE:-$PROJECT_DIR/datasets/lapa_libero/action_bins_libero_spatial.csv}"
# MUST equal the action_vocab_size the checkpoint was TRAINED with (see the
# finetune script's --llama.action_vocab_size). Leave empty to derive from
# ACTION_SCALE_FILE's column count — but note the CSV stores bin EDGES, so for
# this openvla dataset that derives 220 while training used 219 (edges-1),
# which crashes the action embedding load ("error" responses from the server).
ACTION_VOCAB_SIZE="${ACTION_VOCAB_SIZE:-}"

# Pretrained assets (usually unchanged).
VQGAN_CHECKPOINT="${VQGAN_CHECKPOINT:-$PROJECT_DIR/lapa_checkpoints/vqgan}"
VOCAB_FILE="${VOCAB_FILE:-$PROJECT_DIR/lapa_checkpoints/tokenizer.model}"

# Orientation of frames sent to the model — must match the training images:
#   ROT180_FOR_MODEL=1 -> 180° rotation (OpenVLA-style img[::-1, ::-1]; wins over flip)
#   FLIP_FOR_MODEL=1   -> vertical flip only
#   both 0             -> raw env frame
FLIP_FOR_MODEL="${FLIP_FOR_MODEL:-1}"
ROT180_FOR_MODEL="${ROT180_FOR_MODEL:-0}"

# Server port.
PORT="${PORT:-32825}"

# Rendering: 0 = native EGL (this machine), 1 = singularity --nv (bare HPC nodes).
USE_SINGULARITY="${USE_SINGULARITY:-0}"

# ===========================================================================

echo "[eval-config] checkpoint      = ${CHECKPOINT}"
echo "[eval-config] output_dir      = ${OUTPUT_DIR}"
echo "[eval-config] suites          = ${SUITES}"
echo "[eval-config] n_eval_per_task = ${N_EVAL_PER_TASK}"
echo "[eval-config] gpu_mode        = ${GPU_MODE:-<default GPU 0>}"
echo "[eval-config] model image     = rot180=${ROT180_FOR_MODEL} flip=${FLIP_FOR_MODEL}"

export FINETUNED_CHECKPOINT="$CHECKPOINT"
export OUTPUT_DIR SUITES N_EVAL_PER_TASK GPU_MODE GPU_EXCLUDE
export ACTION_SCALE_FILE VQGAN_CHECKPOINT VOCAB_FILE PORT USE_SINGULARITY
export FLIP_FOR_MODEL ROT180_FOR_MODEL
# Only export the optional pins when set, so eval_libero.sh's own
# defaulting/derivation logic still applies when they are left empty.
[ -n "$ACTION_VOCAB_SIZE" ]   && export ACTION_VOCAB_SIZE
[ -n "$SERVER_CUDA_DEVICE" ]  && export CUDA_VISIBLE_DEVICES="$SERVER_CUDA_DEVICE"
[ -n "$CLIENT_EGL_DEVICE" ]   && export MUJOCO_EGL_DEVICE_ID="$CLIENT_EGL_DEVICE"

exec bash "$SCRIPT_DIR/eval_libero.sh"
