#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_DIR="$( cd -- "$( dirname -- "$SCRIPT_DIR" )" &> /dev/null && pwd )"
cd "$PROJECT_DIR"

# Train one suite at a time, using a configurable 4-8 GPU group.
export LAPA_ROOT="${LAPA_ROOT:-$PROJECT_DIR}"
export SUITE="${SUITE:-libero_spatial}"
export OUTPUT_DIR="${OUTPUT_DIR:-$LAPA_ROOT/outputs}"

# Pick one:
#   GPU_IDS=0,1,2,3       for 4 GPUs
#   GPU_IDS=0,1,2,3,4,5,6,7 for 8 GPUs
export GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
N_GPUS="$(awk -F',' '{print NF}' <<< "$GPU_IDS")"

export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export MESH_DIM="${MESH_DIM:-!-1,${N_GPUS},1,1}"

# Stage-3 defaults.
export TOTAL_STEPS="${TOTAL_STEPS:-20000}"
export BATCH_SIZE="${BATCH_SIZE:-128}"
export LR="${LR:-2e-5}"
export SEQ_LENGTH="${SEQ_LENGTH:-384}"

# Throughput knobs. Increase TOKENIZER_PROCESSES if CPU/RAM can handle it.
export TOKENIZER_PROCESSES="${TOKENIZER_PROCESSES:-$N_GPUS}"
export TOKENIZER_PARALLEL_CHUNK_SIZE="${TOKENIZER_PARALLEL_CHUNK_SIZE:-16}"
export TOKENIZER_PARALLEL_BATCH_SIZE="${TOKENIZER_PARALLEL_BATCH_SIZE:-256}"

# Reduce overhead.
export LOG_FREQ="${LOG_FREQ:-50}"
export EVAL_STEPS="${EVAL_STEPS:-0}"
export EVAL_LOG_FREQ="${EVAL_LOG_FREQ:-1000}"
export SAVE_MODEL_FREQ="${SAVE_MODEL_FREQ:-$TOTAL_STEPS}"
export SAVE_MILESTONE_FREQ="${SAVE_MILESTONE_FREQ:-0}"
export RUNTIME_LOG_STEPS="${RUNTIME_LOG_STEPS:-1}"
export WANDB_ONLINE="${WANDB_ONLINE:-False}"

export EXPERIMENT_ID="${EXPERIMENT_ID:-lapa_depth_stage3_${SUITE}_${N_GPUS}gpu_fast}"
export EXPERIMENT_NOTE="${EXPERIMENT_NOTE:-stage3_${SUITE}_${N_GPUS}gpu_fast}"

LOG_DIR="${LOG_DIR:-$OUTPUT_DIR/logs}"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_FILE:-$LOG_DIR/${EXPERIMENT_ID}.log}"

echo "[train-one-suite-fast] suite: $SUITE"
echo "[train-one-suite-fast] gpu ids: $GPU_IDS"
echo "[train-one-suite-fast] n_gpus: $N_GPUS"
echo "[train-one-suite-fast] mesh: $MESH_DIM"
echo "[train-one-suite-fast] total_steps: $TOTAL_STEPS"
echo "[train-one-suite-fast] batch_size: $BATCH_SIZE"
echo "[train-one-suite-fast] tokenizer_processes: $TOKENIZER_PROCESSES"
echo "[train-one-suite-fast] experiment_id: $EXPERIMENT_ID"
echo "[train-one-suite-fast] log file: $LOG_FILE"

bash "$SCRIPT_DIR/train_lapa_depth_suite.sh" 2>&1 | tee "$LOG_FILE"
