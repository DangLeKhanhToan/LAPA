#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_DIR="$( cd -- "$( dirname -- "$SCRIPT_DIR" )" &> /dev/null && pwd )"
cd "$PROJECT_DIR"

# Main setup. Override any of these from the shell before running this script.
export LAPA_ROOT="${LAPA_ROOT:-$PROJECT_DIR}"
export OUTPUT_DIR="${OUTPUT_DIR:-$LAPA_ROOT/outputs}"
export SUITES="${SUITES:-libero_spatial libero_object libero_goal libero_10}"

# Default: four parallel jobs, each using two GPUs. On an 8xH100 node this trains
# four suite models at once. Example override: GPU_GROUPS="0,1,2,3,4,5,6,7" PARALLEL=0
# to run suites sequentially with all 8 GPUs per suite.
export GPU_GROUPS="${GPU_GROUPS:-0,1;2,3;4,5;6,7}"
export PARALLEL="${PARALLEL:-1}"

# Stage-3 hyperparameters.
export TOTAL_STEPS="${TOTAL_STEPS:-20000}"
export BATCH_SIZE="${BATCH_SIZE:-128}"
export LR="${LR:-2e-5}"
export SEQ_LENGTH="${SEQ_LENGTH:-384}"

# Throughput knobs.
export TOKENIZER_PROCESSES="${TOKENIZER_PROCESSES:-4}"
export TOKENIZER_PARALLEL_CHUNK_SIZE="${TOKENIZER_PARALLEL_CHUNK_SIZE:-16}"
export TOKENIZER_PARALLEL_BATCH_SIZE="${TOKENIZER_PARALLEL_BATCH_SIZE:-256}"

# Logging/checkpoint knobs. By default we only save at the end.
export LOG_FREQ="${LOG_FREQ:-50}"
export EVAL_STEPS="${EVAL_STEPS:-0}"
export EVAL_LOG_FREQ="${EVAL_LOG_FREQ:-1000}"
export SAVE_MODEL_FREQ="${SAVE_MODEL_FREQ:-$TOTAL_STEPS}"
export SAVE_MILESTONE_FREQ="${SAVE_MILESTONE_FREQ:-0}"
export RUNTIME_LOG_STEPS="${RUNTIME_LOG_STEPS:-1}"
export WANDB_ONLINE="${WANDB_ONLINE:-False}"

export EXPERIMENT_PREFIX="${EXPERIMENT_PREFIX:-lapa_depth_stage3_fast}"
export LOG_DIR="${LOG_DIR:-$OUTPUT_DIR/logs}"
mkdir -p "$LOG_DIR"

gpu_count() {
  local group="$1"
  awk -F',' '{print NF}' <<< "$group"
}

select_gpu_group() {
  local index="$1"
  local groups_csv="$2"
  IFS=';' read -r -a groups <<< "$groups_csv"
  if [[ "$PARALLEL" == "1" ]]; then
    if (( index >= ${#groups[@]} )); then
      echo "ERROR: not enough GPU groups for parallel suite index $index" >&2
      exit 1
    fi
    echo "${groups[$index]}"
  else
    echo "${groups[0]}"
  fi
}

launch_suite() {
  local suite="$1"
  local gpu_group="$2"
  local n_gpus
  n_gpus="$(gpu_count "$gpu_group")"

  local experiment_id="${EXPERIMENT_PREFIX}_${suite}_${n_gpus}gpu"
  local log_file="$LOG_DIR/${experiment_id}.log"
  local mesh_dim="${MESH_DIM:-!-1,${n_gpus},1,1}"

  echo "[train-4suites] suite=$suite gpu_group=$gpu_group mesh=$mesh_dim log=$log_file"

  CUDA_VISIBLE_DEVICES="$gpu_group" \
  SUITE="$suite" \
  EXPERIMENT_ID="$experiment_id" \
  EXPERIMENT_NOTE="stage3_${suite}_${n_gpus}gpu_fast" \
  MESH_DIM="$mesh_dim" \
  bash "$SCRIPT_DIR/train_lapa_depth_suite.sh" > "$log_file" 2>&1
}

IFS=' ' read -r -a suite_list <<< "$SUITES"

echo "[train-4suites] LAPA_ROOT=$LAPA_ROOT"
echo "[train-4suites] OUTPUT_DIR=$OUTPUT_DIR"
echo "[train-4suites] SUITES=$SUITES"
echo "[train-4suites] GPU_GROUPS=$GPU_GROUPS"
echo "[train-4suites] PARALLEL=$PARALLEL"
echo "[train-4suites] TOTAL_STEPS=$TOTAL_STEPS BATCH_SIZE=$BATCH_SIZE LR=$LR"
echo "[train-4suites] TOKENIZER_PROCESSES=$TOKENIZER_PROCESSES"
echo "[train-4suites] LOG_FREQ=$LOG_FREQ SAVE_MODEL_FREQ=$SAVE_MODEL_FREQ"

pids=()
for i in "${!suite_list[@]}"; do
  suite="${suite_list[$i]}"
  gpu_group="$(select_gpu_group "$i" "$GPU_GROUPS")"
  if [[ "$PARALLEL" == "1" ]]; then
    launch_suite "$suite" "$gpu_group" &
    pids+=("$!")
  else
    launch_suite "$suite" "$gpu_group"
  fi
done

if [[ "$PARALLEL" == "1" ]]; then
  failed=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  if [[ "$failed" != "0" ]]; then
    echo "[train-4suites] one or more jobs failed; see $LOG_DIR" >&2
    exit 1
  fi
fi

echo "[train-4suites] done. Logs: $LOG_DIR"
