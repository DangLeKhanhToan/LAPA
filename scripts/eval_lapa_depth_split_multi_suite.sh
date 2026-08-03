#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_DIR="$( cd -- "$( dirname -- "$SCRIPT_DIR" )" &> /dev/null && pwd )"
cd "$PROJECT_DIR"

LAPA_ROOT="${LAPA_ROOT:-$PROJECT_DIR}"
SUITES="${SUITES:?Set SUITES to a space-separated list of evaluation splits}"
TASK_IDS="${TASK_IDS:-0 1 2 3 4 5 6 7 8 9}"
N_EVAL_PER_TASK="${N_EVAL_PER_TASK:-10}"
MAX_STEPS="${MAX_STEPS:-500}"
PROGRESS_FREQ="${PROGRESS_FREQ:-25}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-eval_split}"
SHARED_CHECKPOINT="${SHARED_CHECKPOINT:?Set SHARED_CHECKPOINT to the policy parameter directory}"
ACTION_SCALE_FILE_TEMPLATE="${ACTION_SCALE_FILE_TEMPLATE:?Set ACTION_SCALE_FILE_TEMPLATE with a literal {suite} placeholder}"
EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:?Set EVAL_OUTPUT_ROOT to a writable directory}"

echo "[multi-suite] suites: $SUITES"
echo "[multi-suite] task_ids: $TASK_IDS"
echo "[multi-suite] n_eval_per_task: $N_EVAL_PER_TASK"
echo "[multi-suite] max_steps: $MAX_STEPS"
echo "[multi-suite] shared checkpoint: $SHARED_CHECKPOINT"

for suite in $SUITES; do
  ckpt="${SHARED_CHECKPOINT#params::}"
  if [[ ! -d "$ckpt" ]]; then
    echo "[multi-suite] ERROR: checkpoint not found: $ckpt" >&2
    exit 1
  fi
  action_scale_file="${ACTION_SCALE_FILE_TEMPLATE//\{suite\}/$suite}"
  if [[ ! -f "$action_scale_file" ]]; then
    echo "[multi-suite] ERROR: action-bin file not found: $action_scale_file" >&2
    exit 1
  fi

  echo "=============================="
  echo "[multi-suite] running suite=$suite"
  echo "[multi-suite] checkpoint=$ckpt"
  echo "=============================="

  pkill -u "$USER" -f "latent_pretraining.deploy" || true
  pkill -u "$USER" -f "eval.stage25_feature_server" || true
  pkill -u "$USER" -f "eval.lapa_rgb_feature_server" || true
  sleep "${SERVER_CLEANUP_SLEEP:-5}"

  export SUITE="$suite"
  export FINETUNED_CHECKPOINT="params::$ckpt"
  export ACTION_SCALE_FILE="$action_scale_file"
  export TASK_IDS="$TASK_IDS"
  export N_EVAL_PER_TASK="$N_EVAL_PER_TASK"
  export MAX_STEPS="$MAX_STEPS"
  export PROGRESS_FREQ="$PROGRESS_FREQ"
  export OUTPUT_DIR="$EVAL_OUTPUT_ROOT/${OUTPUT_PREFIX}_${suite}_tasks$(echo "$TASK_IDS" | wc -w)_eps${N_EVAL_PER_TASK}"

  bash "$SCRIPT_DIR/eval_lapa_depth_split_online_rollout.sh"
done

echo "[multi-suite] all suites complete"
