#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_DIR="$( cd -- "$( dirname -- "$SCRIPT_DIR" )" &> /dev/null && pwd )"
cd "$PROJECT_DIR"

LAPA_ROOT="${LAPA_ROOT:-$PROJECT_DIR}"
SUITES="${SUITES:-libero_spatial libero_object libero_goal libero_90}"
TASK_IDS="${TASK_IDS:-0 1 2 3 4 5 6 7 8 9}"
N_EVAL_PER_TASK="${N_EVAL_PER_TASK:-10}"
MAX_STEPS="${MAX_STEPS:-500}"
PROGRESS_FREQ="${PROGRESS_FREQ:-25}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-eval_split}"
CKPT_ROOT="${CKPT_ROOT:-$LAPA_ROOT/lapa_checkpoints/stage_3_depth_inject/lapa-depth_stage3}"

checkpoint_for_suite() {
  local suffix="${1#libero_}"
  case "$1" in
    libero_spatial|libero_object|libero_goal) echo "$CKPT_ROOT/128_batch_${suffix}" ;;
    libero_90)
      if [[ -d "$CKPT_ROOT/128_batch_90" ]]; then
        echo "$CKPT_ROOT/128_batch_90"
      elif [[ -d "$CKPT_ROOT/128_batch_libero_90" ]]; then
        echo "$CKPT_ROOT/128_batch_libero_90"
      else
        echo "$CKPT_ROOT/streaming_params"
      fi
      ;;
    *) echo "" ;;
  esac
}

echo "[multi-suite] suites: $SUITES"
echo "[multi-suite] task_ids: $TASK_IDS"
echo "[multi-suite] n_eval_per_task: $N_EVAL_PER_TASK"
echo "[multi-suite] max_steps: $MAX_STEPS"

for suite in $SUITES; do
  ckpt="$(checkpoint_for_suite "$suite")"
  if [[ -z "$ckpt" || ! -d "$ckpt" ]]; then
    echo "[multi-suite] ERROR: checkpoint not found for $suite: $ckpt" >&2
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
  export TASK_IDS="$TASK_IDS"
  export N_EVAL_PER_TASK="$N_EVAL_PER_TASK"
  export MAX_STEPS="$MAX_STEPS"
  export PROGRESS_FREQ="$PROGRESS_FREQ"
  export OUTPUT_DIR="$LAPA_ROOT/outputs/${OUTPUT_PREFIX}_${suite}_tasks$(echo "$TASK_IDS" | wc -w)_eps${N_EVAL_PER_TASK}"

  bash "$SCRIPT_DIR/eval_lapa_depth_split_online_rollout.sh"
done

echo "[multi-suite] all suites complete"
