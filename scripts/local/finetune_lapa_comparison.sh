#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

MODE="${MODE:-baseline}"                    # baseline or depth_concat
RUN_KIND="${RUN_KIND:-smoke}"               # smoke or full
MODEL="${MODEL:-model2}"
DATA_ROOT="${DATA_ROOT:-$ROOT/datasets/lapa_libero_v4}"
if [[ -f "$DATA_ROOT/all_train.jsonl" ]]; then
  DEFAULT_TRAIN_JSONL="$DATA_ROOT/all_train.jsonl"
else
  DEFAULT_TRAIN_JSONL="$DATA_ROOT/combined/all_suite_train.jsonl"
fi
TRAIN_JSONL="${TRAIN_JSONL:-$DEFAULT_TRAIN_JSONL}"
DEPTH_BUNDLE="${DEPTH_BUNDLE:-$DATA_ROOT/packed_depth/${MODEL}_all_suite_fp16.pt}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-ring}"
SUITES="${SUITES:-libero_spatial libero_object libero_goal libero_90}"

ACTION_VOCAB_SIZE=0
for suite in $SUITES; do
  bins="$DATA_ROOT/action_bins_${suite}.csv"
  [[ -f "$bins" ]] || { echo "ERROR: missing per-suite action bins: $bins" >&2; exit 1; }
  suite_vocab="$(head -1 "$bins" | awk -F, '{print NF}')"
  if (( suite_vocab > ACTION_VOCAB_SIZE )); then
    ACTION_VOCAB_SIZE="$suite_vocab"
  fi
done

case "$RUN_KIND" in
  smoke)
    TOTAL_STEPS="${TOTAL_STEPS:-10}"
    SAVE_MODEL_FREQ="${SAVE_MODEL_FREQ:-0}"
    ;;
  full)
    TOTAL_STEPS="${TOTAL_STEPS:-20000}"
    SAVE_MODEL_FREQ="${SAVE_MODEL_FREQ:-20000}"
    ;;
  *) echo "ERROR: RUN_KIND must be smoke or full" >&2; exit 1 ;;
esac

COMMON=(
  --suite all_suite
  --model "$MODEL"
  --data-root "$DATA_ROOT"
  --train-jsonl "$TRAIN_JSONL"
  --total-steps "$TOTAL_STEPS"
  --batch-size 128
  --seq-length 384
  --mesh-dim '!-1,4,1,1'
  --lr 2e-5
  --action-vocab-size "$ACTION_VOCAB_SIZE"
  --attention-backend "$ATTENTION_BACKEND"
  --save-model-freq "$SAVE_MODEL_FREQ"
  --save-milestone-freq 0
  --save-optimizer-state false
  --autoresume false
)

case "$MODE" in
  baseline)
    EXPERIMENT_ID="${EXPERIMENT_ID:-baseline_all_suite_${RUN_KIND}}"
    EXTRA=(--no-depth --action-fusion project)
    ;;
  depth_concat)
    [[ -f "$DEPTH_BUNDLE" ]] || {
      echo "ERROR: build the packed depth bundle first: $DEPTH_BUNDLE" >&2
      exit 1
    }
    EXPERIMENT_ID="${EXPERIMENT_ID:-${MODEL}_depth_concat_all_suite_${RUN_KIND}}"
    EXTRA=(--depth-bundle "$DEPTH_BUNDLE" --action-fusion concat)
    ;;
  *) echo "ERROR: MODE must be baseline or depth_concat" >&2; exit 1 ;;
esac

echo "[comparison] mode=$MODE run=$RUN_KIND steps=$TOTAL_STEPS experiment=$EXPERIMENT_ID"
bash scripts/train_lapa_depth_suite.sh \
  "${COMMON[@]}" \
  --experiment-id "$EXPERIMENT_ID" \
  "${EXTRA[@]}"
