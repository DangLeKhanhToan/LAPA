#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_DIR="$( cd -- "$( dirname -- "$SCRIPT_DIR" )" &> /dev/null && pwd )"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"

: "${LAPA_JSONL:?Set LAPA_JSONL to the full LAPA JSONL, e.g. datasets/libero_data/libero_90_train.jsonl.}"

TASK_CONTAINS="${TASK_CONTAINS:-KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet_and_put_the_black_bowl_on_top_of_it_demo}"
MAX_ROWS="${MAX_ROWS:-512}"
SMOKE_JSONL="${SMOKE_JSONL:-datasets/smoke/libero_90_one_task_train.jsonl}"

python3 -m latent_pretraining.depth_fusion.make_one_task_jsonl \
  --input_jsonl "$LAPA_JSONL" \
  --output_jsonl "$SMOKE_JSONL" \
  --task_contains "$TASK_CONTAINS" \
  --max_rows "$MAX_ROWS"

echo "[smoke-data] wrote $SMOKE_JSONL"
