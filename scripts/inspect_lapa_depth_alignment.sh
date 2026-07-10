#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_DIR="$( cd -- "$( dirname -- "$SCRIPT_DIR" )" &> /dev/null && pwd )"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"

: "${LAPA_JSONL:?Set LAPA_JSONL to the LAPA fine-tuning JSONL path.}"
: "${DEPTH_DATA_DIR:?Set DEPTH_DATA_DIR to the directory containing depth .pt/.pth parts.}"

DEPTH_MANIFEST="${DEPTH_MANIFEST:-}"
JSON_ID_KEY="${JSON_ID_KEY:-id}"
JSON_ID_SOURCE="${JSON_ID_SOURCE:-auto}"
DEPTH_ID_KEY="${DEPTH_ID_KEY:-auto}"
DEPTH_FEATURE_KEY="${DEPTH_FEATURE_KEY:-auto}"
SAMPLE_COUNT="${SAMPLE_COUNT:-5}"

args=(
  -m latent_pretraining.depth_fusion.inspect_lapa_depth_alignment
  --jsonl "$LAPA_JSONL"
  --depth_data_dir "$DEPTH_DATA_DIR"
  --json_id_key "$JSON_ID_KEY"
  --json_id_source "$JSON_ID_SOURCE"
  --depth_id_key "$DEPTH_ID_KEY"
  --depth_feature_key "$DEPTH_FEATURE_KEY"
  --sample_count "$SAMPLE_COUNT"
)

if [[ -n "$DEPTH_MANIFEST" ]]; then
  args+=(--depth_manifest "$DEPTH_MANIFEST")
fi

python3 "${args[@]}"
