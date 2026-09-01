#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

MODEL="${MODEL:-model2}"
DATA_ROOT="${DATA_ROOT:-$ROOT/datasets/lapa_libero_v4}"
FEATURE_ROOT="${FEATURE_ROOT:-$ROOT/datasets/features_depth_branch}"
SUITES="${SUITES:-libero_spatial libero_object libero_goal libero_90}"

if [[ -f "$DATA_ROOT/all_train.jsonl" ]]; then
  TRAIN_JSONL="$DATA_ROOT/all_train.jsonl"
else
  TRAIN_JSONL="$DATA_ROOT/combined/all_suite_train.jsonl"
  mkdir -p "$(dirname "$TRAIN_JSONL")"
  : > "$TRAIN_JSONL"
  for suite in $SUITES; do
    if [[ -f "$DATA_ROOT/${suite}_train.jsonl" ]]; then
      source_jsonl="$DATA_ROOT/${suite}_train.jsonl"
    else
      source_jsonl="$DATA_ROOT/${suite}.jsonl"
    fi
    [[ -f "$source_jsonl" ]] || { echo "ERROR: missing $source_jsonl" >&2; exit 1; }
    cat "$source_jsonl" >> "$TRAIN_JSONL"
  done
fi

DEPTH_DIRS=""
MANIFESTS=""
for suite in $SUITES; do
  dir="$FEATURE_ROOT/stage25_libero_features_${MODEL}/$suite/stage25_${MODEL}/z_depth_train_shard0"
  [[ -d "$dir" ]] || { echo "ERROR: missing depth directory $dir" >&2; exit 1; }
  mapfile -t matches < <(find "$dir" -maxdepth 1 -type f -name '*manifest.json' | sort)
  [[ "${#matches[@]}" -eq 1 ]] || {
    echo "ERROR: expected one manifest in $dir, found ${#matches[@]}" >&2
    exit 1
  }
  DEPTH_DIRS="${DEPTH_DIRS:+$DEPTH_DIRS,}$dir"
  MANIFESTS="${MANIFESTS:+$MANIFESTS,}${matches[0]}"
done

OUTPUT="${DEPTH_BUNDLE:-$DATA_ROOT/packed_depth/${MODEL}_all_suite_fp16.pt}"
echo "[prepare] JSONL: $TRAIN_JSONL"
echo "[prepare] output: $OUTPUT"
python scripts/build_lapa_depth_bundle.py \
  --jsonl "$TRAIN_JSONL" \
  --depth-dirs "$DEPTH_DIRS" \
  --manifests "$MANIFESTS" \
  --output "$OUTPUT"

echo "[prepare] PASS: $OUTPUT"
echo "[prepare] Use TRAIN_JSONL=$TRAIN_JSONL for both baseline and depth runs."
