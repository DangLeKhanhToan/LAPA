#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_DIR="$( cd -- "$( dirname -- "$SCRIPT_DIR" )" &> /dev/null && pwd )"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"

MODEL_PY="${MODEL_PY:-/mnt/hdd/linh/long/conda_envs/lapa-depth/bin/python}"
LAPA_ROOT="${LAPA_ROOT:-$PROJECT_DIR}"
SUITE="${SUITE:-libero_spatial}"
FINETUNED_CHECKPOINT="${FINETUNED_CHECKPOINT:-params::$LAPA_ROOT/outputs/lapa_depth_stage3_${SUITE}/streaming_params}"
OUTPUT_JSON="${OUTPUT_JSON:-$LAPA_ROOT/outputs/inspect_lapa_depth_policy_split_${SUITE}.json}"
INSPECT_CUDA_VISIBLE_DEVICES="${INSPECT_CUDA_VISIBLE_DEVICES:-}"

export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.25}"
export TF_FORCE_GPU_ALLOW_GROWTH="${TF_FORCE_GPU_ALLOW_GROWTH:-true}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu,cuda}"

echo "[inspect-split] checkpoint: $FINETUNED_CHECKPOINT"
echo "[inspect-split] output: $OUTPUT_JSON"
echo "[inspect-split] python: $MODEL_PY"
if [[ -n "$INSPECT_CUDA_VISIBLE_DEVICES" ]]; then
  echo "[inspect-split] visible GPUs: $INSPECT_CUDA_VISIBLE_DEVICES"
  CUDA_VISIBLE_DEVICES="$INSPECT_CUDA_VISIBLE_DEVICES" "$MODEL_PY" \
    -m latent_pretraining.inspect_lapa_checkpoint_split \
    --checkpoint "$FINETUNED_CHECKPOINT" \
    --output_json "$OUTPUT_JSON"
else
  "$MODEL_PY" \
    -m latent_pretraining.inspect_lapa_checkpoint_split \
    --checkpoint "$FINETUNED_CHECKPOINT" \
    --output_json "$OUTPUT_JSON"
fi
