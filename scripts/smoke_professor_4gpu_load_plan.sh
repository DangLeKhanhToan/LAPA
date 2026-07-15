#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_DIR="$( cd -- "$( dirname -- "$SCRIPT_DIR" )" &> /dev/null && pwd )"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"

MODEL_PY="${MODEL_PY:-/mnt/hdd/linh/long/conda_envs/lapa-depth/bin/python}"
LAPA_ROOT="${LAPA_ROOT:-$PROJECT_DIR}"
DEPTH_BRANCH_ROOT="${DEPTH_BRANCH_ROOT:-$LAPA_ROOT/Depth_branch}"
SUITE="${SUITE:-libero_spatial}"

FINETUNED_CHECKPOINT="${FINETUNED_CHECKPOINT:-params::$LAPA_ROOT/outputs/lapa_depth_stage3_${SUITE}/streaming_params}"
ORIGINAL_LAPA_CHECKPOINT="${ORIGINAL_LAPA_CHECKPOINT:-params::$LAPA_ROOT/lapa_checkpoints/pretraining_LAPA_Sth2Sth}"
ACTION_SCALE_FILE="${ACTION_SCALE_FILE:-$LAPA_ROOT/datasets/lapa_libero_v2/action_bins_${SUITE}.csv}"
VQGAN_CHECKPOINT="${VQGAN_CHECKPOINT:-$LAPA_ROOT/lapa_checkpoints/vqgan}"
VOCAB_FILE="${VOCAB_FILE:-$LAPA_ROOT/lapa_checkpoints/tokenizer.model}"

STAGE25_MODEL_NAME="${STAGE25_MODEL_NAME:-model4}"
STAGE25_MODEL_CHECKPOINT="${STAGE25_MODEL_CHECKPOINT:-$LAPA_ROOT/lapa_checkpoints/depth_model/${STAGE25_MODEL_NAME}.65000.pt}"
DEPTH_ANYTHING_REPO_DIR="${DEPTH_ANYTHING_REPO_DIR:-$LAPA_ROOT/third_party/depth_anything_v2}"
DEPTH_ANYTHING_CHECKPOINT="${DEPTH_ANYTHING_CHECKPOINT:-$LAPA_ROOT/checkpoints/depth_anything_v2_sth2sth/depth_anything_v2_sth2sth.pth}"
DEPTH_ANYTHING_ENCODER="${DEPTH_ANYTHING_ENCODER:-vitl}"

FINETUNED_GPU="${FINETUNED_GPU:-0}"
STAGE25_GPU="${STAGE25_GPU:-1}"
BASELINE_GPU="${BASELINE_GPU:-2}"
HEAD_GPU="${HEAD_GPU:-3}"

POLICY_PORT="${POLICY_PORT:-32920}"
STAGE25_PORT="${STAGE25_PORT:-32921}"

export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.55}"
export TF_FORCE_GPU_ALLOW_GROWTH="${TF_FORCE_GPU_ALLOW_GROWTH:-true}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda,cpu}"

ACTION_VOCAB_SIZE="${ACTION_VOCAB_SIZE:-$(head -1 "$ACTION_SCALE_FILE" | awk -F, '{print NF}')}"
UPDATE_LLAMA_CONFIG="${UPDATE_LLAMA_CONFIG:-dict(action_vocab_size=${ACTION_VOCAB_SIZE},delta_vocab_size=8,sample_mode='text',theta=50000000,max_sequence_length=32768,scan_attention=False,scan_query_chunk_size=128,scan_key_chunk_size=128,scan_mlp=False,scan_mlp_chunk_size=8192,scan_layers=True)}"

echo "[4gpu-plan] This script tests current load boundaries only."
echo "[4gpu-plan] The existing checkpoint does not expose a standalone 4096+1024 -> 7 action MLP service yet."
echo "[4gpu-plan] fine-tuned full policy GPU: $FINETUNED_GPU"
echo "[4gpu-plan] stage2.5/depthanything GPU: $STAGE25_GPU"
echo "[4gpu-plan] original LAPA baseline GPU: $BASELINE_GPU"
echo "[4gpu-plan] planned small head GPU: $HEAD_GPU"

echo "[4gpu-plan] 1/3 inspect fine-tuned checkpoint split"
INSPECT_CUDA_VISIBLE_DEVICES="$HEAD_GPU" OUTPUT_JSON="$LAPA_ROOT/outputs/professor_split_params_${SUITE}.json" \
  FINETUNED_CHECKPOINT="$FINETUNED_CHECKPOINT" "$SCRIPT_DIR/inspect_lapa_depth_policy_split.sh"

echo "[4gpu-plan] 2/3 smoke-load current full fine-tuned policy on one GPU"
echo "[4gpu-plan] If this OOMs, removing only action/depth heads is unlikely to fix memory."
CUDA_VISIBLE_DEVICES="$FINETUNED_GPU" "$MODEL_PY" -m latent_pretraining.deploy \
  --load_checkpoint "$FINETUNED_CHECKPOINT" \
  --action_scale_file "$ACTION_SCALE_FILE" \
  --vqgan_checkpoint "$VQGAN_CHECKPOINT" \
  --vocab_file "$VOCAB_FILE" \
  --update_llama_config "$UPDATE_LLAMA_CONFIG" \
  --port "$POLICY_PORT" \
  --mesh_dim "1,1,1,1" \
  --tokens_per_delta 4 \
  --tokens_per_action 7 &
POLICY_PID=$!
sleep "${LOAD_WAIT_SECONDS:-90}"
kill "$POLICY_PID" 2>/dev/null || true
wait "$POLICY_PID" 2>/dev/null || true

echo "[4gpu-plan] 3/3 smoke-load original LAPA baseline on one GPU"
cd "$DEPTH_BRANCH_ROOT"
PYTHONPATH="$DEPTH_BRANCH_ROOT:$DEPTH_BRANCH_ROOT/laq:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="$BASELINE_GPU" "$MODEL_PY" - <<PY
from rollout_stage25_model4 import build_lapa

lapa = build_lapa(
    tokens_per_delta=4,
    vqgan_checkpoint="${VQGAN_CHECKPOINT}",
    vocab_file="${VOCAB_FILE}",
    load_checkpoint="${ORIGINAL_LAPA_CHECKPOINT}",
    mesh_dim="1,1,1,1",
    dtype="bf16",
    load_llama_config="7b",
)
print("original LAPA baseline loaded OK on one GPU")
PY

echo "[4gpu-plan] load-plan smoke completed"
