#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_DIR="$( cd -- "$( dirname -- "$SCRIPT_DIR" )" &> /dev/null && pwd )"
cd "$PROJECT_DIR"

usage() {
  cat <<'EOF'
Usage: bash scripts/eval_lapa_baseline_rollout.sh [options]

Required:
  --checkpoint PATH       Policy streaming_params.
  --action-bins PATH      Training action-bin CSV.
  --suites "LIST"         Space-separated suite names.
  --output-dir PATH       Unique rollout directory.

Runtime:
  --lapa-root PATH        Source root. Default: repository root.
  --model-python PATH     JAX policy Python.
  --libero-python PATH    Simulator Python.
  --libero-repo PATH      LIBERO/LIBERO-Pro clone root.
  --libero-config PATH    LIBERO_CONFIG_PATH.
  --model-ld-path PATH    Policy LD_LIBRARY_PATH. Default: empty.
  --libero-ld-path PATH   Simulator LD_LIBRARY_PATH.
  --policy-gpu ID         Default: 2.
  --egl-gpu ID            Default: 3.
  --policy-port PORT      Default: 32931.

Evaluation:
  --task-ids "LIST"       Empty means all tasks.
  --episodes N            Episodes per task. Default: 1.
  --max-steps N           Default: 80.
  --init-offset N         Default: 0.
  --track-diagnostics     Save trajectory/grasp/object telemetry.
  --reference-suite NAME  Default: auto.
  --approach-threshold M  Default: 0.10.
  --expected-episodes N   Require an exact final episode count.
EOF
}

LAPA_ROOT="$PROJECT_DIR"
CHECKPOINT=""
ACTION_BINS=""
SUITES=""
OUTPUT_DIR=""
MODEL_PY="${MODEL_PY:-python3}"
LIBERO_PY="${LIBERO_PY:-python3}"
LIBERO_REPO="${LIBERO_REPO:-$PROJECT_DIR/datasets/LIBERO}"
LIBERO_CONFIG="${LIBERO_CONFIG_PATH:-}"
MODEL_LD="${MODEL_LD_LIBRARY_PATH:-}"
LIBERO_LD="${LIBERO_LD_LIBRARY_PATH:-${LD_LIBRARY_PATH:-}}"
POLICY_GPU="2"
EGL_GPU="3"
POLICY_PORT="32931"
TASK_IDS=""
EPISODES="1"
MAX_STEPS="80"
INIT_OFFSET="0"
TRACK_DIAGNOSTICS="false"
REFERENCE_SUITE="auto"
APPROACH_THRESHOLD="0.10"
EXPECTED_EPISODES=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --lapa-root) LAPA_ROOT="$2"; shift 2 ;;
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --action-bins) ACTION_BINS="$2"; shift 2 ;;
    --suites) SUITES="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --model-python) MODEL_PY="$2"; shift 2 ;;
    --libero-python) LIBERO_PY="$2"; shift 2 ;;
    --libero-repo) LIBERO_REPO="$2"; shift 2 ;;
    --libero-config) LIBERO_CONFIG="$2"; shift 2 ;;
    --model-ld-path) MODEL_LD="$2"; shift 2 ;;
    --libero-ld-path) LIBERO_LD="$2"; shift 2 ;;
    --policy-gpu) POLICY_GPU="$2"; shift 2 ;;
    --egl-gpu) EGL_GPU="$2"; shift 2 ;;
    --policy-port) POLICY_PORT="$2"; shift 2 ;;
    --task-ids) TASK_IDS="$2"; shift 2 ;;
    --episodes) EPISODES="$2"; shift 2 ;;
    --max-steps) MAX_STEPS="$2"; shift 2 ;;
    --init-offset) INIT_OFFSET="$2"; shift 2 ;;
    --track-diagnostics) TRACK_DIAGNOSTICS="true"; shift ;;
    --reference-suite) REFERENCE_SUITE="$2"; shift 2 ;;
    --approach-threshold) APPROACH_THRESHOLD="$2"; shift 2 ;;
    --expected-episodes) EXPECTED_EPISODES="$2"; shift 2 ;;
    --help) usage; exit 0 ;;
    *) echo "ERROR: unknown option: $1" >&2; usage >&2; exit 1 ;;
  esac
done

[[ -n "$CHECKPOINT" && -e "$CHECKPOINT" ]] || { echo "ERROR: checkpoint not found: $CHECKPOINT" >&2; exit 1; }
[[ -n "$ACTION_BINS" && -f "$ACTION_BINS" ]] || { echo "ERROR: action bins not found: $ACTION_BINS" >&2; exit 1; }
[[ -n "$SUITES" ]] || { echo "ERROR: --suites is required" >&2; exit 1; }
[[ -n "$OUTPUT_DIR" ]] || { echo "ERROR: --output-dir is required" >&2; exit 1; }
[[ -x "$MODEL_PY" ]] || { echo "ERROR: model Python is not executable: $MODEL_PY" >&2; exit 1; }
[[ -x "$LIBERO_PY" ]] || { echo "ERROR: simulator Python is not executable: $LIBERO_PY" >&2; exit 1; }

VQGAN="${VQGAN_CHECKPOINT:-$LAPA_ROOT/lapa_checkpoints/vqgan}"
if [[ -f "$LAPA_ROOT/lapa_checkpoints/lapa_7b_sth/tokenizer.model" ]]; then
  VOCAB="${VOCAB_FILE:-$LAPA_ROOT/lapa_checkpoints/lapa_7b_sth/tokenizer.model}"
else
  VOCAB="${VOCAB_FILE:-$LAPA_ROOT/lapa_checkpoints/tokenizer.model}"
fi
ACTION_VOCAB_SIZE="$(awk -F, 'NR == 1 {print NF}' "$ACTION_BINS")"
LOG_DIR="$OUTPUT_DIR/server_logs"
mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

# Use the exact simulator environment for a fail-fast suite registry check.
LD_LIBRARY_PATH="$LIBERO_LD" LIBERO_CONFIG_PATH="$LIBERO_CONFIG" \
PYTHONPATH="$LIBERO_REPO:$LAPA_ROOT" "$LIBERO_PY" - $SUITES <<'PY'
import sys
from libero.libero import benchmark

requested = sys.argv[1:]
available = benchmark.get_benchmark_dict()
missing = [name for name in requested if name not in available]
print(f"[baseline] available suites: {len(available)}")
print(f"[baseline] requested suites: {' '.join(requested)}")
if missing:
    raise SystemExit(f"ERROR: missing simulator suites: {missing}")
print("[baseline] suite registry: PASS")
PY

echo "[baseline] checkpoint: params::$CHECKPOINT"
echo "[baseline] action bins: $ACTION_BINS"
echo "[baseline] action vocabulary: $ACTION_VOCAB_SIZE"
echo "[baseline] suites: $SUITES"
echo "[baseline] task ids: ${TASK_IDS:-all}"
echo "[baseline] episodes/task: $EPISODES | max steps: $MAX_STEPS"
echo "[baseline] output: $OUTPUT_DIR"

POLICY_PID=""
cleanup() {
  [[ -z "$POLICY_PID" ]] || kill "$POLICY_PID" 2>/dev/null || true
  [[ -z "$POLICY_PID" ]] || wait "$POLICY_PID" 2>/dev/null || true
}
trap cleanup EXIT

LD_LIBRARY_PATH="$MODEL_LD" PYTHONPATH="$LAPA_ROOT" \
CUDA_VISIBLE_DEVICES="$POLICY_GPU" JAX_PLATFORMS="${JAX_PLATFORMS:-cuda,cpu}" \
XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}" \
XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.80}" \
"$MODEL_PY" -m latent_pretraining.deploy \
  --load_checkpoint "params::$CHECKPOINT" \
  --action_scale_file "$ACTION_BINS" \
  --vqgan_checkpoint "$VQGAN" \
  --vocab_file "$VOCAB" \
  --update_llama_config "dict(action_vocab_size=${ACTION_VOCAB_SIZE},delta_vocab_size=8,sample_mode='text',theta=50000000,max_sequence_length=32768,scan_attention=False,scan_query_chunk_size=128,scan_key_chunk_size=128,scan_mlp=False,scan_mlp_chunk_size=8192,scan_layers=True)" \
  --port "$POLICY_PORT" --mesh_dim "1,1,1,1" \
  --tokens_per_delta 4 --tokens_per_action 7 \
  > "$LOG_DIR/policy.log" 2>&1 &
POLICY_PID=$!

ready="false"
for _ in $(seq 1 "${SERVER_WAIT_ATTEMPTS:-90}"); do
  if grep -q "Uvicorn running" "$LOG_DIR/policy.log" 2>/dev/null; then ready="true"; break; fi
  if ! kill -0 "$POLICY_PID" 2>/dev/null; then break; fi
  sleep 5
done
if [[ "$ready" != "true" ]]; then
  tail -n 120 "$LOG_DIR/policy.log" >&2
  echo "ERROR: policy server did not become ready" >&2
  exit 1
fi
echo "[baseline] policy server: READY"

task_args=()
if [[ -n "$TASK_IDS" ]]; then read -r -a ids <<< "$TASK_IDS"; task_args+=(--task_ids "${ids[@]}"); fi
diag_args=()
if [[ "$TRACK_DIAGNOSTICS" == "true" ]]; then
  diag_args+=(--track_diagnostics --reference_suite "$REFERENCE_SUITE" --approach_threshold "$APPROACH_THRESHOLD")
fi

LD_LIBRARY_PATH="$LIBERO_LD" LIBERO_CONFIG_PATH="$LIBERO_CONFIG" \
PYTHONPATH="$LIBERO_REPO:$LAPA_ROOT" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
MUJOCO_EGL_DEVICE_ID="$EGL_GPU" "$LIBERO_PY" "$LAPA_ROOT/eval/eval_libero_rollout.py" \
  --server_url "http://127.0.0.1:${POLICY_PORT}/act" --output_dir "$OUTPUT_DIR" \
  --suites $SUITES --n_eval_per_task "$EPISODES" --max_steps "$MAX_STEPS" \
  --init_offset "$INIT_OFFSET" "${task_args[@]}" "${diag_args[@]}"

LD_LIBRARY_PATH="$LIBERO_LD" PYTHONPATH="$LAPA_ROOT" \
"$LIBERO_PY" - "$OUTPUT_DIR/results.json" "$EXPECTED_EPISODES" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as stream:
    results = json.load(stream)
n_eval = int(results.get("overall", {}).get("n_eval", 0))
expected = sys.argv[2]
print(f"[baseline] evaluated episodes: {n_eval}")
if n_eval <= 0:
    raise SystemExit("ERROR: rollout evaluated zero episodes")
if expected and n_eval != int(expected):
    raise SystemExit(f"ERROR: expected {expected} episodes, got {n_eval}")
print("[baseline] episode count: PASS")
PY

if [[ "$TRACK_DIAGNOSTICS" == "true" ]]; then
  LD_LIBRARY_PATH="$LIBERO_LD" PYTHONPATH="$LAPA_ROOT" "$LIBERO_PY" \
    -m eval.summarize_libero_diagnostics "$OUTPUT_DIR" \
    --csv "$OUTPUT_DIR/diagnostic_summary.csv" \
    --json "$OUTPUT_DIR/diagnostic_summary.json" \
    --markdown "$OUTPUT_DIR/diagnostic_summary.md"
fi
echo "[baseline] complete: $OUTPUT_DIR"
