#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_DIR="$( cd -- "$( dirname -- "$SCRIPT_DIR" )" &> /dev/null && pwd )"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export LIBTPU_INIT_ARGS="${LIBTPU_INIT_ARGS:---xla_tpu_megacore_fusion_allow_ags=false --xla_enable_async_collective_permute=true --xla_tpu_enable_ag_backward_pipelining=true --xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_data_parallel_opt_different_sized_ops=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true}"

LAPA_ROOT="${LAPA_ROOT:-$PROJECT_DIR}"
SUITE="${SUITE:-libero_90}"
DATA_ROOT="${DATA_ROOT:-$LAPA_ROOT/datasets/lapa_libero_v2}"
TRAIN_JSONL="${TRAIN_JSONL:-$DATA_ROOT/${SUITE}.jsonl}"
if [[ -z "${IMAGE_ROOT:-}" ]]; then
  if [[ -d "$DATA_ROOT/images" ]]; then
    IMAGE_ROOT="$DATA_ROOT/"
  else
    IMAGE_ROOT="$LAPA_ROOT/datasets/libero_data/"
  fi
fi
ACTION_SCALE_FILE="${ACTION_SCALE_FILE:-$DATA_ROOT/action_bins_${SUITE}.csv}"
DEPTH_BASE_DIR="${DEPTH_BASE_DIR:-$LAPA_ROOT/datasets/features_depth_branch/stage25_libero_features_model4/${SUITE}/stage25_model4}"
DEPTH_DATA_DIR="${DEPTH_DATA_DIR:-}"
DEPTH_MANIFEST="${DEPTH_MANIFEST:-}"

if [[ -z "$DEPTH_DATA_DIR" ]]; then
  if compgen -G "$DEPTH_BASE_DIR/*_part*.pt" >/dev/null || compgen -G "$DEPTH_BASE_DIR/*_part*.pth" >/dev/null; then
    DEPTH_DATA_DIR="$DEPTH_BASE_DIR"
  elif [[ -d "$DEPTH_BASE_DIR/z_depth_train_shard0" ]]; then
    DEPTH_DATA_DIR="$DEPTH_BASE_DIR/z_depth_train_shard0"
  else
    DEPTH_DATA_DIR="$DEPTH_BASE_DIR"
  fi
fi

if [[ -z "$DEPTH_MANIFEST" ]]; then
  for candidate in \
    "$DEPTH_DATA_DIR/z_depth_train_model4_manifest.json" \
    "$DEPTH_DATA_DIR/z_depth_train_shard0_model4_manifest.json" \
    "$DEPTH_DATA_DIR"/*_manifest.json; do
    if [[ -f "$candidate" ]]; then
      DEPTH_MANIFEST="$candidate"
      break
    fi
  done
fi

if [[ ! -f "$TRAIN_JSONL" ]]; then
  echo "ERROR: train JSONL not found: $TRAIN_JSONL" >&2
  exit 1
fi
if [[ ! -f "$ACTION_SCALE_FILE" ]]; then
  echo "ERROR: action bins CSV not found: $ACTION_SCALE_FILE" >&2
  exit 1
fi
if [[ ! -d "$DEPTH_DATA_DIR" ]]; then
  echo "ERROR: depth feature directory not found: $DEPTH_DATA_DIR" >&2
  exit 1
fi

ACTION_VOCAB_SIZE="${ACTION_VOCAB_SIZE:-$(head -1 "$ACTION_SCALE_FILE" | awk -F, '{print NF}')}"
TOKENIZER_PATH="${TOKENIZER_PATH:-$LAPA_ROOT/lapa_checkpoints/tokenizer.model}"
VQGAN_CKPT="${VQGAN_CKPT:-$LAPA_ROOT/lapa_checkpoints/vqgan}"
LAPA_PARAMS="${LAPA_PARAMS:-$LAPA_ROOT/lapa_checkpoints/lapa_7b_sth/params}"
OUTPUT_DIR="${OUTPUT_DIR:-$LAPA_ROOT/outputs}"
PROJECT_ID="${PROJECT_ID:-lapa_depth}"
EXPERIMENT_ID="${EXPERIMENT_ID:-lapa_depth_stage3_${SUITE}}"
EXPERIMENT_NOTE="${EXPERIMENT_NOTE:-stage3_${SUITE}_depth_offline}"
TOTAL_STEPS="${TOTAL_STEPS:-20000}"
BATCH_SIZE="${BATCH_SIZE:-128}"
SEQ_LENGTH="${SEQ_LENGTH:-384}"
MESH_DIM="${MESH_DIM:-!-1,4,1,1}"
LR="${LR:-2e-5}"
DEPTH_ID_KEY="${DEPTH_ID_KEY:-auto}"
DEPTH_FEATURE_KEY="${DEPTH_FEATURE_KEY:-auto}"
DEPTH_FEATURE_DIM="${DEPTH_FEATURE_DIM:-1024}"
JSON_ID_KEY="${JSON_ID_KEY:-id}"
JSON_ID_SOURCE="${JSON_ID_SOURCE:-auto}"
WANDB_ONLINE="${WANDB_ONLINE:-False}"
WANDB_DIR="${WANDB_DIR:-$OUTPUT_DIR/$EXPERIMENT_ID/wandb}"
LOG_FREQ="${LOG_FREQ:-1}"
EVAL_STEPS="${EVAL_STEPS:-0}"
EVAL_LOG_FREQ="${EVAL_LOG_FREQ:-100}"
SAVE_MODEL_FREQ="${SAVE_MODEL_FREQ:-$TOTAL_STEPS}"
SAVE_MILESTONE_FREQ="${SAVE_MILESTONE_FREQ:-0}"
RUNTIME_LOG_STEPS="${RUNTIME_LOG_STEPS:-${runtime_log_steps:-3}}"

args=(
  -u -m latent_pretraining.train
  --modality="vision,action,delta"
  --mesh_dim="$MESH_DIM"
  --dtype="bf16"
  --total_steps="$TOTAL_STEPS"
  --log_freq="$LOG_FREQ"
  --eval_steps="$EVAL_STEPS"
  --save_model_freq="$SAVE_MODEL_FREQ"
  --eval_log_freq="$EVAL_LOG_FREQ"
  --save_milestone_freq="$SAVE_MILESTONE_FREQ"
  --runtime_log_steps="$RUNTIME_LOG_STEPS"
  --load_llama_config="7b"
  --load_checkpoint="params::$LAPA_PARAMS"
  --update_llama_config="dict(action_vocab_size=${ACTION_VOCAB_SIZE},delta_vocab_size=8,theta=50000000,max_sequence_length=2048,use_flash_attention=True,scan_attention=True,scan_query_chunk_size=512,scan_key_chunk_size=1024,remat_attention='nothing_saveable',scan_mlp=True,scan_mlp_chunk_size=8192,remat_mlp='nothing_saveable',remat_block='nothing_saveable',scan_layers=True)"
  --tokenizer.vocab_file="$TOKENIZER_PATH"
  --optimizer.type="adamw"
  --llama.action_vocab_size="$ACTION_VOCAB_SIZE"
  --llama.delta_vocab_size=8
  --optimizer.accumulate_gradient_steps=1
  --optimizer.adamw_optimizer.weight_decay=0
  --optimizer.adamw_optimizer.lr="$LR"
  --optimizer.adamw_optimizer.end_lr="$LR"
  --optimizer.adamw_optimizer.lr_warmup_steps=0
  --optimizer.adamw_optimizer.lr_decay_steps="$TOTAL_STEPS"
  --freeze_vision_params=True
  --use_data_sharded_loader=True
  --train_dataset.type="json_vision_delta_action"
  --train_dataset.delta_vision_action_processor.fields_from_example="fields"
  --train_dataset.delta_vision_action_processor.sample_id_key="$JSON_ID_KEY"
  --train_dataset.delta_vision_action_processor.sample_id_source="$JSON_ID_SOURCE"
  --train_dataset.delta_vision_action_processor.n_tokens_per_action=7
  --train_dataset.delta_vision_action_processor.n_tokens_per_delta=4
  --train_dataset.delta_vision_action_processor.img_aug=True
  --train_dataset.delta_vision_action_processor.vqgan_checkpoint_path="$VQGAN_CKPT"
  --train_dataset.delta_vision_action_processor.image_absolute_path="$IMAGE_ROOT"
  --train_dataset.delta_vision_action_processor.max_n_frames=1
  --train_dataset.json_delta_action_dataset.mode="pad"
  --train_dataset.json_delta_action_dataset.path="$TRAIN_JSONL"
  --train_dataset.json_delta_action_dataset.seq_length="$SEQ_LENGTH"
  --train_dataset.json_delta_action_dataset.batch_size="$BATCH_SIZE"
  --train_dataset.json_delta_action_dataset.tokenizer_processes=1
  --train_dataset.json_delta_action_dataset.tokenizer_parallel_chunk_size=128
  --train_dataset.json_delta_action_dataset.tokenizer_parallel_batch_size=128
  --train_dataset.json_delta_action_dataset.use_data_sharded_loader=True
  --train_dataset.json_delta_action_dataset.depth_feature_data_dir="$DEPTH_DATA_DIR"
  --train_dataset.json_delta_action_dataset.depth_feature_key="$DEPTH_FEATURE_KEY"
  --train_dataset.json_delta_action_dataset.depth_feature_id_key="$DEPTH_ID_KEY"
  --train_dataset.json_delta_action_dataset.depth_feature_dim="$DEPTH_FEATURE_DIM"
  --checkpointer.save_optimizer_state=False
  --autoresume=False
  --logger.append_uuid=False
  --logger.online="$WANDB_ONLINE"
  --logger.project_id="$PROJECT_ID"
  --logger.experiment_id="$EXPERIMENT_ID"
  --logger.experiment_note="$EXPERIMENT_NOTE"
  --logger.output_dir="$OUTPUT_DIR"
  --logger.wandb_dir="$WANDB_DIR"
)

if [[ -n "$DEPTH_MANIFEST" ]]; then
  args+=(--train_dataset.json_delta_action_dataset.depth_feature_manifest="$DEPTH_MANIFEST")
fi

echo "[train-depth-suite] suite: $SUITE"
echo "[train-depth-suite] train jsonl: $TRAIN_JSONL"
echo "[train-depth-suite] image root: $IMAGE_ROOT"
echo "[train-depth-suite] action bins: $ACTION_SCALE_FILE"
echo "[train-depth-suite] action_vocab_size: $ACTION_VOCAB_SIZE"
echo "[train-depth-suite] depth dir: $DEPTH_DATA_DIR"
echo "[train-depth-suite] depth manifest: ${DEPTH_MANIFEST:-<none>}"
echo "[train-depth-suite] output: $OUTPUT_DIR/$EXPERIMENT_ID"
echo "[train-depth-suite] mesh: $MESH_DIM"
echo "[train-depth-suite] total_steps: $TOTAL_STEPS"
echo "[train-depth-suite] batch_size: $BATCH_SIZE"
echo "[train-depth-suite] tokenizer_processes: 1"
echo "[train-depth-suite] log_freq: $LOG_FREQ"
echo "[train-depth-suite] save_model_freq: $SAVE_MODEL_FREQ"

python3 "${args[@]}"

echo "[train-depth-suite] checkpoint: $OUTPUT_DIR/$EXPERIMENT_ID/streaming_params"
