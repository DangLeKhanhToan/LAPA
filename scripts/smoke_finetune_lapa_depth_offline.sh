#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_DIR="$( cd -- "$( dirname -- "$SCRIPT_DIR" )" &> /dev/null && pwd )"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export LIBTPU_INIT_ARGS="${LIBTPU_INIT_ARGS:---xla_tpu_megacore_fusion_allow_ags=false --xla_enable_async_collective_permute=true --xla_tpu_enable_ag_backward_pipelining=true --xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_data_parallel_opt_different_sized_ops=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true}"

: "${LAPA_ROOT:?Set LAPA_ROOT to the directory containing lapa_checkpoints and data.}"
: "${DATASET_JSONL:?Set DATASET_JSONL to the LAPA fine-tuning JSONL.}"
: "${DEPTH_DATA_DIR:?Set DEPTH_DATA_DIR to the directory containing depth .pt/.pth parts.}"

DEPTH_MANIFEST="${DEPTH_MANIFEST:-}"
JSON_ID_KEY="${JSON_ID_KEY:-id}"
JSON_ID_SOURCE="${JSON_ID_SOURCE:-auto}"
DEPTH_ID_KEY="${DEPTH_ID_KEY:-auto}"
DEPTH_FEATURE_KEY="${DEPTH_FEATURE_KEY:-auto}"
DEPTH_FEATURE_DIM="${DEPTH_FEATURE_DIM:-1024}"
IMAGE_ROOT="${IMAGE_ROOT:-$LAPA_ROOT/data/}"
TOKENIZER_PATH="${TOKENIZER_PATH:-$LAPA_ROOT/lapa_checkpoints/tokenizer.model}"
VQGAN_CKPT="${VQGAN_CKPT:-$LAPA_ROOT/lapa_checkpoints/vqgan}"
LAPA_PARAMS="${LAPA_PARAMS:-$LAPA_ROOT/lapa_checkpoints/params}"
OUTPUT_DIR="${OUTPUT_DIR:-$LAPA_ROOT/outputs}"
PROJECT_ID="${PROJECT_ID:-lapa}"
EXPERIMENT_ID="${EXPERIMENT_ID:-smoke_lapa_depth_offline}"
EXPERIMENT_NOTE="${EXPERIMENT_NOTE:-smoke_lapa_depth_offline}"
TOTAL_STEPS="${TOTAL_STEPS:-1}"
BATCH_SIZE="${BATCH_SIZE:-8}"
SEQ_LENGTH="${SEQ_LENGTH:-384}"
MESH_DIM="${MESH_DIM:-!-1,4,1,1}"
WANDB_ONLINE="${WANDB_ONLINE:-False}"

args=(
  -u -m latent_pretraining.train
  --modality="vision,action,delta"
  --mesh_dim="$MESH_DIM"
  --dtype="bf16"
  --total_steps="$TOTAL_STEPS"
  --log_freq=1
  --eval_steps=0
  --save_model_freq=0
  --eval_log_freq=100
  --save_milestone_freq=0
  --load_llama_config="7b"
  --load_checkpoint="params::$LAPA_PARAMS"
  --update_llama_config="dict(action_vocab_size=256,delta_vocab_size=8,theta=50000000,max_sequence_length=2048,use_flash_attention=True,scan_attention=True,scan_query_chunk_size=512,scan_key_chunk_size=1024,remat_attention='nothing_saveable',scan_mlp=True,scan_mlp_chunk_size=8192,remat_mlp='nothing_saveable',remat_block='nothing_saveable',scan_layers=True)"
  --tokenizer.vocab_file="$TOKENIZER_PATH"
  --optimizer.type="adamw"
  --llama.action_vocab_size=256
  --llama.delta_vocab_size=8
  --optimizer.accumulate_gradient_steps=1
  --optimizer.adamw_optimizer.weight_decay=0
  --optimizer.adamw_optimizer.lr=2e-5
  --optimizer.adamw_optimizer.end_lr=2e-5
  --optimizer.adamw_optimizer.lr_warmup_steps=0
  --optimizer.adamw_optimizer.lr_decay_steps=100
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
  --train_dataset.json_delta_action_dataset.path="$DATASET_JSONL"
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
  --logger.wandb_dir="$HOME/experiment_output/$PROJECT_ID"
)

if [[ -n "$DEPTH_MANIFEST" ]]; then
  args+=(--train_dataset.json_delta_action_dataset.depth_feature_manifest="$DEPTH_MANIFEST")
fi

python3 "${args[@]}"
