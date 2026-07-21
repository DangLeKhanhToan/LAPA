export SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
export PROJECT_DIR="$( cd -- "$( dirname -- "$SCRIPT_DIR" )" &> /dev/null && pwd )"
cd $PROJECT_DIR
export PYTHONPATH="$PYTHONPATH:$PROJECT_DIR"
export LIBTPU_INIT_ARGS="--xla_tpu_megacore_fusion_allow_ags=false --xla_enable_async_collective_permute=true --xla_tpu_enable_ag_backward_pipelining=true --xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_data_parallel_opt_different_sized_ops=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true"

# Fine-tune LAPA on a single LIBERO suite (libero_object), trained on the whole suite.
# Data produced by:  python data/process_libero.py --output_dir datasets/lapa_libero_v1

export absolute_path="/home/users/create/smrvmdo/scratch/projects/lapa_finetune/LAPA"
export llama_tokenizer_path="$absolute_path/lapa_checkpoints/tokenizer.model"
export output_dir="$absolute_path/outputs"
mkdir -p "$output_dir"

export project_id='lapa'

# ── Suite selection ───────────────────────────────────────────────────────────
export suite='libero_object'
export experiment_id="finetune_${suite}"
export experiment_note="lapa_finetune_${suite}"

export data_dir="$absolute_path/datasets/lapa_libero_v1"
export dataset_path="$data_dir/${suite}.jsonl"
export bins_csv="$data_dir/action_bins_${suite}.csv"

# ── Training config ───────────────────────────────────────────────────────────
export mesh_dim='!-1,4,1,1'          # 4 GPUs (dp inferred, fsdp=4)
export total_steps=20000
export save_milestone_freq=2000
export load_checkpoint="$absolute_path/lapa_checkpoints/streaming_params_22485"
export batch_size=128

# action_vocab_size = number of columns in this suite's bins CSV (repo convention,
# = #bin edges = max #bins + 1; same rule as scripts/finetune_simpler.sh = 245).
# Per-suite tokenization; the pretrained action head is re-initialized on load,
# so a per-suite vocab is safe (train.py inits target params missing from ckpt).
if [ ! -f "$dataset_path" ]; then
    echo "ERROR: $dataset_path not found. Run data/process_libero.py --output_dir datasets/lapa_libero_v1 first." >&2
    exit 1
fi
if [ ! -f "$bins_csv" ]; then
    echo "ERROR: $bins_csv not found." >&2
    exit 1
fi
export action_vocab_size=$(python3 -c "import pandas as pd; print(pd.read_csv('$bins_csv').shape[1])")
echo "[$suite] action_vocab_size=$action_vocab_size  dataset=$dataset_path"

python3 -u -m latent_pretraining.train \
    --modality='vision,action,delta' \
    --mesh_dim="$mesh_dim" \
    --dtype='bf16' \
    --total_steps="$total_steps" \
    --log_freq=1 \
    --eval_steps=0 \
    --save_model_freq=0 \
    --eval_log_freq=100 \
    --save_milestone_freq="$save_milestone_freq" \
    --load_llama_config='7b' \
    --load_checkpoint="params::$load_checkpoint" \
    --update_llama_config="dict(action_vocab_size=$action_vocab_size,delta_vocab_size=8,theta=50000000,max_sequence_length=2048,use_flash_attention=True,scan_attention=True,scan_query_chunk_size=512,scan_key_chunk_size=1024,remat_attention='nothing_saveable',scan_mlp=True,scan_mlp_chunk_size=8192,remat_mlp='nothing_saveable',remat_block='nothing_saveable',scan_layers=True)" \
    --tokenizer.vocab_file="$llama_tokenizer_path" \
    --optimizer.type='adamw' \
    --llama.action_vocab_size="$action_vocab_size" \
    --llama.delta_vocab_size=8 \
    --optimizer.accumulate_gradient_steps=1 \
    --optimizer.adamw_optimizer.weight_decay=0 \
    --optimizer.adamw_optimizer.lr=2e-5 \
    --optimizer.adamw_optimizer.end_lr=2e-5 \
    --optimizer.adamw_optimizer.lr_warmup_steps=0 \
    --optimizer.adamw_optimizer.lr_decay_steps=100 \
    --use_data_sharded_loader=True \
    --train_dataset.type='json_vision_delta_action' \
    --train_dataset.delta_vision_action_processor.fields_from_example='fields' \
    --train_dataset.delta_vision_action_processor.n_tokens_per_action=7 \
    --train_dataset.delta_vision_action_processor.n_tokens_per_delta=4 \
    --train_dataset.delta_vision_action_processor.img_aug=True \
    --train_dataset.delta_vision_action_processor.vqgan_checkpoint_path="$absolute_path/lapa_checkpoints/vqgan" \
    --train_dataset.delta_vision_action_processor.image_absolute_path="$data_dir/" \
    --train_dataset.delta_vision_action_processor.max_n_frames=1 \
    --train_dataset.json_delta_action_dataset.mode="pad" \
    --train_dataset.json_delta_action_dataset.path="$dataset_path" \
    --train_dataset.json_delta_action_dataset.seq_length=384 \
    --train_dataset.json_delta_action_dataset.batch_size="$batch_size" \
    --train_dataset.json_delta_action_dataset.tokenizer_processes=1 \
    --train_dataset.json_delta_action_dataset.tokenizer_parallel_chunk_size=128 \
    --train_dataset.json_delta_action_dataset.tokenizer_parallel_batch_size=128 \
    --train_dataset.json_delta_action_dataset.use_data_sharded_loader=True \
    --checkpointer.save_optimizer_state=False \
    --autoresume=False \
    --logger.append_uuid=False \
    --logger.online=False \
    --logger.project_id="$project_id" \
    --logger.experiment_id="$experiment_id" \
    --logger.experiment_note="$experiment_note" \
    --logger.output_dir="$output_dir" \
    --logger.wandb_dir="$output_dir"
