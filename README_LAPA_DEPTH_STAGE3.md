# LAPA-Depth Stage 3 Offline Fine-Tuning

This guide uses the stable/original data path: one tokenizer process, online RGB
VQGAN encoding, offline Stage-2.5 depth features, and no multiprocessing data
workers.

## 1. Setup

Run from the project root on the server:

```bash
cd ~/scratch/projects/lapa-depth-modified-levi
export LAPA_ROOT="$(pwd -P)"
export PYTHONPATH="$LAPA_ROOT:${PYTHONPATH:-}"
```

Choose a suite:

```bash
export SUITE=libero_spatial
# export SUITE=libero_object
# export SUITE=libero_goal
# export SUITE=libero_10
# export SUITE=libero_90
```

## 2. Check Depth Feature Alignment

The JSONL rows must map to IDs inside the offline depth `.pt` shards. This must
show `match_rate: 1.0` before training.

```bash
export LAPA_JSONL="datasets/lapa_libero_v2/${SUITE}.jsonl"
export DEPTH_DATA_DIR="datasets/features_depth_branch/stage25_libero_features_model4/${SUITE}/stage25_model4/z_depth_train_shard0"

if [ "$SUITE" = "libero_90" ]; then
  export DEPTH_MANIFEST="$DEPTH_DATA_DIR/z_depth_train_model4_manifest.json"
else
  export DEPTH_MANIFEST="$DEPTH_DATA_DIR/z_depth_train_shard0_model4_manifest.json"
fi

bash scripts/inspect_lapa_depth_alignment.sh
```

Expected:

```text
"match_rate": 1.0
```

## 3. Smoke Overfit One Task

Use this before full training to verify image loading, depth lookup, model init,
and first train steps.

```bash
export SUITE=libero_spatial
export LAPA_JSONL="datasets/lapa_libero_v2/${SUITE}.jsonl"
export DEPTH_DATA_DIR="datasets/features_depth_branch/stage25_libero_features_model4/${SUITE}/stage25_model4/z_depth_train_shard0"

if [ "$SUITE" = "libero_90" ]; then
  export DEPTH_MANIFEST="$DEPTH_DATA_DIR/z_depth_train_model4_manifest.json"
else
  export DEPTH_MANIFEST="$DEPTH_DATA_DIR/z_depth_train_shard0_model4_manifest.json"
fi

export TASK_CONTAINS=pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo
export MAX_ROWS=512
export SMOKE_JSONL="datasets/smoke/${SUITE}_one_task_train.jsonl"

bash scripts/make_smoke_one_task_jsonl.sh
```

Train the smoke subset:

```bash
export TOTAL_STEPS=200
export BATCH_SIZE=8
export MESH_DIM='!-1,1,1,1'
export LOG_FREQ=1
export SAVE_MODEL_FREQ=200
export RUNTIME_LOG_STEPS=3
export EXPERIMENT_ID="smoke_overfit_${SUITE}_depth"

bash scripts/smoke_overfit_lapa_depth_one_task.sh
```

For 2 GPUs, use:

```bash
export CUDA_VISIBLE_DEVICES=0,1
export MESH_DIM='!-1,2,1,1'
```

## 4. Full Offline Stage-3 Training

Stable/original config:

- trainable: LAPA language model and action head
- frozen: vision params, VQGAN, offline Stage-2.5 depth features
- batch size: 128
- steps: 20,000
- learning rate: 2e-5
- tokenizer workers: 1
- save optimizer state: false

Run one suite:

```bash
export SUITE=libero_spatial
export TOTAL_STEPS=20000
export BATCH_SIZE=128
export LR=2e-5
export LOG_FREQ=1
export EVAL_STEPS=0
export SAVE_MODEL_FREQ=20000
export SAVE_MILESTONE_FREQ=0
export RUNTIME_LOG_STEPS=1
export WANDB_ONLINE=False
export EXPERIMENT_ID="lapa_depth_stage3_${SUITE}"
```

For 1 GPU:

```bash
export CUDA_VISIBLE_DEVICES=0
export MESH_DIM='!-1,1,1,1'
bash scripts/train_lapa_depth_suite.sh
```

For 2 GPUs:

```bash
export CUDA_VISIBLE_DEVICES=0,1
export MESH_DIM='!-1,2,1,1'
bash scripts/train_lapa_depth_suite.sh
```

For 4 GPUs, only if the node has 4 free GPUs:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export MESH_DIM='!-1,4,1,1'
bash scripts/train_lapa_depth_suite.sh
```

## 5. Monitor

```bash
watch -n 5 nvidia-smi
tail -f "outputs/${EXPERIMENT_ID}/wandb/${EXPERIMENT_ID}/wandb"/*/logs/debug.log
```

The checkpoint is saved at:

```text
outputs/${EXPERIMENT_ID}/streaming_params
```

## 6. Common Checks

If training prints `train_loop_start` but never prints `batch_ready`, rerun:

```bash
bash scripts/inspect_lapa_depth_alignment.sh
```

If `match_rate` is `0.0`, the JSONL image IDs and depth shard IDs do not match.

If multiprocessing errors appear, keep the stable setting used by
`scripts/train_lapa_depth_suite.sh`:

```text
tokenizer_processes=1
```

## 7. Rollout Smoke

Use the online rollout path after training. Rollout computes depth features from
the current RGB observation through DepthAnythingV2 Sth2Sth + Stage-2.5; it does
not load offline `.pt` feature shards by ID.

```bash
export FINETUNED_CHECKPOINT="params::$LAPA_ROOT/outputs/${EXPERIMENT_ID}/streaming_params"
export LIBERO_PY=/scratch/users/create/smrvmdo/venvs/LIBERO/bin/python
export SUITE=libero_spatial
export TASK_IDS=0
export N_EVAL_PER_TASK=1
export MAX_STEPS=80
export DEPTH_ANYTHING_REPO_DIR="$LAPA_ROOT/third_party/depth_anything_v2"
export DEPTH_ANYTHING_CHECKPOINT="$LAPA_ROOT/checkpoints/depth_anything_v2_sth2sth/depth_anything_v2_sth2sth.pth"
export DEPTH_ANYTHING_ENCODER=vitl

bash scripts/eval_lapa_depth_online_rollout.sh
```
