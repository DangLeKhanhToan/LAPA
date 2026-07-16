# LAPA-Depth: Stage-3 Offline Fine-Tuning and Online LIBERO Rollout

This repository is a modified LAPA codebase for the depth-injection Stage-3
pipeline. It supports:

- offline Stage-3 fine-tuning with precomputed Stage-2.5 depth features;
- online rollout with RGB observations, DepthAnythingV2, Stage-2.5 model4, and a
  fine-tuned LAPA-Depth policy;
- split-server rollout so the two LAPA 7B models do not OOM on one GPU.

The original LAPA README has been replaced by this team runbook. Use this file
as the first checklist when setting up a new machine or reproducing training and
evaluation.

## Pipeline Summary

### Offline Training

Training uses raw RGB images, instructions, action-bin labels, and offline
1024-D depth features:

```text
RGB image + instruction + offline z_depth_feature(1024)
    -> fine-tune LAPA language model + action head
    -> robot action token bins
```

Frozen modules:

```text
LAPA vision encoder, VQGAN, Stage-2.5 depth feature source
```

Trainable modules:

```text
LAPA language model, depth_action_proj, action/action-token head
```

### Online Rollout

At rollout time we do not load offline depth features. We compute depth features
online:

```text
LIBERO simulator RGB
  -> fine-tuned LAPA-Depth policy server
      -> Stage2.5 feature server
          -> baseline LAPA RGB feature server
          -> DepthAnythingV2
          -> model4
      -> 1024-D depth feature
  -> 7-D robot action
```

Recommended 4-GPU allocation on 30GB RTX 5000 Ada:

```text
GPU 1: fine-tuned LAPA-Depth policy
GPU 2: DepthAnythingV2 + Stage-2.5 model4
GPU 3: baseline LAPA RGB feature server
GPU 0: avoid if unstable, or use for simulator only
```

## Repository Layout

Important modified files:

```text
latent_pretraining/train.py                 Stage-3 training entrypoint
latent_pretraining/deploy.py                fine-tuned policy HTTP server
eval/lapa_rgb_feature_server.py             baseline LAPA -> 4096-D RGB feature server
eval/stage25_feature_server.py              DepthAnythingV2 + model4 -> 1024-D depth feature server
eval/eval_libero_rollout_depth.py           LIBERO rollout client with progress logs
scripts/train_lapa_depth_suite.sh           offline Stage-3 training wrapper
scripts/eval_lapa_depth_split_online_rollout.sh
scripts/eval_lapa_depth_split_multi_suite.sh
scripts/inspect_lapa_depth_alignment.sh
scripts/smoke_stage25_online_feature.sh
```

External bundle expected inside this repo:

```text
Depth_branch/
  laq/rollout_stage25_model4.py
  latent_pretraining/inference_update_jsonl_train.py
```

## Environment Setup

Example setup on the cluster:

```bash
cd ~/scratch/projects
git clone <repo-url> lapa-depth-modified-levi
cd lapa-depth-modified-levi

python3 -m venv ~/scratch/venvs/lapa-depth
source ~/scratch/venvs/lapa-depth/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

On Linh workstation we usually use:

```bash
export MODEL_PY=/mnt/hdd/linh/long/conda_envs/lapa-depth/bin/python
export LIBERO_PY=/mnt/hdd/linh/long/conda_envs/lapa-depth/bin/python
```

On A2AP we usually use:

```bash
source ~/scratch/venvs/lapa-depth/bin/activate
```

## LIBERO Setup

Clone or place LIBERO under:

```text
datasets/LIBERO
```

Then expose it during rollout:

```bash
export LAPA_ROOT="$(pwd -P)"
export LIBERO_REPO="$LAPA_ROOT/datasets/LIBERO"
export PYTHONPATH="$LIBERO_REPO:$LAPA_ROOT:${PYTHONPATH:-}"
```

The rollout wrapper already sets:

```bash
LIBERO_REPO="${LIBERO_REPO:-$PROJECT_DIR/datasets/LIBERO}"
```

LIBERO data/checkpoints should provide BDDL files and init states via LIBERO's
standard `get_libero_path("bddl_files")` mechanism.

## Required Data and Checkpoints

Expected local layout:

```text
lapa_checkpoints/
  tokenizer.model
  vqgan
  lapa_7b_sth/params                         # base LAPA for training
  pretraining_LAPA_Sth2Sth                   # baseline LAPA for online RGB feature
  depth_model/model4.65000.pt                # Stage-2.5 model4
  stage_3_depth_inject/lapa-depth_stage3/
    128_batch_spatial
    128_batch_object
    128_batch_goal
    streaming_params                         # current fallback for libero_90 if no 128_batch_90

datasets/
  lapa_libero_v2/
    images/
    libero_spatial.jsonl
    libero_object.jsonl
    libero_goal.jsonl
    libero_90.jsonl
    action_bins_libero_spatial.csv
    action_bins_libero_object.csv
    action_bins_libero_goal.csv
    action_bins_libero_90.csv
  features_depth_branch/
    stage25_libero_features_model4/
      libero_spatial/stage25_model4/z_depth_train_shard0/*.pt
      libero_object/stage25_model4/z_depth_train_shard0/*.pt
      libero_goal/stage25_model4/z_depth_train_shard0/*.pt
      libero_90/stage25_model4/z_depth_train_shard0/*.pt

Depth_branch/
third_party/depth_anything_v2/
checkpoints/depth_anything_v2_sth2sth/depth_anything_v2_sth2sth.pth
```

DepthAnythingV2 Sth2Sth checkpoint uses the `vitl` encoder.

## Download / Install DepthAnythingV2

If the repo and checkpoint are not present:

```bash
cd /path/to/LAPA-depth
export DEPTH_ANYTHING_CKPT_LOCAL=/path/to/depth_anything_v2_sth2sth.pth
bash scripts/download_depthanythingv2_sth2sth.sh
```

If downloading from a URL:

```bash
export DEPTH_ANYTHING_CKPT_URL=https://.../depth_anything_v2_sth2sth.pth
bash scripts/download_depthanythingv2_sth2sth.sh
```

## Inspect Depth Feature Alignment

Before training, verify JSONL rows match offline depth features:

```bash
cd ~/scratch/projects/lapa-depth-modified-levi

export LAPA_JSONL=datasets/lapa_libero_v2/libero_spatial.jsonl
export DEPTH_DATA_DIR=datasets/features_depth_branch/stage25_libero_features_model4/libero_spatial/stage25_model4/z_depth_train_shard0
export DEPTH_MANIFEST="$DEPTH_DATA_DIR/z_depth_train_shard0_model4_manifest.json"

bash scripts/inspect_lapa_depth_alignment.sh
```

Expected:

```text
match_rate: 1.0
depth_shape: [1024]
```

Some suites have depth ids with an extra `_depth` token. The current alignment
code handles this normalization.

## Offline Stage-3 Training

Train one suite:

```bash
cd ~/scratch/projects/lapa-depth-modified-levi
source ~/scratch/venvs/lapa-depth/bin/activate

export LAPA_ROOT="$(pwd -P)"
export SUITE=libero_spatial
export TOTAL_STEPS=20000
export BATCH_SIZE=128
export MESH_DIM='!-1,4,1,1'
export LR=2e-5

export SAVE_MODEL_FREQ=200
export SAVE_MILESTONE_FREQ=1000
export AUTORESUME=True
export SAVE_OPTIMIZER_STATE=True

export WANDB_ONLINE=False
export EXPERIMENT_ID="128_batch_model_2_${SUITE}"

SUITE="$SUITE" bash scripts/train_lapa_depth_suite.sh
```

The wrapper automatically resolves:

```text
datasets/lapa_libero_v2/${SUITE}.jsonl
datasets/lapa_libero_v2/action_bins_${SUITE}.csv
datasets/features_depth_branch/stage25_libero_features_model4/${SUITE}/stage25_model4/...
```

## PBS Training Script

Use this as a resume-safe PBS template:

```bash
#!/bin/bash
#PBS -N training_libero_depth
#PBS -q normal
#PBS -P 11714283
#PBS -l select=1:ncpus=112:ngpus=4
#PBS -l walltime=24:00:00
#PBS -j oe

cd ~/scratch/projects/lapa-depth-modified-levi
source ~/scratch/venvs/lapa-depth/bin/activate

export LAPA_ROOT="$(pwd -P)"
export SUITE="${SUITE:-libero_spatial}"

export TOTAL_STEPS=20000
export BATCH_SIZE=128
export MESH_DIM='!-1,4,1,1'

export LOG_FREQ=1
export EVAL_STEPS=0
export SAVE_MODEL_FREQ=200
export SAVE_MILESTONE_FREQ=1000
export AUTORESUME=True
export SAVE_OPTIMIZER_STATE=True
export RUNTIME_LOG_STEPS=3
export WANDB_ONLINE=False

export EXPERIMENT_ID="128_batch_model_2_${SUITE}"

echo "Job ID: $PBS_JOBID"
echo "Suite: $SUITE"
echo "Experiment: $EXPERIMENT_ID"
echo "Host: $(hostname)"
nvidia-smi

SUITE="$SUITE" bash scripts/train_lapa_depth_suite.sh
```

Submit:

```bash
qsub -v SUITE=libero_spatial train_lapa_depth.pbs
qsub -v SUITE=libero_object  train_lapa_depth.pbs
qsub -v SUITE=libero_goal    train_lapa_depth.pbs
qsub -v SUITE=libero_90      train_lapa_depth.pbs
```

Resume after walltime by submitting the same suite and same `EXPERIMENT_ID`.
Exact resume requires both:

```bash
AUTORESUME=True
SAVE_OPTIMIZER_STATE=True
```

If only `streaming_params` exists, training can warm-start from weights but
optimizer state, dataset position, and LR schedule are reset.

## Check Saved Training State

```bash
cd ~/scratch/projects/lapa-depth-modified-levi

for EXP in 128_batch_model_2_libero_spatial 128_batch_model_2_libero_object 128_batch_model_2_libero_goal 128_batch_model_2_libero_90; do
  echo "==== $EXP ===="
  find outputs/$EXP -maxdepth 2 -type d \( -name "streaming_train_state" -o -name "streaming_params" \) 2>/dev/null
  find outputs/$EXP -maxdepth 2 -type f \( -name "metadata.pkl" -o -name "dataset.pkl" \) 2>/dev/null
done
```

## Online Rollout: Split Services

The current reliable rollout path starts three servers:

```text
1. eval.lapa_rgb_feature_server       baseline LAPA -> 4096-D RGB feature
2. eval.stage25_feature_server        DepthAnythingV2 + model4 -> 1024-D depth feature
3. latent_pretraining.deploy          fine-tuned LAPA-Depth -> 7-D robot action
```

Run one suite:

```bash
cd /home/linhkastner/lapa/LAPA-depth

export MODEL_PY=/mnt/hdd/linh/long/conda_envs/lapa-depth/bin/python
export LIBERO_PY=/mnt/hdd/linh/long/conda_envs/lapa-depth/bin/python
export LAPA_ROOT="$(pwd -P)"

export ORIGINAL_LAPA_CHECKPOINT="params::$LAPA_ROOT/lapa_checkpoints/pretraining_LAPA_Sth2Sth"

export POLICY_CUDA_VISIBLE_DEVICES=1
export STAGE25_CUDA_VISIBLE_DEVICES=2
export RGB_CUDA_VISIBLE_DEVICES=3
export MUJOCO_EGL_DEVICE_ID=2

export DEPTH_ANYTHING_ENCODER=vitl
export DEPTH_ANYTHING_INPUT_SIZE=518
export DEPTH_ANYTHING_DEVICE=cuda

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.80
export TF_FORCE_GPU_ALLOW_GROWTH=true
export JAX_PLATFORMS=cuda,cpu

export SUITE=libero_spatial
export FINETUNED_CHECKPOINT="params::$LAPA_ROOT/lapa_checkpoints/stage_3_depth_inject/lapa-depth_stage3/128_batch_spatial"
export TASK_IDS="0 1 2 3 4 5 6 7 8 9"
export N_EVAL_PER_TASK=10
export MAX_STEPS=500
export PROGRESS_FREQ=25
export OUTPUT_DIR="$LAPA_ROOT/outputs/eval_split_${SUITE}_10tasks_10eps"

bash scripts/eval_lapa_depth_split_online_rollout.sh
```

Task ids must be space-separated:

```bash
export TASK_IDS="0 1 2 3 4 5 6 7 8 9"
```

Do not use comma-separated ids:

```bash
export TASK_IDS=0,1,2,3,4,5,6,7,8,9  # wrong
```

## Online Rollout: Multiple Suites

Sequentially evaluate four suites:

```bash
cd /home/linhkastner/lapa/LAPA-depth

export MODEL_PY=/mnt/hdd/linh/long/conda_envs/lapa-depth/bin/python
export LIBERO_PY=/mnt/hdd/linh/long/conda_envs/lapa-depth/bin/python
export LAPA_ROOT="$(pwd -P)"
export ORIGINAL_LAPA_CHECKPOINT="params::$LAPA_ROOT/lapa_checkpoints/pretraining_LAPA_Sth2Sth"

export POLICY_CUDA_VISIBLE_DEVICES=1
export STAGE25_CUDA_VISIBLE_DEVICES=2
export RGB_CUDA_VISIBLE_DEVICES=3
export MUJOCO_EGL_DEVICE_ID=2

export SUITES="libero_spatial libero_object libero_goal libero_90"
export TASK_IDS="0 1 2 3 4 5 6 7 8 9"
export N_EVAL_PER_TASK=10
export MAX_STEPS=500
export PROGRESS_FREQ=25
export OUTPUT_PREFIX="eval_split_10tasks_10eps"

bash scripts/eval_lapa_depth_split_multi_suite.sh
```

Default checkpoint mapping in the multi-suite wrapper:

```text
libero_spatial -> lapa-depth_stage3/128_batch_spatial
libero_object  -> lapa-depth_stage3/128_batch_object
libero_goal    -> lapa-depth_stage3/128_batch_goal
libero_90      -> lapa-depth_stage3/128_batch_90 if present, else lapa-depth_stage3/streaming_params
```

## Rollout Logs and Results

Server logs:

```text
outputs/server_logs/rgb_feature_gpu*.log
outputs/server_logs/stage25_split_gpu*.log
outputs/server_logs/policy_gpu*.log
```

Rollout output:

```text
outputs/<OUTPUT_DIR>/results.json
outputs/<OUTPUT_DIR>/<suite>/<task_name>/ep*_success.mp4
outputs/<OUTPUT_DIR>/<suite>/<task_name>/ep*_fail.mp4
```

The evaluator prints progress every `PROGRESS_FREQ` simulator steps:

```text
[libero_spatial] task 0 ep 0 step 25/500 | elapsed=... | eta=...
```

## Smoke Tests

Check a Stage2.5 online feature:

```bash
export RGB_IMAGE=/path/to/step_0.jpg
export INSTRUCTION="pick up the black bowl between the plate and the ramekin and place it on the plate"
bash scripts/smoke_stage25_online_feature.sh \
  --rgb_image "$RGB_IMAGE" \
  --instruction "$INSTRUCTION" \
  --port 32823
```

Inspect checkpoint parameter groups:

```bash
export FINETUNED_CHECKPOINT="params::$LAPA_ROOT/lapa_checkpoints/stage_3_depth_inject/lapa-depth_stage3/128_batch_spatial"
bash scripts/inspect_lapa_depth_policy_split.sh
```

Expected depth-injection additions are tiny compared with the LAPA trunk:

```text
depth_action_proj/kernel: (1024, 4096)
action_head/kernel:       (4096, 256)
```

## Troubleshooting

### `argument --task_ids: invalid int value: '0,1,2'`

Use spaces:

```bash
export TASK_IDS="0 1 2"
```

### Policy server returns `"error"`

Inspect server logs:

```bash
tail -200 outputs/server_logs/policy_gpu*.log
tail -200 outputs/server_logs/stage25_split_gpu*.log
tail -200 outputs/server_logs/rgb_feature_gpu*.log
```

Most common cause: Stage2.5 returned HTTP 500, often from DepthAnything CUDA
timeout.

### CUDA timeout on GPU0

GPU0 has repeatedly shown:

```text
CUDA error: the launch timed out and was terminated
```

Avoid GPU0 for model servers when possible:

```bash
export POLICY_CUDA_VISIBLE_DEVICES=1
export STAGE25_CUDA_VISIBLE_DEVICES=2
export RGB_CUDA_VISIBLE_DEVICES=3
export MUJOCO_EGL_DEVICE_ID=2
```

Kill stale servers:

```bash
pkill -u "$USER" -f "latent_pretraining.deploy" || true
pkill -u "$USER" -f "eval.stage25_feature_server" || true
pkill -u "$USER" -f "eval.lapa_rgb_feature_server" || true
pkill -u "$USER" -f "eval_libero_rollout_depth" || true
```

### Stage2.5 OOM

Do not run baseline LAPA, DepthAnythingV2, and model4 in one process/GPU. Use the
split rollout scripts. The old bundled Stage2.5 path can OOM on 30GB GPUs.

### Rollout seems stuck

The first episode may be slow because JAX/XLA, PyTorch CUDA, DepthAnythingV2,
and MuJoCo/EGL warm up. The evaluator now prints heartbeat logs every
`PROGRESS_FREQ` steps.

### `libero_90` checkpoint looks smaller

Check actual checkpoint folders:

```bash
du -sh lapa_checkpoints/stage_3_depth_inject/lapa-depth_stage3/*
find lapa_checkpoints/stage_3_depth_inject/lapa-depth_stage3 -maxdepth 2 -type d -print
```

If `128_batch_90` is missing, the multi-suite script uses root
`streaming_params` as fallback. Verify this is the intended `libero_90`
checkpoint before reporting results.

## Notes on Speed

Online LAPA-Depth rollout is slower than baseline LAPA because each action
requires:

```text
fine-tuned LAPA-Depth policy
+ baseline LAPA RGB feature inference
+ DepthAnythingV2 vitl
+ model4
+ HTTP calls and image reads
+ simulator step
```

The baseline rollout usually only runs one LAPA policy inference per simulator
step. The split design is slower but prevents OOM and is faithful to the online
depth pipeline.
