# LAPA-Depth Supplementary Code

This archive contains anonymized supplementary code for the LAPA-Depth experiments. The code supports Stage-3 downstream policy fine-tuning with depth-aware features and online LIBERO rollout evaluation.

The archive is intended for double-blind review. It does not include author names, institution names, private paths, git history, experiment logs, large datasets, or full model checkpoints.

## Contents

```text
README.md
requirements.txt
LICENSE
latent_pretraining/          Training, model, checkpoint, and policy-server code
eval/                        LIBERO rollout and feature-server code
Depth_branch/                Stage-2.5 depth-feature model code
scripts/
  train_lapa_depth_suite.sh
  eval_lapa_depth_split_online_rollout.sh
  eval_lapa_depth_split_multi_suite.sh
datasets/                    Expected location for prepared data
lapa_checkpoints/            Expected location for checkpoints
```

Only a small, source-only archive should be submitted. Large checkpoints, full datasets, videos, generated outputs, cache folders, and W&B logs should be omitted to satisfy the 50 MB limit.

## Hardware Requirements

Recommended hardware:

- Linux workstation or cluster node;
- NVIDIA GPU with CUDA support;
- 4 GPUs for the split online rollout pipeline;
- at least 30 GB GPU memory per process for 7B-scale LAPA inference;
- sufficient CPU memory for LIBERO simulation and dataset loading.

Training was run with multi-GPU JAX sharding. Rollout evaluation can run with split services on separate GPUs:

```text
GPU 0: LAPA-Depth policy server
GPU 1: Stage-2.5 depth branch and depth estimation
GPU 2: baseline LAPA RGB feature server
GPU 3: optional simulator or spare device
```

This allocation can be changed through environment variables.

## Software Requirements

The code expects Python 3.10 and CUDA-compatible versions of:

- PyTorch;
- JAX and jaxlib;
- Flax;
- Transformers;
- MuJoCo / robosuite;
- LIBERO;
- NumPy, Pillow, requests, tqdm, and standard training utilities.

Install the core Python dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

Depending on the machine, PyTorch, JAX, MuJoCo, and LIBERO may need to be installed separately with versions matching the local CUDA driver and simulator setup.

Set the repository environment:

```bash
export LAPA_ROOT="$(pwd -P)"
export PYTHONPATH="$LAPA_ROOT:${PYTHONPATH:-}"
```

If separate Python environments are used for model inference and LIBERO simulation, set:

```bash
export MODEL_PY=/path/to/model/python
export LIBERO_PY=/path/to/libero/python
```

Otherwise, the scripts use the active `python`.

## Data Preparation

The full LIBERO dataset, pretrained LAPA checkpoints, Stage-2.5 checkpoints, DepthAnythingV2 checkpoint, and precomputed depth features are not included in this archive because they exceed the 50 MB supplementary limit.

Prepare the following layout before training or evaluation:

```text
datasets/
  LIBERO/
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
    stage25_libero_features_model2/
    stage25_libero_features_model4/
    stage25_libero_features_model5/

lapa_checkpoints/
  tokenizer.model
  vqgan/
  base_lapa/
  depth_model/
    model2.65000.pt
    model4.65000.pt
    model5.65000.pt
  stage_3_depth_inject/
    lapa-depth_stage3/

checkpoints/
  depth_anything_v2_sth2sth/
    depth_anything_v2_sth2sth.pth
```

The training JSONL files should contain one robot demonstration frame per line with at least:

```json
{"instruction": "...", "image": "images/<suite>/<task>/demo_0/step_0.jpg", "raw_actions": [...], "action": ["..."], "fields": "[instruction],[vision],action"}
```

The precomputed Stage-2.5 feature shards should contain sample identifiers that can be matched to the JSONL image rows. The expected depth feature dimensionality is 1024.

## Check Data Alignment

Before launching training, verify that JSONL samples and depth-feature shards align:

```bash
export SUITE=libero_spatial
export LAPA_JSONL="$LAPA_ROOT/datasets/lapa_libero_v2/${SUITE}.jsonl"
export DEPTH_DATA_DIR="$LAPA_ROOT/datasets/features_depth_branch/stage25_libero_features_model4/${SUITE}/stage25_model4/z_depth_train_shard0"
export DEPTH_MANIFEST="$DEPTH_DATA_DIR/z_depth_train_shard0_model4_manifest.json"

python -m latent_pretraining.depth_fusion.inspect_lapa_depth_alignment \
  --jsonl "$LAPA_JSONL" \
  --depth_data_dir "$DEPTH_DATA_DIR" \
  --depth_manifest "$DEPTH_MANIFEST"
```

The expected match rate is close to `1.0`.

## Stage-3 Training

Run one suite at a time:

```bash
export SUITE=libero_spatial
export STAGE25_MODEL_NAME=model4
export TOTAL_STEPS=20000
export BATCH_SIZE=128
export MESH_DIM='!-1,4,1,1'
export EXPERIMENT_ID="stage3_${STAGE25_MODEL_NAME}_${SUITE}"

bash scripts/train_lapa_depth_suite.sh
```

Common suite values:

```text
libero_spatial
libero_object
libero_goal
libero_90
```

Common Stage-2.5 model values:

```text
model2
model4
model5
```

The main paper setting uses:

```text
batch size: 128
training steps: 20000
learning rate: 2e-5
trainable modules: LAPA language model and action head
frozen modules: LAPA vision encoder, VQGAN, Stage-2.5 encoder
```

Training outputs are written to:

```text
outputs/<EXPERIMENT_ID>/
```

`streaming_params` is the params-only checkpoint used for rollout. `streaming_train_state` is required for exact resume when optimizer-state saving is enabled.

## Online Rollout Evaluation

Run split online rollout for one suite:

```bash
export SUITE=libero_spatial
export STAGE25_MODEL_NAME=model4

export FINETUNED_CHECKPOINT="params::$LAPA_ROOT/lapa_checkpoints/stage_3_depth_inject/lapa-depth_stage3/128_batch_spatial/streaming_params"
export ORIGINAL_LAPA_CHECKPOINT="params::$LAPA_ROOT/lapa_checkpoints/base_lapa/streaming_params"

export POLICY_CUDA_VISIBLE_DEVICES=0
export STAGE25_CUDA_VISIBLE_DEVICES=1
export RGB_CUDA_VISIBLE_DEVICES=2
export MUJOCO_EGL_DEVICE_ID=1

export TASK_IDS="0 1 2 3 4 5 6 7 8 9"
export N_EVAL_PER_TASK=10
export MAX_STEPS=500
export PROGRESS_FREQ=25
export OUTPUT_DIR="$LAPA_ROOT/outputs/eval_${SUITE}_${STAGE25_MODEL_NAME}"

bash scripts/eval_lapa_depth_split_online_rollout.sh
```

Run multiple suites:

```bash
export SUITES="libero_spatial libero_object libero_goal"
export TASK_IDS="0 1 2 3 4 5 6 7 8 9"
export N_EVAL_PER_TASK=10
export MAX_STEPS=500
export OUTPUT_PREFIX="eval_lapa_depth"

bash scripts/eval_lapa_depth_split_multi_suite.sh
```

Task IDs must be separated by spaces.

## Reproducing Main Table Results

The following commands reproduce the per-suite rollout measurements after checkpoints and data are prepared:

```bash
# LAPA-Depth Model 2
export STAGE25_MODEL_NAME=model2
export SUITES="libero_spatial libero_object libero_goal"
bash scripts/eval_lapa_depth_split_multi_suite.sh

# LAPA-Depth Model 4
export STAGE25_MODEL_NAME=model4
export SUITES="libero_spatial libero_object libero_goal"
bash scripts/eval_lapa_depth_split_multi_suite.sh

# LAPA-Depth Model 5
export STAGE25_MODEL_NAME=model5
export SUITES="libero_spatial libero_object libero_goal"
bash scripts/eval_lapa_depth_split_multi_suite.sh
```

Expected success rates:

| Method | Spatial | Object | Goal | Average |
| --- | ---: | ---: | ---: | ---: |
| LAPA baseline | 52% | 64% | 58% | 58.0% |
| LAPA-Depth Model 2 | 61% | 69% | 61% | 63.7% |
| LAPA-Depth Model 4 | 54% | 67% | 67% | 62.7% |
| LAPA-Depth Model 5 | 38% | 57% | 53% | 49.3% |

Minor variation is expected due to simulator randomness, GPU kernels, and environment setup.

## Expected Outputs

Training produces:

```text
outputs/<experiment_id>/streaming_params
outputs/<experiment_id>/metadata.pkl
outputs/<experiment_id>/dataset.pkl
```

Evaluation produces:

```text
outputs/<evaluation_id>/results.json
outputs/<evaluation_id>/summary.json
outputs/server_logs/
```

Videos and full rollout traces should not be included in the AAAI supplementary archive unless they are very small and explicitly needed.

## Approximate Runtime

Runtime depends strongly on GPU type and simulator speed.

Approximate ranges:

- Stage-3 training, one suite, 20k steps, 4 high-memory GPUs: 12-48 hours;
- split online rollout, 10 tasks x 10 episodes, 4 GPUs: several hours per suite;
- data-alignment inspection: minutes;
- checkpoint loading and first rollout episode: slower due to JAX/XLA, PyTorch, and MuJoCo initialization.

## Omitted Files

The following are intentionally omitted from the supplementary archive:

- full LAPA checkpoints;
- Stage-2.5 model checkpoints;
- DepthAnythingV2 checkpoint;
- LIBERO full datasets and image folders;
- precomputed depth-feature shards;
- generated rollout videos;
- `outputs/`, `checkpoints/`, `wandb/`, cache folders, virtual environments, and `.git/`.

These files are large and should be prepared separately according to the dataset and checkpoint instructions used by the paper.

## Creating the Submission Archive

From the parent directory:

```bash
zip -r code_data_supplement.zip LAPA \
  -x "*/.git/*" \
     "*/__pycache__/*" \
     "*.pyc" \
     "*/wandb/*" \
     "*/checkpoints/*" \
     "*/lapa_checkpoints/*" \
     "*/outputs/*" \
     "*/rollouts/*" \
     "*/datasets/LIBERO/*" \
     "*/datasets/lapa_libero_v2/images/*" \
     "*/datasets/features_depth_branch/*"
```

Verify the archive size:

```bash
du -h code_data_supplement.zip
```

The final `.zip` should be below 50 MB.

## Anonymization Checklist

Before submission, verify that the archive does not contain:

- author names or institution names;
- personal usernames;
- absolute local paths;
- repository URLs;
- W&B project links or run logs;
- git history;
- full model checkpoints;
- large datasets;
- generated videos.

Suggested checks:

```bash
find LAPA -name ".git" -o -name "__pycache__" -o -name "wandb"
grep -R "home/users\|scratch/users\|C:\\\\Users\|D:\\\\Project" LAPA || true
```

## License

This code builds on public research software for LAPA-style policy learning, LIBERO simulation, and depth estimation. Follow the licenses and citation requirements of all upstream software and datasets used to run the experiments.
