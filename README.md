# LAPA-Depth

LAPA-Depth extends LAPA with depth-aware features for robot policy fine-tuning and evaluation in LIBERO. The repository provides scripts for:

- Stage-3 offline fine-tuning with precomputed depth features;
- online depth-feature extraction from RGB observations;
- LIBERO rollout evaluation;
- split-service inference to reduce GPU memory usage;
- checkpoint inspection and data-alignment validation.

## Repository

```bash
git clone https://github.com/DangLeKhanhToan/LAPA.git
cd LAPA
```

## Method Overview

### Offline fine-tuning

During Stage-3 training, the policy receives an RGB observation, a language instruction, an action label, and a precomputed 1024-dimensional depth feature.

```text
RGB observation + instruction + offline depth feature
    -> LAPA-Depth policy
    -> discretized robot action
```

The depth features are expected to be generated beforehand by the Stage-2.5 depth branch.

### Online rollout

During evaluation, the depth feature is generated from the current RGB observation instead of being loaded from disk.

```text
LIBERO RGB observation
    -> RGB feature extractor
    -> depth estimator
    -> Stage-2.5 depth feature model
    -> LAPA-Depth policy
    -> continuous robot action
```

The inference components can run as separate services on different GPUs. This is recommended when the complete pipeline does not fit on one GPU.

## Repository Structure

The main files used by the depth-aware pipeline are:

```text
latent_pretraining/
  train.py                         Stage-3 training entry point
  deploy.py                        LAPA-Depth policy server

eval/
  lapa_rgb_feature_server.py       RGB feature extraction service
  stage25_feature_server.py        Online depth-feature service
  eval_libero_rollout_depth.py     LIBERO rollout evaluator

scripts/
  train_lapa_depth_suite.sh
  eval_lapa_depth_split_online_rollout.sh
  eval_lapa_depth_split_multi_suite.sh
  inspect_lapa_depth_alignment.sh
  inspect_lapa_depth_policy_split.sh
  smoke_stage25_online_feature.sh
```

Additional depth modules are expected under:

```text
Depth_branch/
third_party/depth_anything_v2/
```

## Requirements

A Linux machine with an NVIDIA GPU is recommended.

Typical requirements include:

- Python 3.10;
- CUDA-compatible PyTorch;
- JAX with GPU support;
- LIBERO and MuJoCo;
- DepthAnythingV2;
- model checkpoints for LAPA, Stage-2.5, and DepthAnythingV2.

Create an isolated environment and install the project dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

Alternatively, use a Conda environment:

```bash
conda create -n lapa-depth python=3.10 -y
conda activate lapa-depth
pip install -r requirements.txt
```

GPU packages may need to be installed separately to match the CUDA version available on the machine.

## Configure the Project

From the repository root, define:

```bash
export LAPA_ROOT="$(pwd -P)"
export PYTHONPATH="$LAPA_ROOT:${PYTHONPATH:-}"
```

The provided scripts read most configuration from environment variables, allowing paths and hardware settings to be changed without editing the source code.

## Install LIBERO

Place or clone LIBERO inside the repository:

```text
datasets/LIBERO/
```

Then configure the Python path:

```bash
export LIBERO_REPO="$LAPA_ROOT/datasets/LIBERO"
export PYTHONPATH="$LIBERO_REPO:$LAPA_ROOT:${PYTHONPATH:-}"
```

Follow the official LIBERO installation instructions to install its dependencies and download the required BDDL files, initial states, and demonstration data.

Verify that LIBERO can locate its resources through its standard path configuration, including:

```python
get_libero_path("bddl_files")
get_libero_path("init_states")
```

## Data and Checkpoint Layout

The exact directory names are configurable, but the following structure is recommended:

```text
LAPA/
├── lapa_checkpoints/
│   ├── tokenizer.model
│   ├── vqgan/
│   ├── base_lapa/
│   ├── rgb_feature_lapa/
│   ├── depth_model/
│   │   └── model4.pt
│   └── stage3/
│       ├── libero_spatial/
│       ├── libero_object/
│       ├── libero_goal/
│       └── libero_90/
│
├── datasets/
│   ├── LIBERO/
│   ├── lapa_libero/
│   │   ├── images/
│   │   ├── libero_spatial.jsonl
│   │   ├── libero_object.jsonl
│   │   ├── libero_goal.jsonl
│   │   ├── libero_90.jsonl
│   │   └── action_bins_*.csv
│   └── features_depth_branch/
│       └── stage25_libero_features_model4/
│           ├── libero_spatial/
│           ├── libero_object/
│           ├── libero_goal/
│           └── libero_90/
│
├── Depth_branch/
├── third_party/
│   └── depth_anything_v2/
└── checkpoints/
    └── depth_anything_v2.pth
```

Required assets include:

1. a base LAPA checkpoint for Stage-3 fine-tuning;
2. a baseline LAPA checkpoint for online RGB feature extraction;
3. a trained Stage-2.5 depth feature model;
4. a DepthAnythingV2 checkpoint;
5. LIBERO training JSONL files and action-bin files;
6. precomputed depth features for offline training;
7. fine-tuned LAPA-Depth checkpoints for evaluation.

## Configure Dataset Paths

For a LIBERO suite, define the training files explicitly:

```bash
export SUITE=libero_spatial
export LAPA_JSONL="$LAPA_ROOT/datasets/lapa_libero/${SUITE}.jsonl"
export ACTION_BINS="$LAPA_ROOT/datasets/lapa_libero/action_bins_${SUITE}.csv"
export DEPTH_DATA_DIR="$LAPA_ROOT/datasets/features_depth_branch/stage25_libero_features_model4/${SUITE}/stage25_model4/z_depth_train_shard0"
```

Supported suite names normally include:

```text
libero_spatial
libero_object
libero_goal
libero_90
```

## Validate Depth-Feature Alignment

Before training, verify that the dataset rows and precomputed depth features use matching sample identifiers.

```bash
export SUITE=libero_spatial
export LAPA_JSONL="$LAPA_ROOT/datasets/lapa_libero/${SUITE}.jsonl"
export DEPTH_DATA_DIR="$LAPA_ROOT/datasets/features_depth_branch/stage25_libero_features_model4/${SUITE}/stage25_model4/z_depth_train_shard0"
export DEPTH_MANIFEST="$DEPTH_DATA_DIR/z_depth_train_shard0_model4_manifest.json"

bash scripts/inspect_lapa_depth_alignment.sh
```

A correctly prepared dataset should report values similar to:

```text
match_rate: 1.0
depth_shape: [1024]
```

Do not start a long training run until the match rate is correct.

## Stage-3 Fine-Tuning

The training wrapper accepts configuration through environment variables.

```bash
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
export EXPERIMENT_ID="lapa_depth_${SUITE}"

bash scripts/train_lapa_depth_suite.sh
```

The wrapper should resolve or receive paths for:

```text
training JSONL
action-bin CSV
offline depth features
base LAPA checkpoint
tokenizer and VQGAN checkpoints
output directory
```

Review `scripts/train_lapa_depth_suite.sh` before running it and override any default paths that do not match the local machine.

### Multi-GPU configuration

`MESH_DIM` must match the number of devices allocated to the job. For example:

```bash
export MESH_DIM='!-1,4,1,1'   # four devices
```

A larger global batch size does not necessarily improve throughput. Measure step time and GPU utilization when changing batch size, data-loader workers, or device count.

## Resume Training

To resume an interrupted run, use the same:

```text
SUITE
EXPERIMENT_ID
output directory
model configuration
```

Enable:

```bash
export AUTORESUME=True
export SAVE_OPTIMIZER_STATE=True
```

A full resume requires both model parameters and training state. If only model parameters are available, the run can usually warm-start, but optimizer state, scheduler state, and dataset position may be reset.

Inspect saved state with:

```bash
find "outputs/$EXPERIMENT_ID" -maxdepth 3 \
  \( -name 'streaming_params' \
  -o -name 'streaming_train_state' \
  -o -name 'metadata.pkl' \
  -o -name 'dataset.pkl' \) \
  -print
```

## Example Scheduler Script

The project can be launched through Slurm, PBS, or another scheduler. The following generic PBS example must be adapted to the target cluster:

```bash
#!/usr/bin/env bash
#PBS -N lapa_depth
#PBS -q <queue-name>
#PBS -P <project-id>
#PBS -l select=1:ncpus=<cpu-count>:ngpus=<gpu-count>
#PBS -l walltime=<hours>:00:00
#PBS -j oe

set -Eeuo pipefail

cd /path/to/LAPA
source .venv/bin/activate

export LAPA_ROOT="$(pwd -P)"
export SUITE="${SUITE:-libero_spatial}"

export TOTAL_STEPS=20000
export BATCH_SIZE=128
export MESH_DIM='!-1,4,1,1'
export LR=2e-5

export SAVE_MODEL_FREQ=200
export SAVE_MILESTONE_FREQ=1000
export AUTORESUME=True
export SAVE_OPTIMIZER_STATE=True
export WANDB_ONLINE=False
export EXPERIMENT_ID="lapa_depth_${SUITE}"

nvidia-smi
bash scripts/train_lapa_depth_suite.sh
```

Submit a suite by passing it as an environment variable using the syntax supported by the cluster.

## Online LIBERO Rollout

### Why split the services?

The complete online pipeline may contain:

- the fine-tuned LAPA-Depth policy;
- a baseline LAPA model for RGB features;
- DepthAnythingV2;
- the Stage-2.5 depth model;
- LIBERO and MuJoCo.

Running everything on one device can exceed GPU memory. The split rollout launches independent services and assigns them to separate GPUs.

### Recommended GPU assignment

For a machine with at least three usable GPUs:

```bash
export POLICY_CUDA_VISIBLE_DEVICES=0
export STAGE25_CUDA_VISIBLE_DEVICES=1
export RGB_CUDA_VISIBLE_DEVICES=2
```

The simulator can use one of these GPUs or another available device:

```bash
export MUJOCO_EGL_DEVICE_ID=0
```

Adjust the mapping according to available memory and utilization.

### Configure model checkpoints

```bash
export ORIGINAL_LAPA_CHECKPOINT="params::$LAPA_ROOT/lapa_checkpoints/rgb_feature_lapa"
export FINETUNED_CHECKPOINT="params::$LAPA_ROOT/lapa_checkpoints/stage3/libero_spatial"
```

Configure DepthAnythingV2:

```bash
export DEPTH_ANYTHING_ENCODER=vitl
export DEPTH_ANYTHING_INPUT_SIZE=518
export DEPTH_ANYTHING_DEVICE=cuda
```

Optional memory settings:

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.80
export TF_FORCE_GPU_ALLOW_GROWTH=true
export JAX_PLATFORMS=cuda,cpu
```

### Evaluate one suite

```bash
export MODEL_PY="$(which python)"
export LIBERO_PY="$(which python)"

export SUITE=libero_spatial
export TASK_IDS="0 1 2 3 4 5 6 7 8 9"
export N_EVAL_PER_TASK=10
export MAX_STEPS=500
export PROGRESS_FREQ=25
export OUTPUT_DIR="$LAPA_ROOT/outputs/eval_${SUITE}"

bash scripts/eval_lapa_depth_split_online_rollout.sh
```

Task IDs must be separated by spaces:

```bash
export TASK_IDS="0 1 2"
```

Do not use a comma-separated string unless the evaluation script explicitly supports it.

### Evaluate multiple suites

```bash
export SUITES="libero_spatial libero_object libero_goal libero_90"
export TASK_IDS="0 1 2 3 4 5 6 7 8 9"
export N_EVAL_PER_TASK=10
export MAX_STEPS=500
export PROGRESS_FREQ=25
export OUTPUT_PREFIX="eval_lapa_depth"

bash scripts/eval_lapa_depth_split_multi_suite.sh
```

Check the checkpoint mapping inside the multi-suite wrapper before evaluation. Each suite should point to the intended fine-tuned checkpoint.

## Outputs

Training outputs are normally stored under:

```text
outputs/<experiment-id>/
```

Evaluation outputs may include:

```text
outputs/<evaluation-name>/results.json
outputs/<evaluation-name>/<suite>/<task>/ep*_success.mp4
outputs/<evaluation-name>/<suite>/<task>/ep*_fail.mp4
```

Service logs are normally written under:

```text
outputs/server_logs/
```

The exact paths depend on the wrapper configuration.

## Smoke Tests

### Test online depth-feature extraction

```bash
export RGB_IMAGE=/path/to/example.jpg
export INSTRUCTION="pick up the object and place it in the target location"

bash scripts/smoke_stage25_online_feature.sh \
  --rgb_image "$RGB_IMAGE" \
  --instruction "$INSTRUCTION" \
  --port 32823
```

### Inspect a fine-tuned checkpoint

```bash
export FINETUNED_CHECKPOINT="params::$LAPA_ROOT/lapa_checkpoints/stage3/libero_spatial"
bash scripts/inspect_lapa_depth_policy_split.sh
```

Expected depth-aware parameter groups may include:

```text
depth_action_proj
action_head
action-token head
```

## Troubleshooting

### Invalid task ID

Example error:

```text
argument --task_ids: invalid int value
```

Use a space-separated list:

```bash
export TASK_IDS="0 1 2"
```

### Policy server returns an error

Inspect the logs from all services:

```bash
tail -n 200 outputs/server_logs/policy*.log
tail -n 200 outputs/server_logs/stage25*.log
tail -n 200 outputs/server_logs/rgb_feature*.log
```

The policy request may fail because an upstream feature service failed, timed out, or ran out of GPU memory.

### CUDA out-of-memory error

Use separate GPUs for:

```text
LAPA-Depth policy
baseline LAPA RGB feature model
DepthAnythingV2 and Stage-2.5 model
```

Also consider:

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.70
```

Reduce the fraction further when multiple frameworks share a device.

### CUDA timeout or unstable device

Move the affected service to another GPU and verify that no stale process is still using the device.

Stop old services with:

```bash
pkill -u "$USER" -f 'latent_pretraining.deploy' || true
pkill -u "$USER" -f 'eval.stage25_feature_server' || true
pkill -u "$USER" -f 'eval.lapa_rgb_feature_server' || true
pkill -u "$USER" -f 'eval_libero_rollout_depth' || true
```

### Rollout appears to be stuck

The first episode can be slow because JAX/XLA, PyTorch, DepthAnythingV2, and MuJoCo may initialize or compile kernels on first use.

Check:

```bash
nvidia-smi
```

Then monitor the service logs and confirm that heartbeat messages continue to appear.

### Dataset and depth features do not match

Run:

```bash
bash scripts/inspect_lapa_depth_alignment.sh
```

Common causes include:

- different ordering between the JSONL and feature manifest;
- inconsistent sample IDs;
- an incorrect suite path;
- missing feature files;
- features generated from a different dataset version.

### Checkpoint appears incomplete

Inspect directory sizes and contents:

```bash
du -sh lapa_checkpoints/stage3/*
find lapa_checkpoints/stage3 -maxdepth 3 -type d -print
```

Confirm that the selected path contains the expected parameter tree and belongs to the correct LIBERO suite.

## Performance Notes

Online LAPA-Depth rollout is expected to be slower than a baseline LAPA rollout because each simulator step may require:

```text
LAPA-Depth policy inference
+ baseline LAPA RGB feature extraction
+ DepthAnythingV2 inference
+ Stage-2.5 depth feature inference
+ inter-process or HTTP communication
+ simulator execution
```

The split-service design prioritizes reproducibility and memory safety over minimum latency.

For performance analysis, record:

- average simulator-step time;
- policy latency;
- RGB feature latency;
- depth feature latency;
- GPU utilization and memory;
- service errors and retries.

## Reproducibility Checklist

Before reporting results, record:

- commit hash;
- Python and CUDA versions;
- PyTorch and JAX versions;
- GPU model and GPU count;
- LIBERO version;
- suite and task IDs;
- checkpoint paths or checkpoint identifiers;
- batch size and training steps;
- number of evaluation episodes;
- random seeds;
- success-rate calculation method.

## Citation and License

This repository builds on LAPA, LIBERO, DepthAnythingV2, and related research code. Follow the licenses and citation requirements of all upstream projects and downloaded checkpoints.

Add project-specific citation information here when a technical report or paper becomes available.
