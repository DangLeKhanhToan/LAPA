# LAPA-Depth Supplementary Code

This archive contains anonymized supplementary code for the LAPA-Depth experiments. The code supports Stage-3 downstream policy fine-tuning with depth-aware features and online LIBERO rollout evaluation.

The archive is intended for double-blind review. It does not include author names, institution names, private paths, git history, experiment logs, large datasets, or full model checkpoints.

## Configuration and Anonymity Policy

Public scripts contain no laboratory filesystem layout, scheduler account,
queue name, host name, or experiment-specific checkpoint name. All external
resources are supplied through environment variables. The scripts fail with a
descriptive message when a required variable is missing instead of guessing a
private path.

Scheduler submission files (for example PBS/QSUB jobs), local `.env` files,
one-off smoke-training utilities, and machine-specific wrappers are deliberately
excluded by `.gitignore`. Maintain those files outside Git or under an ignored
local directory such as `scripts/local/`.

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

The repository does not prescribe a private directory layout. Keep data and
checkpoints outside the source tree when possible and provide their locations
through environment variables. A generic external layout could be:

```text
<private-data-root>/
  images/
  <train-split>.jsonl
  <evaluation-split>.jsonl
  <evaluation-bin-edges>.csv
  <depth-feature-shards>/

<private-checkpoint-root>/
  <tokenizer-model>
  <visual-tokenizer-checkpoint>/
  <base-policy-parameters>/
  <stage25-checkpoint>
  <depth-estimator-checkpoint>
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
export LAPA_JSONL=/path/to/prepared/train.jsonl
export DEPTH_DATA_DIR=/path/to/depth/feature/shards
# Optional; omit this argument when shards are self-describing.
export DEPTH_MANIFEST=/path/to/depth/feature/manifest.json

python -m latent_pretraining.depth_fusion.inspect_lapa_depth_alignment \
  --jsonl "$LAPA_JSONL" \
  --depth_data_dir "$DEPTH_DATA_DIR" \
  --depth_manifest "$DEPTH_MANIFEST"
```

The expected match rate is close to `1.0`.

## Stage-3 Training

Configure all external paths explicitly, then launch training:

```bash
export SUITE=<anonymous-split-name>
export STAGE25_MODEL_NAME=<feature-extractor-name>
export TRAIN_JSONL=/path/to/prepared/train.jsonl
export IMAGE_ROOT=/path/to/image/root
export DEPTH_DATA_DIR=/path/to/depth/feature/shards
# For an aggregate dataset, provide a comma-separated directory list:
# export DEPTH_DATA_DIR=/path/to/shards-a,/path/to/shards-b
export DEPTH_MANIFEST=                         # optional
export ACTION_SCALE_FILE=/path/to/bin-edges.csv
# Alternatively set ACTION_VOCAB_SIZE directly for an aggregate dataset.
export TOKENIZER_PATH=/path/to/tokenizer.model
export VQGAN_CKPT=/path/to/visual/tokenizer/checkpoint
export LAPA_PARAMS=/path/to/base/policy/parameters
export OUTPUT_DIR=/path/to/output/root
export EXPERIMENT_ID=<anonymous-run-id>
export ACTION_FUSION_METHOD=project            # project or concat
export TOTAL_STEPS=20000
export BATCH_SIZE=128
export MESH_DIM='!-1,4,1,1'

bash scripts/train_lapa_depth_suite.sh
```

`project` is the backward-compatible default. `concat` concatenates the
4096-dimensional Stage-2 representation and the 1024-dimensional Stage-2.5
feature before the action head. The feature dimension is configurable; these
numbers describe the released experimental configuration.

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
export SUITE=<evaluation-split-name>
export STAGE25_MODEL_NAME=<feature-extractor-name>
export ACTION_FUSION_METHOD=project            # must match training
export FINETUNED_CHECKPOINT=params::/path/to/fine-tuned/parameters
export ORIGINAL_LAPA_CHECKPOINT=params::/path/to/base/policy/parameters
export ACTION_SCALE_FILE=/path/to/evaluation/bin-edges.csv
export STAGE25_MODEL_CHECKPOINT=/path/to/stage25/checkpoint
export VQGAN_CHECKPOINT=/path/to/visual/tokenizer/checkpoint
export VOCAB_FILE=/path/to/tokenizer.model
export DEPTH_BRANCH_ROOT=/path/to/stage25/source/bundle
export LIBERO_REPO=/path/to/simulator/repository
export DEPTH_ESTIMATOR_REQUIRED=true
export DEPTH_ANYTHING_REPO_DIR=/path/to/depth/estimator/source
export DEPTH_ANYTHING_CHECKPOINT=/path/to/depth/estimator/checkpoint

export POLICY_CUDA_VISIBLE_DEVICES=0
export STAGE25_CUDA_VISIBLE_DEVICES=1
export RGB_CUDA_VISIBLE_DEVICES=2
export MUJOCO_EGL_DEVICE_ID=1

export TASK_IDS="0 1 2 3 4 5 6 7 8 9"
export N_EVAL_PER_TASK=10
export MAX_STEPS=500
export PROGRESS_FREQ=25
export OUTPUT_DIR=/path/to/evaluation/output
export LOG_DIR=/path/to/server/logs

bash scripts/eval_lapa_depth_split_online_rollout.sh
```

Run multiple suites:

```bash
export SUITES="<split-a> <split-b> <split-c>"
export SHARED_CHECKPOINT=/path/to/fine-tuned/parameters
export ACTION_SCALE_FILE_TEMPLATE='/path/to/bin-edges/{suite}.csv'
export EVAL_OUTPUT_ROOT=/path/to/evaluation/output/root
export TASK_IDS="0 1 2 3 4 5 6 7 8 9"
export N_EVAL_PER_TASK=10
export MAX_STEPS=500
export OUTPUT_PREFIX="eval_lapa_depth"

bash scripts/eval_lapa_depth_split_multi_suite.sh
```

Task IDs must be separated by spaces.

## Reproducing Evaluation Results

Use the same feature-extractor configuration and `ACTION_FUSION_METHOD` used
during training. Supply one bin-edge CSV per evaluation split through
`ACTION_SCALE_FILE_TEMPLATE`. Minor variation is expected due to simulator
randomness, GPU kernels, and environment setup.

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

Videos and full rollout traces should not be included in a supplementary archive unless they are very small and explicitly needed.

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
     "*/datasets/*"
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
grep -R -E '(/home/|/Users/|[A-Za-z]:\\)' LAPA || true
```

## License

This code builds on public research software for LAPA-style policy learning, LIBERO simulation, and depth estimation. Follow the licenses and citation requirements of all upstream software and datasets used to run the experiments.
