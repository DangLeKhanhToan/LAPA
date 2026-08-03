# LAPA-Depth


## Getting started

Install the core Python dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

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

## Data Preparation

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
{"instruction": "...", "image": "images/<suite>/<task>/<episode>/<step>.jpg", "raw_actions": [...], "action": ["..."], "fields": "[instruction],[vision],action"}
```
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
export TOTAL_STEPS=20000
export BATCH_SIZE=128
export MESH_DIM='!-1,4,1,1'

bash scripts/train_lapa_depth_suite.sh
```

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
## License

This code builds on public research software for LAPA-style policy learning, LIBERO simulation, and depth estimation. Follow the licenses and citation requirements of all upstream software and datasets used to run the experiments.
