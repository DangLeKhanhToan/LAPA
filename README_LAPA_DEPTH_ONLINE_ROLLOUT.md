# LAPA-Depth Online Rollout with Stage-2.5

This rollout path uses four model components:

1. DepthAnythingV2 trained on Sth2Sth: `rgb -> depth`
2. Original LAPA: `rgb + instruction -> z_rgb_feature`
3. Stage-2.5 model2/model4: `depth + z_rgb_feature -> z_depth_feature_pred`
4. Fine-tuned LAPA-Depth policy: `rgb + instruction + z_depth_feature_pred -> action`

The implementation runs Stage-2.5 in a separate feature server process. This
avoids Python package collisions because the Stage-2.5 bundle has its own
`latent_pretraining` package.

Offline `.pt` depth features are used for Stage-3 training only. Rollout should
compute depth features online from the current observation.

## 1. Download DepthAnythingV2 Sth2Sth

Use either a local Sth2Sth checkpoint or a URL provided by the depth-branch owner.

```bash
cd ~/scratch/projects/lapa-depth-modified-levi

export DEPTH_ANYTHING_CKPT_LOCAL=/path/to/depth_anything_v2_sth2sth.pth
# or:
# export DEPTH_ANYTHING_CKPT_URL=https://.../depth_anything_v2_sth2sth.pth

bash scripts/download_depthanythingv2_sth2sth.sh
```

The script prints the two env vars used by rollout:

```bash
export DEPTH_ANYTHING_REPO_DIR="$PWD/third_party/depth_anything_v2"
export DEPTH_ANYTHING_CHECKPOINT="$PWD/checkpoints/depth_anything_v2_sth2sth/depth_anything_v2_sth2sth.pth"
```

Set the encoder to match the checkpoint:

```bash
export DEPTH_ANYTHING_ENCODER=vitl
```

## 2. Smoke Test Stage-2.5 Feature Server

Use one RGB frame and one instruction. DepthAnythingV2 will generate the depth
image online.

```bash
cd /home/linhkastner/lapa/LAPA-depth

export MODEL_PY=/mnt/hdd/linh/long/conda_envs/lapa-depth/bin/python
export LAPA_ROOT="$(pwd -P)"
export DEPTH_BRANCH_ROOT="$LAPA_ROOT/Depth_branch"
export STAGE25_MODEL_NAME=model4
export STAGE25_MODEL_CHECKPOINT="$LAPA_ROOT/lapa_checkpoints/depth_model/${STAGE25_MODEL_NAME}.65000.pt"
export ORIGINAL_LAPA_CHECKPOINT="params::$LAPA_ROOT/lapa_checkpoints/pretraining_LAPA_Sth2Sth"
export DEPTH_ANYTHING_REPO_DIR="$LAPA_ROOT/third_party/depth_anything_v2"
export DEPTH_ANYTHING_CHECKPOINT="$LAPA_ROOT/checkpoints/depth_anything_v2_sth2sth/depth_anything_v2_sth2sth.pth"
export DEPTH_ANYTHING_ENCODER=vitl
export DEPTH_ANYTHING_INPUT_SIZE=518
export DEPTH_ANYTHING_DEVICE=cuda

export STAGE25_CUDA_VISIBLE_DEVICES=2,3
export STAGE25_MESH_DIM=1,2,1,1
export JAX_PLATFORMS=cuda,cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.55
export TF_FORCE_GPU_ALLOW_GROWTH=true

export RGB_IMAGE=/path/to/rgb_step_0.jpg
export INSTRUCTION="pick up the black bowl between the plate and the ramekin and place it on the plate"

bash scripts/smoke_stage25_online_feature.sh
```

Expected:

```text
"z_depth_shape": [1024]
"depth_source": "depth_anything_v2"
```

To test model2:

```bash
export STAGE25_MODEL_NAME=model2
export STAGE25_MODEL_CHECKPOINT="$DEPTH_BRANCH_ROOT/model2.65000.pt"
bash scripts/smoke_stage25_online_feature.sh
```

If you want to debug Stage-2.5 with a manually provided depth file, set
`DEPTH_IMAGE=/path/to/depth.png`; otherwise leave it unset.

## 2.5 A5000 Unit Tests

Run these before a rollout on the 4x RTX 5000 Ada workstation.

Check JAX sees CPU and the sharded GPUs:

```bash
cd /home/linhkastner/lapa/LAPA-depth/Depth_branch

export MODEL_PY=/mnt/hdd/linh/long/conda_envs/lapa-depth/bin/python
export CUDA_VISIBLE_DEVICES=2,3
export JAX_PLATFORMS=cuda,cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false

$MODEL_PY - <<'PY'
import jax
print("devices:", jax.devices())
print("cpu:", jax.devices("cpu"))
print("gpu:", jax.devices("cuda"))
PY
```

Load model4 alone:

```bash
cd /home/linhkastner/lapa/LAPA-depth/Depth_branch

export MODEL_PY=/mnt/hdd/linh/long/conda_envs/lapa-depth/bin/python
export PYTHONPATH="$PWD:$PWD/laq"
export CUDA_VISIBLE_DEVICES=2

$MODEL_PY - <<'PY'
from rollout_stage25_model4 import build_model4
model = build_model4(checkpoint="../lapa_checkpoints/depth_model/model4.65000.pt", strict=True)
print("model4 loaded OK")
PY
```

Load original LAPA sharded across GPUs 2,3:

```bash
cd /home/linhkastner/lapa/LAPA-depth/Depth_branch

export MODEL_PY=/mnt/hdd/linh/long/conda_envs/lapa-depth/bin/python
export PYTHONPATH="$PWD:$PWD/laq"
export CUDA_VISIBLE_DEVICES=2,3
export JAX_PLATFORMS=cuda,cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.70
export TF_FORCE_GPU_ALLOW_GROWTH=true

$MODEL_PY - <<'PY'
from rollout_stage25_model4 import build_lapa
lapa = build_lapa(
    tokens_per_delta=4,
    vqgan_checkpoint="../lapa_checkpoints/vqgan",
    vocab_file="../lapa_checkpoints/tokenizer.model",
    load_checkpoint="params::../lapa_checkpoints/pretraining_LAPA_Sth2Sth",
    mesh_dim="1,2,1,1",
    dtype="bf16",
    load_llama_config="7b",
)
print("original LAPA loaded OK on GPUs 2,3")
PY
```

## 3. One-Episode Online Rollout

Set the fine-tuned policy checkpoint:

```bash
cd /home/linhkastner/lapa/LAPA-depth

export SUITE=libero_spatial
export MODEL_PY=/mnt/hdd/linh/long/conda_envs/lapa-depth/bin/python
export LIBERO_PY=/mnt/hdd/linh/long/conda_envs/lapa-depth/bin/python
export LAPA_ROOT="$(pwd -P)"
export FINETUNED_CHECKPOINT="params::$LAPA_ROOT/outputs/smoke_overfit_lapa_depth_one_task/streaming_params"
export ORIGINAL_LAPA_CHECKPOINT="params::$LAPA_ROOT/lapa_checkpoints/pretraining_LAPA_Sth2Sth"
export DEPTH_BRANCH_ROOT="$LAPA_ROOT/Depth_branch"
export STAGE25_MODEL_NAME=model4
export STAGE25_MODEL_CHECKPOINT="$LAPA_ROOT/lapa_checkpoints/depth_model/${STAGE25_MODEL_NAME}.65000.pt"
export DEPTH_ANYTHING_REPO_DIR="$LAPA_ROOT/third_party/depth_anything_v2"
export DEPTH_ANYTHING_CHECKPOINT="$LAPA_ROOT/checkpoints/depth_anything_v2_sth2sth/depth_anything_v2_sth2sth.pth"
export DEPTH_ANYTHING_ENCODER=vitl
export DEPTH_ANYTHING_INPUT_SIZE=518
export DEPTH_ANYTHING_DEVICE=cuda

export STAGE25_CUDA_VISIBLE_DEVICES=2,3
export STAGE25_MESH_DIM=1,2,1,1
export POLICY_CUDA_VISIBLE_DEVICES=0
export POLICY_MESH_DIM=1,-1,1,1
export JAX_PLATFORMS=cuda,cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.55
export TF_FORCE_GPU_ALLOW_GROWTH=true

export TASK_IDS=0
export N_EVAL_PER_TASK=1
export MAX_STEPS=80

bash scripts/eval_lapa_depth_online_rollout.sh
```

This script starts:

- Stage-2.5 feature server on port `32821`
- Fine-tuned LAPA-Depth policy server on port `32820`
- LIBERO rollout client that sends only current RGB frame + instruction
- DepthAnythingV2 runs inside the Stage-2.5 feature server

## 4. Full Rollout

After one episode works, increase the eval counts:

```bash
export SUITE=libero_spatial
export FINETUNED_CHECKPOINT="params::$PWD/outputs/lapa_depth_stage3_${SUITE}/streaming_params"
export TASK_IDS=""
export N_EVAL_PER_TASK=5
export MAX_STEPS=520

bash scripts/eval_lapa_depth_online_rollout.sh
```

## Notes

- Stage-3 training still uses offline `.pt` features from the depth branch.
- Rollout does not use `depth_id` or `.pt` depth shards.
- Rollout does not require LIBERO ground-truth depth observations.
