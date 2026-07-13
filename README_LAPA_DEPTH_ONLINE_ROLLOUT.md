# LAPA-Depth Online Rollout with Stage-2.5

This rollout path uses three model components:

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
cd ~/scratch/projects/lapa-depth-modified-levi

export DEPTH_BRANCH_ROOT="$PWD/../Depth_branch"
export STAGE25_MODEL_NAME=model4
export STAGE25_MODEL_CHECKPOINT="$DEPTH_BRANCH_ROOT/model4.65000.pt"
export ORIGINAL_LAPA_CHECKPOINT="params::$PWD/lapa_checkpoints/lapa_7b_sth/params"
export DEPTH_ANYTHING_REPO_DIR="$PWD/third_party/depth_anything_v2"
export DEPTH_ANYTHING_CHECKPOINT="$PWD/checkpoints/depth_anything_v2_sth2sth/depth_anything_v2_sth2sth.pth"
export DEPTH_ANYTHING_ENCODER=vitl

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

## 3. One-Episode Online Rollout

Set the fine-tuned policy checkpoint:

```bash
cd ~/scratch/projects/lapa-depth-modified-levi

export SUITE=libero_spatial
export FINETUNED_CHECKPOINT="params::$PWD/outputs/smoke_overfit_lapa_depth_one_task/streaming_params"
export ORIGINAL_LAPA_CHECKPOINT="params::$PWD/lapa_checkpoints/lapa_7b_sth/params"
export DEPTH_BRANCH_ROOT="$PWD/../Depth_branch"
export STAGE25_MODEL_NAME=model4
export STAGE25_MODEL_CHECKPOINT="$DEPTH_BRANCH_ROOT/model4.65000.pt"
export DEPTH_ANYTHING_REPO_DIR="$PWD/third_party/depth_anything_v2"
export DEPTH_ANYTHING_CHECKPOINT="$PWD/checkpoints/depth_anything_v2_sth2sth/depth_anything_v2_sth2sth.pth"
export DEPTH_ANYTHING_ENCODER=vitl

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
