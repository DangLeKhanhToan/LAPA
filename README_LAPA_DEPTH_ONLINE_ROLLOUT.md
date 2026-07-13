# LAPA-Depth Online Rollout with Stage-2.5

This rollout path uses three model components:

1. Original LAPA: `rgb + instruction -> z_rgb_feature`
2. Stage-2.5 model2/model4: `depth + z_rgb_feature -> z_depth_feature_pred`
3. Fine-tuned LAPA-Depth policy: `rgb + instruction + z_depth_feature_pred -> action`

The implementation runs Stage-2.5 in a separate feature server process. This
avoids Python package collisions because the Stage-2.5 bundle has its own
`latent_pretraining` package.

## 1. Smoke Test Stage-2.5 Feature Server

Use one RGB/depth pair and one instruction.

```bash
cd ~/scratch/projects/lapa-depth-modified-levi

export DEPTH_BRANCH_ROOT="$PWD/../Depth_branch"
export STAGE25_MODEL_NAME=model4
export STAGE25_MODEL_CHECKPOINT="$DEPTH_BRANCH_ROOT/model4.65000.pt"
export ORIGINAL_LAPA_CHECKPOINT="params::$PWD/lapa_checkpoints/lapa_7b_sth/params"

export RGB_IMAGE=/path/to/rgb_step_0.jpg
export DEPTH_IMAGE=/path/to/depth_step_0.png
export INSTRUCTION="pick up the black bowl between the plate and the ramekin and place it on the plate"

bash scripts/smoke_stage25_online_feature.sh
```

Expected:

```text
"z_depth_shape": [1024]
```

To test model2:

```bash
export STAGE25_MODEL_NAME=model2
export STAGE25_MODEL_CHECKPOINT="$DEPTH_BRANCH_ROOT/model2.65000.pt"
bash scripts/smoke_stage25_online_feature.sh
```

## 2. One-Episode Online Rollout

Set the fine-tuned policy checkpoint:

```bash
cd ~/scratch/projects/lapa-depth-modified-levi

export SUITE=libero_spatial
export FINETUNED_CHECKPOINT="params::$PWD/outputs/smoke_overfit_lapa_depth_one_task/streaming_params"
export ORIGINAL_LAPA_CHECKPOINT="params::$PWD/lapa_checkpoints/lapa_7b_sth/params"
export DEPTH_BRANCH_ROOT="$PWD/../Depth_branch"
export STAGE25_MODEL_NAME=model4
export STAGE25_MODEL_CHECKPOINT="$DEPTH_BRANCH_ROOT/model4.65000.pt"

export TASK_IDS=0
export N_EVAL_PER_TASK=1
export MAX_STEPS=80

bash scripts/eval_lapa_depth_online_rollout.sh
```

This script starts:

- Stage-2.5 feature server on port `32821`
- Fine-tuned LAPA-Depth policy server on port `32820`
- LIBERO rollout client with `--send_depth_image`

## 3. Full Rollout

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

- Offline rollout still exists through `depth_id` and `.pt` feature shards.
- Online rollout uses simulator depth observations and does not need depth IDs.
- If LIBERO depth keys are missing, first test the env with `camera_depths=True`.
