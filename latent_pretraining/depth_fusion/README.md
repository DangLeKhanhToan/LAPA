# Depth Fusion Fine-Tuning

This module is for the LAPA-depth experiment where the depth branch is owned by
another model/codebase.

## Offline LIBERO Fine-Tuning

Offline training consumes colleague-provided `.pt` shards. Each shard must include:

```python
{
    "z_rgb_feature_input": Tensor[N, 4096],
    "z_depth_feature_pred": Tensor[N, 1024],  # or another selected depth key
    "action_vector": Tensor[N, 7],
    "image": list[str],  # optional, useful for checking feature-to-image alignment
}
```

Keys can be supplied explicitly or discovered with `auto`. For model4 manifests,
the depth key is read from the manifest field `feature_key`, usually
`z_depth_feature_pred`.

Before training, inspect one shard:

```bash
PYTHONPATH="$PWD" python -m latent_pretraining.depth_fusion.inspect_pt_shard \
  --data_dir /path/to/stage25_model4/z_depth_train_shard0 \
  --manifest /path/to/stage25_model4/z_depth_train_shard0/z_depth_train_shard0_model4_manifest.json
```

The inspector prints all `.pt` keys, tensor shapes, inferred RGB/depth/action/image
keys, and a few sample image identifiers when present.

Your current inspected shards show two important cases:

- model4 depth shards contain `z_depth_feature_pred` and `id`, but not
  `z_rgb_feature_input` or actions.
- model2 depth shards contain `z_rgb_feature_input`, `z_depth_feature_pred`, and
  `id`, but not actions.

For training, missing sources can be joined by `id`:

```bash
PYTHONPATH="$PWD" python -m latent_pretraining.depth_fusion.train_depth_fusion \
  --data_dir /path/to/z_depth_train_shard0 \
  --manifest /path/to/z_depth_train_shard0/z_depth_train_shard0_model4_manifest.json \
  --rgb_data_dir /path/to/z_rgb_train_shard0 \
  --rgb_manifest /path/to/z_rgb_train_shard0/z_rgb_train_shard0_manifest.json \
  --action_jsonl /path/to/libero_actions.jsonl \
  --output_dir outputs/depth_fusion_smoke \
  --epochs 1 \
  --max_samples 2048 \
  --max_train_batches 8 \
  --max_val_batches 2
```

The convenience smoke script wraps the same command:

```bash
DEPTH_DATA_DIR=/path/to/z_depth_train_shard0 \
DEPTH_MANIFEST=/path/to/z_depth_train_shard0/z_depth_train_shard0_model4_manifest.json \
RGB_DATA_DIR=/path/to/z_rgb_train_shard0 \
RGB_MANIFEST=/path/to/z_rgb_train_shard0/z_rgb_train_shard0_manifest.json \
ACTION_JSONL=/path/to/libero_actions.jsonl \
./scripts/smoke_finetune_depth_fusion_libero.sh
```

After smoke training, quickly verify checkpoint loading and feature fusion:

```bash
PYTHONPATH="$PWD" python -m latent_pretraining.depth_fusion.predict_depth_fusion \
  --checkpoint outputs/depth_fusion_smoke/best.pt \
  --data_dir /path/to/z_depth_train_shard0 \
  --manifest /path/to/z_depth_train_shard0/z_depth_train_shard0_model4_manifest.json \
  --rgb_data_dir /path/to/z_rgb_train_shard0 \
  --rgb_manifest /path/to/z_rgb_train_shard0/z_rgb_train_shard0_manifest.json \
  --output_jsonl outputs/depth_fusion_smoke/predictions.jsonl \
  --max_samples 32
```

For a single server-side pipeline script, set the required paths and run:

```bash
DEPTH_DATA_DIR=/path/to/stage25_model4/z_depth_train_shard0 \
DEPTH_MANIFEST=/path/to/stage25_model4/z_depth_train_shard0/z_depth_train_shard0_model4_manifest.json \
RGB_DATA_DIR=/path/to/z_rgb_train_shard0 \
RGB_MANIFEST=/path/to/z_rgb_train_shard0/z_rgb_train_shard0_manifest.json \
ACTION_JSONL=/path/to/libero_actions.jsonl \
./scripts/run_depth_fusion_stage3_pipeline.sh
```

This runs inspection, smoke fine-tuning, smoke checkpoint prediction, and full
offline depth-fusion fine-tuning in order.

The training model is:

```text
z_rgb_feature_input (4096) + selected depth feature (1024)
    -> DepthFusionPolicy MLP
    -> action_vector (7)
```

The depth branch is not implemented here for offline training. We only load its
precomputed 1024-dimensional feature.

## Online Inference Contract

When the online depth model is ready, implement:

```python
depth_feature = depth_branch.encode(
    rgb_image,
    instruction=instruction,
    latent_action_4096=rgb_feature,
)
```

It must return `np.ndarray` with shape `(1024,)` and dtype `float32`.
