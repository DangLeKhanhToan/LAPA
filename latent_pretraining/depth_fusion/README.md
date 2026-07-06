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
