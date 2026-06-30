# Depth Fusion Fine-Tuning

This module is for the LAPA-depth experiment where the depth branch is owned by
another model/codebase.

## Offline LIBERO Fine-Tuning

Offline training consumes colleague-provided `.pt` shards. Each shard must include:

```python
{
    "z_rgb_feature_input": Tensor[N, 4096],
    "z_depth_feature_pred_model7_1": Tensor[N, 1024],  # or another selected depth key
    "action_vector": Tensor[N, 7],
}
```

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
