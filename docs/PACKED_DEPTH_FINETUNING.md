# Packed Stage-2.5 fine-tuning

This workflow keeps the original LAPA JSONL, model, loss, action bins, batch
size, and sequence length. It replaces per-sample Stage-2.5 shard loading with
one prejoined contiguous tensor bundle loaded once into CPU RAM.

## 1. Build the all-suite bundle

Run from the repository root after activating the LAPA training environment.
The directory and manifest lists must use the same suite order.

```bash
MODEL=model2 bash scripts/local/prepare_lapa_depth_bundle.sh
```

If `all_train.jsonl` is absent, the preparation script creates
`combined/all_suite_train.jsonl` from the four suite files. The comparison
launcher selects the same file automatically.

The builder stops on missing IDs, duplicate IDs, wrong dimensions, and
non-finite features. It also writes a `.manifest.json` audit file.

## 2. Ten-step smoke tests

Use the same interactive allocation and four visible GPUs for both runs.

```bash
MODE=baseline RUN_KIND=smoke MODEL=model2 \
  bash scripts/local/finetune_lapa_comparison.sh

MODE=depth_concat RUN_KIND=smoke MODEL=model2 \
  bash scripts/local/finetune_lapa_comparison.sh
```

Verify that both runs complete ten steps, losses are finite, the depth run
prints `packed_depth_features`, and batch fetch time remains small after the
first JAX compilation step.

## 3. Full 20K runs

```bash
MODE=baseline RUN_KIND=full MODEL=model2 \
  bash scripts/local/finetune_lapa_comparison.sh

MODE=depth_concat RUN_KIND=full MODEL=model2 \
  bash scripts/local/finetune_lapa_comparison.sh
```

The comparison launcher disables intermediate milestones and optimizer-state
saving so checkpoint I/O does not bias timing. Full runs save final params at
step 20K. Give distinct `EXPERIMENT_ID` values for repeated seeds.
