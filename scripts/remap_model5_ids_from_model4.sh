#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." >/dev/null 2>&1 && pwd)"
cd "$PROJECT_DIR"

SUITE="${SUITE:-libero_spatial}"
FEATURE_ROOT="${FEATURE_ROOT:-$PROJECT_DIR/datasets/features_depth_branch}"

MODEL4_DIR="${MODEL4_DIR:-$FEATURE_ROOT/stage25_libero_features_model4/$SUITE/stage25_model4/z_depth_train_shard0}"
MODEL5_DIR="${MODEL5_DIR:-$FEATURE_ROOT/stage25_libero_features_model5/$SUITE/stage25_model5/z_depth_train_shard0}"

# Write a corrected copy so the original Model5 features remain recoverable.
OUTPUT_DIR="${OUTPUT_DIR:-$FEATURE_ROOT/stage25_libero_features_model5_semantic_ids/$SUITE/stage25_model5/z_depth_train_shard0}"

if [[ ! -d "$MODEL4_DIR" ]]; then
  echo "ERROR: Model4 feature directory not found: $MODEL4_DIR" >&2
  exit 1
fi
if [[ ! -d "$MODEL5_DIR" ]]; then
  echo "ERROR: Model5 feature directory not found: $MODEL5_DIR" >&2
  exit 1
fi
if [[ -e "$OUTPUT_DIR" ]] && [[ -n "$(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "ERROR: output directory is not empty: $OUTPUT_DIR" >&2
  echo "Choose another OUTPUT_DIR; the script never overwrites an existing mapping." >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "[remap-model5-ids] suite:       $SUITE"
echo "[remap-model5-ids] model4 dir:  $MODEL4_DIR"
echo "[remap-model5-ids] model5 dir:  $MODEL5_DIR"
echo "[remap-model5-ids] output dir:  $OUTPUT_DIR"

python3 - "$MODEL4_DIR" "$MODEL5_DIR" "$OUTPUT_DIR" "$SUITE" <<'PY'
import gc
import json
import os
import sys
from pathlib import Path

import torch


model4_dir = Path(sys.argv[1]).resolve()
model5_dir = Path(sys.argv[2]).resolve()
output_dir = Path(sys.argv[3]).resolve()
suite = sys.argv[4]


def discover_parts(directory: Path):
    parts = sorted(directory.glob("*_part*.pt"))
    if not parts:
        parts = sorted(directory.glob("*_part*.pth"))
    return parts


model4_parts = discover_parts(model4_dir)
model5_parts = discover_parts(model5_dir)

if not model4_parts:
    raise SystemExit(f"ERROR: no Model4 part files found under {model4_dir}")
if not model5_parts:
    raise SystemExit(f"ERROR: no Model5 part files found under {model5_dir}")
if len(model4_parts) != len(model5_parts):
    raise SystemExit(
        "ERROR: different number of parts: "
        f"Model4={len(model4_parts)}, Model5={len(model5_parts)}"
    )

output_dir.mkdir(parents=True, exist_ok=True)
output_parts = []
seen_semantic_ids = set()
seen_model5_source_ids = set()
source_id_formats = set()
total_samples = 0

for part_index, (model4_path, model5_path) in enumerate(
    zip(model4_parts, model5_parts)
):
    model4_shard = torch.load(model4_path, map_location="cpu")
    model5_shard = torch.load(model5_path, map_location="cpu")

    if "id" not in model4_shard:
        raise SystemExit(f"ERROR: Model4 shard has no 'id': {model4_path}")
    if "id" not in model5_shard:
        raise SystemExit(f"ERROR: Model5 shard has no 'id': {model5_path}")

    model4_ids = [str(value) for value in model4_shard["id"]]
    model5_ids = [str(value) for value in model5_shard["id"]]

    if len(model4_ids) != len(model5_ids):
        raise SystemExit(
            f"ERROR: part {part_index} length mismatch: "
            f"Model4={len(model4_ids)}, Model5={len(model5_ids)}"
        )

    expected_shard_local_ids = [
        f"{part_index}:{local_index}"
        for local_index in range(len(model5_ids))
    ]
    if model5_ids == expected_shard_local_ids:
        source_id_format = "shard_index:local_index"
    else:
        # Some suites (notably libero_90) retain upstream image IDs such as
        # "1_img0001".  Their part boundaries still match Model4 exactly, so
        # positional remapping remains valid without interpreting the old ID.
        source_id_format = "external_unique_id"
    source_id_formats.add(source_id_format)

    duplicate_source_ids = seen_model5_source_ids.intersection(model5_ids)
    if duplicate_source_ids:
        example = next(iter(duplicate_source_ids))
        raise SystemExit(
            f"ERROR: duplicate Model5 source ID across parts: {example!r}"
        )
    if len(set(model5_ids)) != len(model5_ids):
        raise SystemExit(
            f"ERROR: duplicate Model5 source IDs inside part {part_index}"
        )
    seen_model5_source_ids.update(model5_ids)

    duplicates = seen_semantic_ids.intersection(model4_ids)
    if duplicates:
        example = next(iter(duplicates))
        raise SystemExit(
            f"ERROR: duplicate Model4 semantic ID across parts: {example!r}"
        )
    seen_semantic_ids.update(model4_ids)

    # Preserve every Model5 tensor/value and replace only its storage IDs.
    model5_shard["id"] = list(model4_ids)

    output_path = output_dir / model5_path.name
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save(model5_shard, temporary_path)
    os.replace(temporary_path, output_path)

    # Read the saved file back before accepting it.
    verified = torch.load(output_path, map_location="cpu")
    verified_ids = [str(value) for value in verified["id"]]
    if verified_ids != model4_ids:
        raise SystemExit(f"ERROR: saved ID verification failed: {output_path}")

    count = len(model4_ids)
    total_samples += count
    output_parts.append(
        {
            "part": part_index,
            "path": str(output_path),
            "num_samples": count,
        }
    )
    print(
        f"[part {part_index}] count={count} "
        f"source_format={source_id_format} "
        f"source_first={model5_ids[0]!r} -> semantic_first={model4_ids[0]!r}"
    )

    del model4_shard, model5_shard, verified
    gc.collect()


source_manifests = sorted(model5_dir.glob("*manifest.json"))
if len(source_manifests) > 1:
    raise SystemExit(
        "ERROR: multiple Model5 manifests found: "
        + ", ".join(str(path) for path in source_manifests)
    )

if source_manifests:
    source_manifest = json.loads(source_manifests[0].read_text())
    manifest_name = source_manifests[0].name
else:
    source_manifest = {}
    manifest_name = "z_depth_train_shard0_model5_manifest.json"

manifest = dict(source_manifest)
manifest.update(
    {
        "total_samples": total_samples,
        "num_parts": len(output_parts),
        "id_key": "id",
        "id_format": "model4_semantic_image_id",
        "id_mapping": {
            "suite": suite,
            "method": "same_part_and_local_index",
            "model5_source_id_formats": sorted(source_id_formats),
            "semantic_id_source": str(model4_dir),
            "feature_source": str(model5_dir),
        },
        "parts": output_parts,
    }
)

manifest_path = output_dir / manifest_name
temporary_manifest = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
temporary_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
os.replace(temporary_manifest, manifest_path)

if len(seen_semantic_ids) != total_samples:
    raise SystemExit(
        "ERROR: final unique-ID verification failed: "
        f"unique={len(seen_semantic_ids)}, total={total_samples}"
    )
if len(seen_model5_source_ids) != total_samples:
    raise SystemExit(
        "ERROR: final Model5 source-ID verification failed: "
        f"unique={len(seen_model5_source_ids)}, total={total_samples}"
    )

print(
    json.dumps(
        {
            "status": "ok",
            "suite": suite,
            "total_samples": total_samples,
            "unique_semantic_ids": len(seen_semantic_ids),
            "unique_model5_source_ids": len(seen_model5_source_ids),
            "model5_source_id_formats": sorted(source_id_formats),
            "num_parts": len(output_parts),
            "manifest": str(manifest_path),
            "output_dir": str(output_dir),
        },
        indent=2,
    )
)
PY

echo
echo "[remap-model5-ids] Completed without modifying the original Model5 files."
echo "[remap-model5-ids] For a smoke test, set:"
echo "DEPTH_DATA_DIR=\"$OUTPUT_DIR\""
echo "DEPTH_MANIFEST=\"$OUTPUT_DIR/z_depth_train_shard0_model5_manifest.json\""
