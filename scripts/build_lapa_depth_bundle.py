#!/usr/bin/env python3
"""Join LAPA JSONL sample ids with Stage-2.5 features into one training bundle."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from latent_pretraining.depth_fusion.data_libero import (
    DEPTH_FEATURE_CANDIDATES,
    PackedDepthFeatureIndex,
    _first_present_key,
    _manifest_key,
    _sequence_len,
    _value_at,
    discover_part_files,
    load_manifest,
)
from latent_pretraining.depth_fusion.id_mapping import resolve_lapa_sample_id


def csv_paths(value):
    return [Path(item.strip()) for item in str(value).split(",") if item.strip()]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_requested_ids(jsonl_path, id_key, id_source):
    ordered = []
    seen = set()
    with Path(jsonl_path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            example = json.loads(line)
            sample_id = resolve_lapa_sample_id(example, id_key=id_key, source=id_source)
            if sample_id is None:
                raise ValueError(f"Cannot resolve sample id at {jsonl_path}:{line_number}")
            sample_id = str(sample_id)
            if sample_id in seen:
                raise ValueError(f"Duplicate JSONL sample id {sample_id!r} at line {line_number}")
            seen.add(sample_id)
            ordered.append(sample_id)
    return ordered


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--depth-dirs", required=True, help="Comma-separated Stage-2.5 directories")
    parser.add_argument("--manifests", default="", help="Optional comma-separated manifests")
    parser.add_argument("--output", required=True)
    parser.add_argument("--id-key", default="id")
    parser.add_argument("--id-source", choices=("auto", "id", "image"), default="auto")
    parser.add_argument("--depth-key", default="auto")
    parser.add_argument("--depth-id-key", default="auto")
    parser.add_argument("--feature-dim", type=int, default=1024)
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    args = parser.parse_args()

    jsonl_path = Path(args.jsonl).resolve()
    output_path = Path(args.output).resolve()
    depth_dirs = csv_paths(args.depth_dirs)
    manifests = csv_paths(args.manifests)
    if manifests and len(manifests) not in (1, len(depth_dirs)):
        raise ValueError("Provide zero, one, or one manifest per depth directory")

    ordered_ids = load_requested_ids(jsonl_path, args.id_key, args.id_source)
    required = set(ordered_ids)
    found = {}
    source_files = []

    for directory_index, depth_dir in enumerate(depth_dirs):
        manifest_path = None
        if manifests:
            manifest_path = manifests[directory_index] if len(manifests) > 1 else manifests[0]
        manifest = load_manifest(manifest_path)
        part_files = discover_part_files(depth_dir, manifest_path)
        if not part_files:
            raise FileNotFoundError(f"No Stage-2.5 parts found under {depth_dir}")

        for part_file in part_files:
            source_files.append(str(Path(part_file).resolve()))
            shard = torch_load(part_file)
            depth_key = args.depth_key
            if depth_key == "auto":
                depth_key = (
                    _manifest_key(manifest, ("feature_key", "feature_key_pred", "depth_feature_key"))
                    or _first_present_key(shard, DEPTH_FEATURE_CANDIDATES)
                )
            id_key = args.depth_id_key
            if id_key == "auto":
                id_key = (
                    _manifest_key(manifest, ("id_key", "image_key", "image_path_key"))
                    or _first_present_key(shard, ("image", "image_path", "image_paths", "id", "ids"))
                )
            if not depth_key or depth_key not in shard or not id_key or id_key not in shard:
                raise KeyError(f"Cannot resolve id/depth keys in {part_file}; keys={sorted(shard)}")
            ids = shard[id_key]
            values = shard[depth_key]
            if _sequence_len(ids) != _sequence_len(values):
                raise ValueError(f"Mismatched ids/features in {part_file}")
            for row, raw_id in enumerate(ids):
                sample_id = str(raw_id)
                if sample_id not in required:
                    continue
                if sample_id in found:
                    raise ValueError(f"Duplicate Stage-2.5 sample id across shards: {sample_id}")
                value = torch.as_tensor(_value_at(values, row)).detach().cpu().flatten()
                if value.numel() != args.feature_dim:
                    raise ValueError(
                        f"Feature {sample_id} has {value.numel()} values, expected {args.feature_dim}"
                    )
                if not torch.isfinite(value).all():
                    raise ValueError(f"Non-finite depth feature: {sample_id}")
                found[sample_id] = value
            del shard

    missing = [sample_id for sample_id in ordered_ids if sample_id not in found]
    if missing:
        preview = "\n".join(missing[:20])
        raise ValueError(f"Missing {len(missing)} of {len(ordered_ids)} depth features. First ids:\n{preview}")

    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    features = torch.stack([found[sample_id] for sample_id in ordered_ids]).to(dtype).contiguous()
    metadata = {
        "jsonl": str(jsonl_path),
        "jsonl_sha256": sha256(jsonl_path),
        "sample_count": len(ordered_ids),
        "feature_dim": args.feature_dim,
        "feature_dtype": args.dtype,
        "id_key": args.id_key,
        "id_source": args.id_source,
        "source_files": source_files,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": PackedDepthFeatureIndex.FORMAT,
            "ids": ordered_ids,
            "depth_features": features,
            "metadata": metadata,
        },
        output_path,
    )
    report_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    report_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output_path), **metadata}, indent=2))


if __name__ == "__main__":
    main()
