#!/usr/bin/env python3
"""Preflight validation for offline LIBERO depth-feature .pt shards.

The scanner loads one shard at a time on CPU and never participates in the
training graph. A nonzero exit status means at least one shard is unsafe to
use for training.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import torch


FEATURE_KEY_CANDIDATES = (
    "z_depth_feature_pred",
    "pred_z_depth_feature",
    "z_depth_feature_pred_model7_1",
    "z_depth_feature_gt",
    "z_depth_feature",
    "depth_feature",
)
ID_KEY_CANDIDATES = (
    "id",
    "ids",
    "image",
    "image_path",
    "image_paths",
    "file_name",
    "file_names",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Recursively scan offline depth-feature shards before training."
    )
    parser.add_argument(
        "roots",
        nargs="+",
        type=Path,
        help="One or more model feature roots, e.g. stage25_libero_features_model2.",
    )
    parser.add_argument(
        "--feature-key",
        default="auto",
        help="Feature key to scan. Default: resolve separately for each shard.",
    )
    parser.add_argument(
        "--expected-dim",
        type=int,
        default=1024,
        help="Required final feature dimension. Set 0 to disable. Default: 1024.",
    )
    parser.add_argument(
        "--warn-absmax",
        type=float,
        default=100.0,
        help="Warn when a finite absolute value exceeds this value. Default: 100.",
    )
    parser.add_argument(
        "--max-error-examples",
        type=int,
        default=20,
        help="Maximum detailed error locations retained in the report.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional path for the complete JSON report.",
    )
    return parser.parse_args()


def discover_parts(root):
    return sorted(
        set(root.rglob("*_part*.pt")) | set(root.rglob("*_part*.pth"))
    )


def first_present(mapping, candidates):
    for key in candidates:
        if key in mapping:
            return key
    return None


def sequence_length(value):
    if torch.is_tensor(value):
        return int(value.shape[0]) if value.ndim else None
    if isinstance(value, (list, tuple)):
        return len(value)
    return None


def update_extrema(summary, finite_values):
    if finite_values.numel() == 0:
        return
    shard_min = float(finite_values.min().item())
    shard_max = float(finite_values.max().item())
    shard_absmax = float(finite_values.abs().max().item())
    summary["finite_min"] = min(summary["finite_min"], shard_min)
    summary["finite_max"] = max(summary["finite_max"], shard_max)
    summary["finite_absmax"] = max(summary["finite_absmax"], shard_absmax)
    summary["finite_sum"] += float(finite_values.double().sum().item())


def add_error(summary, args, kind, part, **details):
    summary["errors"] += 1
    if len(summary["error_examples"]) < args.max_error_examples:
        summary["error_examples"].append(
            {"kind": kind, "file": str(part), **details}
        )


def scan_root(root, args):
    parts = discover_parts(root)
    summary = {
        "root": str(root),
        "files": len(parts),
        "files_scanned": 0,
        "samples": 0,
        "values": 0,
        "finite_values": 0,
        "nan_values": 0,
        "posinf_values": 0,
        "neginf_values": 0,
        "finite_min": math.inf,
        "finite_max": -math.inf,
        "finite_absmax": 0.0,
        "finite_sum": 0.0,
        "feature_keys": {},
        "shapes": {},
        "warnings": [],
        "errors": 0,
        "error_examples": [],
    }
    if not root.is_dir():
        add_error(summary, args, "root_not_found", root)
        summary.pop("finite_sum")
        summary["finite_min"] = None
        summary["finite_max"] = None
        summary["finite_absmax"] = None
        summary["finite_mean"] = None
        return summary
    if not parts:
        add_error(summary, args, "no_part_files", root)
        summary.pop("finite_sum")
        summary["finite_min"] = None
        summary["finite_max"] = None
        summary["finite_absmax"] = None
        summary["finite_mean"] = None
        return summary

    for file_index, part in enumerate(parts, start=1):
        print(
            f"[scan-depth] {root.name}: {file_index}/{len(parts)} {part}",
            file=sys.stderr,
            flush=True,
        )
        try:
            shard = torch.load(part, map_location="cpu", weights_only=False)
        except Exception as exc:
            add_error(
                summary,
                args,
                "load_failed",
                part,
                exception=f"{type(exc).__name__}: {exc}",
            )
            continue
        if not isinstance(shard, dict):
            add_error(
                summary,
                args,
                "not_a_dict",
                part,
                actual_type=type(shard).__name__,
            )
            continue

        feature_key = (
            first_present(shard, FEATURE_KEY_CANDIDATES)
            if args.feature_key == "auto"
            else args.feature_key
        )
        if feature_key is None or feature_key not in shard:
            add_error(
                summary,
                args,
                "feature_key_missing",
                part,
                requested=args.feature_key,
                available_keys=sorted(shard),
            )
            continue
        feature = shard[feature_key]
        if not torch.is_tensor(feature):
            add_error(
                summary,
                args,
                "feature_not_tensor",
                part,
                feature_key=feature_key,
                actual_type=type(feature).__name__,
            )
            continue
        if feature.ndim < 2:
            add_error(
                summary,
                args,
                "invalid_feature_rank",
                part,
                shape=list(feature.shape),
            )
            continue

        samples = int(feature.shape[0])
        shape_key = str(list(feature.shape[1:]))
        summary["feature_keys"][feature_key] = (
            summary["feature_keys"].get(feature_key, 0) + 1
        )
        summary["shapes"][shape_key] = summary["shapes"].get(shape_key, 0) + 1
        summary["files_scanned"] += 1
        summary["samples"] += samples
        summary["values"] += feature.numel()

        if args.expected_dim and int(feature.shape[-1]) != args.expected_dim:
            add_error(
                summary,
                args,
                "unexpected_feature_dim",
                part,
                shape=list(feature.shape),
                expected_dim=args.expected_dim,
            )

        id_key = first_present(shard, ID_KEY_CANDIDATES)
        if id_key is not None:
            id_count = sequence_length(shard[id_key])
            if id_count != samples:
                add_error(
                    summary,
                    args,
                    "sample_id_count_mismatch",
                    part,
                    feature_samples=samples,
                    id_key=id_key,
                    id_count=id_count,
                )

        if feature.is_floating_point():
            finite_mask = torch.isfinite(feature)
            nan_count = int(torch.isnan(feature).sum().item())
            posinf_count = int(torch.isposinf(feature).sum().item())
            neginf_count = int(torch.isneginf(feature).sum().item())
            finite_count = int(finite_mask.sum().item())
            summary["finite_values"] += finite_count
            summary["nan_values"] += nan_count
            summary["posinf_values"] += posinf_count
            summary["neginf_values"] += neginf_count
            update_extrema(summary, feature[finite_mask])
            if nan_count or posinf_count or neginf_count:
                bad_rows = torch.nonzero(
                    ~finite_mask.reshape(samples, -1).all(dim=1),
                    as_tuple=False,
                ).flatten()
                add_error(
                    summary,
                    args,
                    "nonfinite_feature",
                    part,
                    nan=nan_count,
                    posinf=posinf_count,
                    neginf=neginf_count,
                    first_bad_sample_rows=bad_rows[:10].tolist(),
                )
        else:
            add_error(
                summary,
                args,
                "feature_not_floating_point",
                part,
                dtype=str(feature.dtype),
            )

        del shard, feature

    if summary["finite_values"]:
        summary["finite_mean"] = (
            summary.pop("finite_sum") / summary["finite_values"]
        )
    else:
        summary.pop("finite_sum")
        summary["finite_min"] = None
        summary["finite_max"] = None
        summary["finite_absmax"] = None
        summary["finite_mean"] = None
    if (
        summary["finite_absmax"] is not None
        and summary["finite_absmax"] > args.warn_absmax
    ):
        summary["warnings"].append(
            {
                "kind": "large_absolute_value",
                "finite_absmax": summary["finite_absmax"],
                "threshold": args.warn_absmax,
            }
        )
    return summary


def main():
    args = parse_args()
    summaries = [scan_root(root.resolve(), args) for root in args.roots]
    report = {
        "ok": all(summary["errors"] == 0 for summary in summaries),
        "roots": summaries,
        "totals": {
            "files": sum(summary["files"] for summary in summaries),
            "files_scanned": sum(
                summary["files_scanned"] for summary in summaries
            ),
            "samples": sum(summary["samples"] for summary in summaries),
            "values": sum(summary["values"] for summary in summaries),
            "nan_values": sum(summary["nan_values"] for summary in summaries),
            "posinf_values": sum(
                summary["posinf_values"] for summary in summaries
            ),
            "neginf_values": sum(
                summary["neginf_values"] for summary in summaries
            ),
            "errors": sum(summary["errors"] for summary in summaries),
        },
    }
    output = json.dumps(report, indent=2, allow_nan=False)
    print(output)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(output + "\n")
    raise SystemExit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
