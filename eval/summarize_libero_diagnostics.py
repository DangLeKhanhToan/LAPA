"""Summarize one or more LIBERO diagnostic rollout directories."""

import argparse
import csv
import gzip
import json
import os

from eval.libero_diagnostics import aggregate_episode_summaries


def load_episodes(root):
    episodes = []
    diagnostic_root = os.path.join(root, "diagnostics")
    for current_root, _, files in os.walk(diagnostic_root):
        for filename in sorted(files):
            if not filename.endswith(".json.gz"):
                continue
            path = os.path.join(current_root, filename)
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                payload = json.load(handle)
            episodes.append(
                {
                    "path": path,
                    "meta": payload["meta"],
                    "summary": payload["summary"],
                }
            )
    return episodes


def percent(value):
    return "NA" if value is None else f"{100.0 * value:.1f}%"


def narrative(suite, metrics):
    return (
        f"In {metrics['n_episodes']} {suite} episodes, "
        f"{percent(metrics['approached_original_target_location_rate'])} "
        "approached the original target location, "
        f"{percent(metrics['grasped_wrong_object_rate'])} grasped a wrong object, "
        f"{percent(metrics['grasped_correct_object_rate'])} grasped the correct object, "
        "and "
        f"{percent(metrics['drop_rate_among_episodes_with_grasp'])} of episodes "
        "that grasped an object dropped it before task success."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("rollout_dirs", nargs="+")
    parser.add_argument("--csv", dest="csv_path", default="")
    parser.add_argument("--json", dest="json_path", default="")
    parser.add_argument("--markdown", dest="markdown_path", default="")
    args = parser.parse_args()

    rows = []
    for rollout_dir in args.rollout_dirs:
        episodes = load_episodes(rollout_dir)
        suites = sorted(set(ep["meta"]["suite"] for ep in episodes))
        for suite in suites:
            selected = [ep for ep in episodes if ep["meta"]["suite"] == suite]
            metrics = aggregate_episode_summaries(selected)
            unresolved = sum(ep["meta"].get("target_object") is None for ep in selected)
            fallback = sum(
                ep["meta"].get("target_selection_method")
                == "fallback_first_movable_object"
                for ep in selected
            )
            row = {
                "rollout_dir": os.path.realpath(rollout_dir),
                "run_name": os.path.basename(os.path.realpath(rollout_dir)),
                "suite": suite,
                **metrics,
                "n_unresolved_targets": unresolved,
                "n_fallback_targets": fallback,
            }
            row["narrative"] = narrative(suite, metrics)
            rows.append(row)

            print(f"\n[{suite}] {metrics['n_episodes']} episodes")
            print(row["narrative"])
            print(
                "Validation: "
                f"exact_grasp={metrics['n_episodes_with_exact_grasp_check']}/"
                f"{metrics['n_episodes']}, unresolved_target={unresolved}, "
                f"fallback_target={fallback}"
            )

    if args.csv_path:
        os.makedirs(os.path.dirname(os.path.realpath(args.csv_path)), exist_ok=True)
        fieldnames = sorted(set().union(*(row.keys() for row in rows))) if rows else []
        with open(args.csv_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote {args.csv_path}")

    if args.json_path:
        os.makedirs(os.path.dirname(os.path.realpath(args.json_path)), exist_ok=True)
        with open(args.json_path, "w", encoding="utf-8") as handle:
            json.dump(rows, handle, indent=2)
        print(f"Wrote {args.json_path}")

    if args.markdown_path:
        os.makedirs(os.path.dirname(os.path.realpath(args.markdown_path)), exist_ok=True)
        with open(args.markdown_path, "w", encoding="utf-8") as handle:
            handle.write("# LIBERO-Pro diagnostic summary\n\n")
            handle.write(
                "| Run | Suite | Episodes | Success | Original location | "
                "Wrong object | Correct object | Drop after grasp |\n"
            )
            handle.write("|---|---|---:|---:|---:|---:|---:|---:|\n")
            for row in rows:
                handle.write(
                    f"| {row['run_name']} | {row['suite']} | {row['n_episodes']} | "
                    f"{percent(row['success_rate'])} | "
                    f"{percent(row['approached_original_target_location_rate'])} | "
                    f"{percent(row['grasped_wrong_object_rate'])} | "
                    f"{percent(row['grasped_correct_object_rate'])} | "
                    f"{percent(row['drop_rate_among_episodes_with_grasp'])} |\n"
                )
            handle.write("\n## Narrative\n\n")
            for row in rows:
                handle.write(f"- **{row['run_name']}**: {row['narrative']}\n")
        print(f"Wrote {args.markdown_path}")


if __name__ == "__main__":
    main()
