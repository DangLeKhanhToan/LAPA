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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("rollout_dirs", nargs="+")
    parser.add_argument("--csv", dest="csv_path", default="")
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
                "suite": suite,
                **metrics,
                "n_unresolved_targets": unresolved,
                "n_fallback_targets": fallback,
            }
            rows.append(row)

            print(f"\n[{suite}] {metrics['n_episodes']} episodes")
            print(
                f"In {metrics['n_episodes']} {suite} episodes, "
                f"{percent(metrics['approached_original_target_location_rate'])} "
                "approached the original target location, "
                f"{percent(metrics['grasped_wrong_object_rate'])} grasped a wrong object, "
                f"{percent(metrics['grasped_correct_object_rate'])} grasped the correct object, "
                "and "
                f"{percent(metrics['drop_rate_among_episodes_with_grasp'])} of episodes "
                "that grasped an object dropped it before task success."
            )
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


if __name__ == "__main__":
    main()
