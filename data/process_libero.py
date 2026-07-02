"""
Convert LIBERO HDF5 demonstration files to LAPA-compatible JSONL format.

Output layout:
  {output_dir}/images/{suite}/{task_stem}/demo_{i}/step_{t}.jpg
  {output_dir}/{suite}_train.jsonl
  {output_dir}/{suite}_test.jsonl
  {output_dir}/all_train.jsonl   (shuffled merge of all train splits)
  {output_dir}/all_test.jsonl
  {output_dir}/action_bins.csv   (bin edges for deployment server)

Usage:
  python data/process_libero.py \
      --libero_root datasets/libero_raw \
      --output_dir  datasets/lapa_libero
"""

import argparse
import json
import random
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from PIL import Image


# Demo index ranges for each suite.
# None means that split is empty for this suite.
SUITE_SPLITS = {
    'libero_goal':    (range(45), range(45, 50)),
    'libero_object':  (range(45), range(45, 50)),
    'libero_spatial': (range(45), range(45, 50)),
    'libero_90':      (range(50), None),   # entire suite = train (for LIBERO-100)
    'libero_10':      (None, range(50)),   # entire suite = test  (for LIBERO-100)
}


def assign_bin(data_point, bins):
    """Map a scalar value to its bin index (same logic as finetune_preprocess.py)."""
    for i in range(len(bins) - 1):
        if bins[i] <= data_point < bins[i + 1]:
            return i
    if data_point >= bins[len(bins) - 1]:
        return len(bins) - 2
    if data_point <= bins[0]:
        return 0
    return None


def process_hdf5(hdf5_path, suite, output_dir, camera):
    """
    Save images and collect raw records from one HDF5 file.

    Returns:
        task_stem (str): filename stem used as image subdirectory name.
        records (list): tuples of (demo_idx, step, instruction, raw_action, img_rel_path).
    """
    task_stem = Path(hdf5_path).stem
    records = []

    with h5py.File(hdf5_path, 'r') as f:
        problem_info = json.loads(f['data'].attrs['problem_info'])
        instruction = problem_info['language_instruction']
        num_demos = int(f['data'].attrs['num_demos'])

        for demo_idx in range(num_demos):
            demo_key = f'data/demo_{demo_idx}'
            actions = f[f'{demo_key}/actions'][:]          # (T, 7) float64
            images_arr = f[f'{demo_key}/obs/{camera}'][:]  # (T, H, W, 3) uint8
            T = actions.shape[0]

            img_base = output_dir / 'images' / suite / task_stem / f'demo_{demo_idx}'
            img_base.mkdir(parents=True, exist_ok=True)

            for t in range(T):
                img_rel = f'images/{suite}/{task_stem}/demo_{demo_idx}/step_{t}.jpg'
                img_abs = output_dir / img_rel
                if not img_abs.exists():
                    Image.fromarray(images_arr[t]).save(img_abs)

                records.append((demo_idx, t, instruction, actions[t].tolist(), img_rel))

    return task_stem, records


def main(args):
    libero_root = Path(args.libero_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    suites = args.suites if args.suites else list(SUITE_SPLITS.keys())
    random.seed(args.seed)

    # ── Step 1: Extract images and collect records ────────────────────────────
    suite_records = {}  # suite -> {task_stem -> [records]}
    for suite in suites:
        suite_dir = libero_root / suite
        if not suite_dir.exists():
            print(f"[warn] skipping missing suite dir: {suite_dir}")
            continue
        hdf5_files = sorted(suite_dir.glob('*.hdf5'))
        print(f"\n[{suite}] {len(hdf5_files)} task files")
        suite_records[suite] = {}
        for hdf5_path in hdf5_files:
            print(f"  {hdf5_path.name} ...", flush=True)
            task_stem, records = process_hdf5(hdf5_path, suite, output_dir, args.camera)
            suite_records[suite][task_stem] = records

    # ── Step 2: Compute global action bins from training demos only ───────────
    dim_lists = [[] for _ in range(7)]
    for suite in suites:
        if suite not in suite_records:
            continue
        train_range, _ = SUITE_SPLITS[suite]
        if train_range is None:
            continue
        for records in suite_records[suite].values():
            for (demo_idx, _step, _inst, raw_action, _img) in records:
                if demo_idx in train_range:
                    for dim in range(7):
                        dim_lists[dim].append(raw_action[dim])

    print("\nComputing action bins ...")
    total_bin = []
    for dim in range(7):
        series = pd.Series(dim_lists[dim])
        _, bins = pd.qcut(
            series, args.discretize_bins, labels=False, retbins=True, duplicates='drop'
        )
        total_bin.append(bins)
        print(f"  dim {dim}: {len(bins)-1} bins  [{bins[0]:.4f}, {bins[-1]:.4f}]")

    # Gripper override: LIBERO gripper is in [-1, 1]; treat as binary open/close.
    # bin 0 -> [-1.5, 0.0) -> open (-1)
    # bin 1 -> [0.0,  1.5) -> close (1)
    total_bin[6] = np.array([-1.5, 0.0, 1.5])

    pd.DataFrame(total_bin).to_csv(output_dir / 'action_bins.csv', index=False)
    print("Saved action_bins.csv")

    # ── Step 3 & 4: Write per-suite JSONL files and combined files ────────────
    def make_record(instruction, raw_action, img_rel):
        action_bins = [str(assign_bin(raw_action[i], total_bin[i])) for i in range(7)]
        # Prompt must match the inference sampler (DeltaActionSampler.__call__), which wraps
        # the task language as: "... USER: What action should the robot take to `{task}` ASSISTANT: ..."
        # Training/eval prompt consistency is critical for the fine-tuned action head.
        return {
            'instruction': f'<s> You are a helpful assistant. USER: What action should the robot take to `{instruction}` ASSISTANT:',
            'image': img_rel,
            'raw_actions': raw_action,
            'action': action_bins,
            'fields': '[instruction],[vision],action',
        }

    all_train, all_test = [], []

    for suite in suites:
        if suite not in suite_records:
            continue
        train_range, test_range = SUITE_SPLITS[suite]
        train_out, test_out = [], []

        for records in suite_records[suite].values():
            for (demo_idx, _step, instruction, raw_action, img_rel) in records:
                rec = make_record(instruction, raw_action, img_rel)
                if train_range is not None and demo_idx in train_range:
                    train_out.append(rec)
                elif test_range is not None and demo_idx in test_range:
                    test_out.append(rec)

        def write_jsonl(path, records_list):
            with open(path, 'w') as fout:
                for rec in records_list:
                    fout.write(json.dumps(rec) + '\n')
            print(f"  Wrote {len(records_list):>8,} records → {Path(path).name}")

        print(f"\n[{suite}]")
        if train_out:
            write_jsonl(output_dir / f'{suite}_train.jsonl', train_out)
            all_train.extend(train_out)
        if test_out:
            write_jsonl(output_dir / f'{suite}_test.jsonl', test_out)
            all_test.extend(test_out)

    print("\n[combined]")
    random.shuffle(all_train)
    write_jsonl(output_dir / 'all_train.jsonl', all_train)
    write_jsonl(output_dir / 'all_test.jsonl', all_test)
    print("\nDone.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Convert LIBERO HDF5 demos to LAPA JSONL format.'
    )
    parser.add_argument(
        '--libero_root', type=str, default='datasets/libero_raw',
        help='Root dir containing suite subdirs with .hdf5 files.',
    )
    parser.add_argument(
        '--output_dir', type=str, default='datasets/lapa_libero',
        help='Output dir for images and JSONL files.',
    )
    parser.add_argument(
        '--suites', type=str, nargs='+', choices=list(SUITE_SPLITS.keys()),
        help='Suites to process (default: all 5).',
    )
    parser.add_argument(
        '--discretize_bins', type=int, default=256,
        help='Bins per continuous action dimension.',
    )
    parser.add_argument(
        '--camera', type=str, default='agentview_rgb',
        choices=['agentview_rgb', 'eye_in_hand_rgb'],
        help='Camera view key to extract from each timestep.',
    )
    parser.add_argument('--seed', type=int, default=42, help='Random seed for shuffling.')
    args = parser.parse_args()
    main(args)
