"""
Convert LIBERO HDF5 demonstration files to LAPA-compatible JSONL format.

Each suite is processed as a whole (no train/test split) and gets its own
action-bin table, computed the same way finetune_preprocess.py does it:
qcut over ALL records of that suite, with a binary gripper override.

Output layout:
  {output_dir}/images/{suite}/{task_stem}/demo_{i}/step_{t}.jpg
  {output_dir}/{suite}.jsonl
  {output_dir}/action_bins_{suite}.csv

The number of columns of each action_bins_{suite}.csv is the value to use as
--llama.action_vocab_size when fine-tuning on that suite.

Usage:
  python data/process_libero.py \
      --libero_root datasets/libero_raw \
      --output_dir  datasets/lapa_libero
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from PIL import Image

SUITES = ['libero_goal', 'libero_object', 'libero_spatial', 'libero_90', 'libero_10']


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
        records (list): tuples of (instruction, raw_action, img_rel_path).
    """
    task_stem = Path(hdf5_path).stem
    records = []

    with h5py.File(hdf5_path, 'r') as f:
        problem_info = json.loads(f['data'].attrs['problem_info'])
        task = problem_info['language_instruction']
        # Raw conversation-style instruction, same shape finetune_preprocess.py
        # expects in conversations[0].value before it strips '<image>\n'.
        instruction = f'<image>\nWhat action should the robot take to `{task}`'
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
                # Do NOT flip: save frames exactly as stored in the HDF5.
                # Always overwrite so a re-run replaces images from older versions.
                Image.fromarray(images_arr[t]).save(img_abs)

                records.append((instruction, actions[t].tolist(), img_rel))

    return records


def compute_suite_bins(records, discretize_bins):
    """Per-dim qcut over all records of one suite (finetune_preprocess.py style)."""
    total_list = [[] for _ in range(7)]
    for (_inst, raw_action, _img) in records:
        for dim in range(7):
            total_list[dim].append(raw_action[dim])

    total_bin = []
    for dim, individual_list in enumerate(total_list):
        values = np.asarray(individual_list, dtype=np.float64)
        # Bin the raw action values as-is (nothing added / no jitter). Since the
        # raw values are already coarsely discretized, qcut with duplicates='drop'
        # may return fewer than discretize_bins bins for some dims — that's fine.
        _, bins = pd.qcut(
            pd.Series(values), discretize_bins,
            labels=False, retbins=True, duplicates='drop',
        )
        total_bin.append(bins)
        print(f"  dim {dim}: {len(bins)-1} bins  [{bins[0]:.4f}, {bins[-1]:.4f}]")
        print(f"    edges: {np.array2string(np.asarray(bins), precision=8, threshold=300, max_line_width=120)}")

    # Gripper override: LIBERO gripper is in {-1, 1}; treat as binary open/close.
    # (finetune_preprocess.py uses int(action[6]), which would yield -1 here,
    # so re-bin instead: bin 0 -> open (-1), bin 1 -> close (+1).)
    total_bin[6] = np.array([-1.5, 0.0, 1.5])
    return total_bin


def make_record(instruction, raw_action, img_rel, total_bin):
    """Build one jsonl record, mirroring finetune_preprocess.py's output fields."""
    action_bins = [str(assign_bin(raw_action[i], total_bin[i])) for i in range(7)]
    instruction = instruction.replace('<image>\n', '')
    return {
        'instruction': f'<s> You are a helpful assistant. USER: {instruction} ASSISTANT:',
        'image': img_rel,
        'raw_actions': raw_action,
        'action': action_bins,
        'fields': '[instruction],[vision],action',
    }


def main(args):
    libero_root = Path(args.libero_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    suites = args.suites if args.suites else SUITES
    summary = []

    for suite in suites:
        suite_dir = libero_root / suite
        if not suite_dir.exists():
            print(f"[warn] skipping missing suite dir: {suite_dir}")
            continue
        hdf5_files = sorted(suite_dir.glob('*.hdf5'))
        print(f"\n[{suite}] {len(hdf5_files)} task files")

        # ── Step 1: extract images and collect raw records ────────────────────
        records = []
        for hdf5_path in hdf5_files:
            print(f"  {hdf5_path.name} ...", flush=True)
            records.extend(process_hdf5(hdf5_path, suite, output_dir, args.camera))

        # ── Step 2: per-suite action bins over ALL records of this suite ──────
        print(f"[{suite}] computing action bins from {len(records):,} records ...")
        total_bin = compute_suite_bins(records, args.discretize_bins)

        bins_path = output_dir / f'action_bins_{suite}.csv'
        pd.DataFrame(total_bin).to_csv(bins_path, index=False)
        vocab = max(len(b) for b in total_bin)
        print(f"[{suite}] saved {bins_path.name} (action_vocab_size = {vocab})")

        # ── Step 3: write one jsonl for the whole suite ────────────────────────
        jsonl_path = output_dir / f'{suite}.jsonl'
        with open(jsonl_path, 'w') as fout:
            for (instruction, raw_action, img_rel) in records:
                rec = make_record(instruction, raw_action, img_rel, total_bin)
                fout.write(json.dumps(rec) + '\n')
        print(f"[{suite}] wrote {len(records):>9,} records -> {jsonl_path.name}")

        summary.append((suite, len(records), vocab))

    print("\n=== summary ===")
    for suite, n, vocab in summary:
        print(f"  {suite:<16} records={n:>9,}  action_vocab_size={vocab}")
    print("Done.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Convert LIBERO HDF5 demos to LAPA JSONL format (per-suite, no split).'
    )
    parser.add_argument(
        '--libero_root', type=str, default='datasets/libero_raw',
        help='Root dir containing suite subdirs with .hdf5 files.',
    )
    parser.add_argument(
        '--output_dir', type=str, default='datasets/lapa_libero',
        help='Output dir for images, per-suite jsonl and per-suite bins CSV.',
    )
    parser.add_argument(
        '--suites', type=str, nargs='+', choices=SUITES,
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
    args = parser.parse_args()
    main(args)
