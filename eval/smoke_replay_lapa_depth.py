import argparse
import json
import time
from pathlib import Path

import numpy as np
import requests

from latent_pretraining.depth_fusion.id_mapping import resolve_lapa_sample_id


def parse_args():
    parser = argparse.ArgumentParser(description="Replay a small LAPA JSONL through the LAPA-Depth deploy server.")
    parser.add_argument("--jsonl", type=Path, required=True)
    parser.add_argument("--image_root", type=Path, default=Path("."))
    parser.add_argument("--server_url", type=str, default="http://127.0.0.1:32820/act")
    parser.add_argument("--output_json", type=Path, default=Path("outputs/smoke_lapa_depth_replay/results.json"))
    parser.add_argument("--max_rows", type=int, default=64)
    parser.add_argument("--json_id_source", type=str, default="auto", choices=("auto", "id", "image"))
    parser.add_argument("--json_id_key", type=str, default="id")
    parser.add_argument("--connect_retries", type=int, default=60)
    parser.add_argument("--retry_wait", type=float, default=10.0)
    return parser.parse_args()


def action_l1(pred, target):
    pred = np.asarray(pred, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    return float(np.mean(np.abs(pred - target)))


def token_acc(pred_tokens, target_tokens):
    pred = np.asarray(pred_tokens, dtype=np.int32)
    target = np.asarray([int(x) for x in target_tokens], dtype=np.int32)
    return float(np.mean(pred == target))


def post_with_retries(url, payload, retries, wait):
    last_err = None
    for _ in range(retries):
        try:
            resp = requests.post(url, json=payload, timeout=180)
            resp.raise_for_status()
            return resp.json()
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as exc:
            last_err = exc
            time.sleep(wait)
    raise RuntimeError(f"server unreachable at {url}: {last_err}")


def main():
    args = parse_args()
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    token_accs = []
    action_l1s = []
    with args.jsonl.open("r", encoding="utf-8") as fin:
        for line_number, line in enumerate(fin, start=1):
            if args.max_rows > 0 and len(rows) >= args.max_rows:
                break
            if not line.strip():
                continue
            item = json.loads(line)
            depth_id = resolve_lapa_sample_id(item, id_key=args.json_id_key, source=args.json_id_source)
            image_path = Path(item["image"])
            if not image_path.is_absolute():
                image_path = args.image_root / image_path
            payload = {
                "image": str(image_path),
                "instruction": item["instruction"],
                "depth_id": depth_id,
                "return_debug": True,
            }
            pred = post_with_retries(args.server_url, payload, args.connect_retries, args.retry_wait)
            pred_action = pred["action"] if isinstance(pred, dict) else pred
            pred_tokens = pred.get("action_tokens") if isinstance(pred, dict) else None
            row = {
                "line": line_number,
                "image": item["image"],
                "depth_id": depth_id,
                "target_tokens": item.get("action"),
                "pred_tokens": pred_tokens,
                "target_raw_actions": item.get("raw_actions"),
                "pred_action": pred_action,
            }
            if pred_tokens is not None and item.get("action") is not None:
                row["token_acc"] = token_acc(pred_tokens, item["action"])
                token_accs.append(row["token_acc"])
            if item.get("raw_actions") is not None:
                row["action_l1"] = action_l1(pred_action, item["raw_actions"])
                action_l1s.append(row["action_l1"])
            rows.append(row)

    result = {
        "jsonl": str(args.jsonl),
        "server_url": args.server_url,
        "n_rows": len(rows),
        "mean_token_acc": float(np.mean(token_accs)) if token_accs else None,
        "mean_action_l1": float(np.mean(action_l1s)) if action_l1s else None,
        "rows": rows,
    }
    args.output_json.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))
    print(f"Wrote detailed replay results to {args.output_json}")


if __name__ == "__main__":
    main()
