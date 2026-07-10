import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Filter a LIBERO LAPA JSONL to one task for smoke overfitting.")
    parser.add_argument("--input_jsonl", type=Path, required=True)
    parser.add_argument("--output_jsonl", type=Path, required=True)
    parser.add_argument("--task_contains", type=str, default="", help="Substring that must appear in the image path.")
    parser.add_argument("--instruction_contains", type=str, default="", help="Substring that must appear in instruction.")
    parser.add_argument("--max_rows", type=int, default=512)
    parser.add_argument("--print_examples", type=int, default=3)
    return parser.parse_args()


def keep_item(item, args):
    if args.task_contains and args.task_contains not in item.get("image", ""):
        return False
    if args.instruction_contains and args.instruction_contains not in item.get("instruction", ""):
        return False
    return True


def main():
    args = parse_args()
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    kept = 0
    examples = []
    with args.input_jsonl.open("r", encoding="utf-8") as fin, args.output_jsonl.open("w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            total += 1
            item = json.loads(line)
            if not keep_item(item, args):
                continue
            fout.write(json.dumps(item) + "\n")
            kept += 1
            if len(examples) < args.print_examples:
                examples.append({"image": item.get("image"), "instruction": item.get("instruction")})
            if args.max_rows > 0 and kept >= args.max_rows:
                break

    print(
        json.dumps(
            {
                "input_jsonl": str(args.input_jsonl),
                "output_jsonl": str(args.output_jsonl),
                "total_scanned": total,
                "kept": kept,
                "task_contains": args.task_contains,
                "instruction_contains": args.instruction_contains,
                "examples": examples,
            },
            indent=2,
        )
    )
    if kept == 0:
        raise SystemExit("No rows matched the requested one-task filter.")


if __name__ == "__main__":
    main()
