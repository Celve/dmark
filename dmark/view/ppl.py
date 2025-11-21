"""Print average perplexity per JSON result file."""

import argparse
from pathlib import Path

from dmark.view.process import process_dir, process_file


def _extract_ppl(result: dict) -> float | None:
    tq = result.get("text_quality") or {}
    val = tq.get("perplexity")
    if isinstance(val, (int, float)):
        return float(val)
    return None


def main():
    parser = argparse.ArgumentParser(description="Print average perplexity per file.")
    parser.add_argument("--input", type=Path, required=True, help="JSON file or directory of JSON files.")
    args = parser.parse_args()

    def aggregate(instance: dict):
        return _extract_ppl(instance)

    if args.input.is_dir():
        mapping = process_dir(args.input, aggregate)
        items = sorted(mapping.items(), key=lambda x: x[0].name)
    else:
        mapping = {args.input: process_file(args.input, aggregate)}
        items = mapping.items()

    for path, vals in items:
        scores = [v for v in vals if isinstance(v, (int, float, float))]
        if not scores:
            print(f"{path.name}: n/a")
        else:
            avg = sum(scores) / len(scores)
            print(f"{path.name}: avg_ppl={avg:.3f}, n={len(scores)}")


if __name__ == "__main__":
    main()
