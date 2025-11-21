"""Print average z-scores for JSON result files."""

import argparse
from pathlib import Path

from dmark.view.process import process_file


def _extract_scores(result: dict) -> list[float]:
    wm = result.get("watermark") or result.get("watermark_metadata") or {}
    scores = []
    for field in ("z_score_original", "z_score_truncated", "z_score_attacked", "z_score"):
        val = wm.get(field)
        if isinstance(val, (int, float)):
            scores.append(float(val))
    return scores


def _collect_scores(file_path: Path) -> list[float]:
    collected: list[float] = []

    def aggregate(instance: dict):
        collected.extend(_extract_scores(instance))

    process_file(file_path, aggregate)
    return collected


def main():
    parser = argparse.ArgumentParser(description="Print average z-score per file.")
    parser.add_argument("--input", type=Path, required=True, help="JSON file or directory of JSON files.")
    args = parser.parse_args()

    paths: list[Path]
    if args.input.is_dir():
        paths = [p for p in args.input.iterdir() if p.is_file() and p.suffix == ".json"]
    else:
        paths = [args.input]

    for path in sorted(paths):
        scores = _collect_scores(path)
        if not scores:
            print(f"{path.name}: n/a")
            continue
        avg = sum(scores) / len(scores)
        print(f"{path.name}: {avg:.3f}")


if __name__ == "__main__":
    main()
