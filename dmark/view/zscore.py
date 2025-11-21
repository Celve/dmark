"""Print average z-scores for JSON result files."""

import argparse
from pathlib import Path

from dmark.view.process import process_dir, process_file
from dmark.view.fpr import percentile


def _extract_scores(result: dict) -> list[float]:
    wm = result.get("watermark_detection") or {}
    scores = []
    for field in ("z_score_original", "z_score_truncated", "z_score_attacked", "z_score"):
        val = wm.get(field)
        if isinstance(val, (int, float)):
            scores.append(float(val))
    return scores


def main():
    parser = argparse.ArgumentParser(description="Print average z-score per file.")
    parser.add_argument("--input", type=Path, required=True, help="JSON file or directory of JSON files.")
    parser.add_argument(
        "--quantiles",
        type=float,
        nargs="+",
        default=[0.99, 0.999],
        help="Quantiles to report (e.g., 0.99 0.999).",
    )
    args = parser.parse_args()

    def aggregate(instance: dict) -> list[float]:
        return _extract_scores(instance)

    if args.input.is_dir():
        mapping = process_dir(args.input, aggregate)
        items = sorted(mapping.items(), key=lambda x: x[0].name)
    else:
        mapping = {args.input: process_file(args.input, aggregate)}
        items = mapping.items()

    for path, chunks in items:
        # chunks is a list of lists of floats; flatten
        scores: list[float] = []
        for chunk in chunks:
            if chunk:
                scores.extend(chunk)
        if not scores:
            print(f"{path.name}: n/a")
        else:
            avg = sum(scores) / len(scores)
            parts = [f"avg={avg:.3f}"]
            for q in args.quantiles:
                thresh = percentile(scores, q)
                parts.append(f"q{q}={thresh:.3f}")
            print(f"{path.name}: " + ", ".join(parts))


if __name__ == "__main__":
    main()
