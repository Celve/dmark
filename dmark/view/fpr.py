"""Compute z-score thresholds (FPR cutoffs) per generation config."""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional

from dmark.view.process import process_dir, process_file


def _extract_scores(instance: dict) -> list[float]:
    wm = instance.get("watermark_detection") or {}
    scores = []
    if instance.get("watermark_metadata") is None:
        for field in ("z_score_original", "z_score_truncated", "z_score_attacked", "z_score"):
            val = wm.get(field)
            if isinstance(val, (int, float)):
                scores.append(float(val))
        return scores
    else:
        return []


def _key_from_instance(instance: dict) -> Optional[str]:
    gen = instance.get("generation_metadata")
    if isinstance(gen, dict) and gen:
        # Canonical, sorted string so identical configs collapse together
        return json.dumps(gen, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return None


def percentile(data: Iterable[float], q: float) -> float:
    values = sorted(data)
    if not values:
        raise ValueError("No data for percentile calculation")
    if q <= 0:
        return values[0]
    if q >= 1:
        return values[-1]
    pos = (len(values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    frac = pos - lo
    return values[lo] * (1 - frac) + values[hi] * frac


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute z-score thresholds (FPR cutoffs) per generation config."
    )
    parser.add_argument("--input", type=Path, required=True, help="JSON file or directory.")
    parser.add_argument(
        "--quantiles",
        type=float,
        nargs="+",
        default=[0.999, 0.99],
        help="Quantiles to report (e.g., 0.999 0.99).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to dump JSON; if omitted, results are printed.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    groups: dict[str, list[float]] = defaultdict(list)

    def aggregate(instance: dict):
        # Return (key, scores) so process_* will collect them
        return _key_from_instance(instance), _extract_scores(instance)

    if args.input.is_dir():
        mapping = process_dir(args.input, aggregate)
    else:
        mapping = {args.input: process_file(args.input, aggregate)}

    for path, results in mapping.items():
        for key, scores in results:
            if key is not None and scores:
                groups[key].extend(scores)

    thresholds: dict[str, dict[str, float | int]] = {}
    for key, scores in groups.items():
        if not scores:
            continue
        entry = {"count": len(scores)}
        for q in args.quantiles:
            entry[f"q{q}"] = percentile(scores, q)
        thresholds[key] = entry

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w") as f:
            json.dump(thresholds, f, indent=2)
    else:
        for key, entry in thresholds.items():
            parts = [f"{k}={v:.3f}" if k != "count" else f"{k}={v}" for k, v in entry.items()]
            print(f"{key}: " + ", ".join(parts))


if __name__ == "__main__":
    main()
