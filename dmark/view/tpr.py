"""Compute TPR at given FPR thresholds using precomputed FPR stats."""

import argparse
import json
from pathlib import Path
from typing import Dict, List

from dmark.view.process import process_dir, process_file


def load_fpr_stats(path: Path) -> Dict[str, dict]:
    with path.open("r") as f:
        data = json.load(f)
    stats = {}
    for entry in data:
        meta = entry.get("generation_metadata")
        if meta is None:
            continue
        key = json.dumps(meta, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        stats[key] = entry
    return stats


def extract_scores(instance: dict) -> List[float]:
    wm = instance.get("watermark_detection") or {}
    scores = []
    for field in ("z_score_original", "z_score_truncated", "z_score_attacked", "z_score"):
        val = wm.get(field)
        if isinstance(val, (int, float)):
            scores.append(float(val))
    return scores


def key_from_instance(instance: dict) -> str | None:
    gen = instance.get("generation_metadata")
    wm = instance.get("watermark_metadata")
    if isinstance(gen, dict) and gen:
        key_obj = {"generation_metadata": gen}
        if isinstance(wm, dict):
            key_obj["watermark_metadata"] = wm
        return json.dumps(key_obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute TPR given FPR-derived z-score thresholds."
    )
    parser.add_argument("--fpr-json", type=Path, required=True, help="JSON produced by dmark.view.fpr.")
    parser.add_argument("--input", type=Path, required=True, help="Watermarked results (file or directory).")
    parser.add_argument(
        "--quantiles",
        type=float,
        nargs="+",
        default=[0.99, 0.999],
        help="Quantiles to evaluate (match keys in FPR JSON, e.g., 0.99 -> q0.99).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    fpr_stats = load_fpr_stats(args.fpr_json)

    def aggregate(instance: dict):
        k = key_from_instance(instance)
        if k is None:
            return None
        return k, extract_scores(instance)

    if args.input.is_dir():
        mapping = process_dir(args.input, aggregate)
    else:
        mapping = {args.input: process_file(args.input, aggregate)}

    # Collect scores grouped by generation config key
    grouped: Dict[str, List[float]] = {}
    for _, results in mapping.items():
        for pair in results:
            if not pair:
                continue
            key, scores = pair
            if key is None or not scores:
                continue
            grouped.setdefault(key, []).extend(scores)

    for key, scores in grouped.items():
        if key not in fpr_stats:
            continue  # no matching FPR baseline; skip
        entry = fpr_stats[key]
        total = len(scores)
        if total == 0:
            continue
        parts = []
        for q in args.quantiles:
            thresh_key = f"q{q}"
            if thresh_key not in entry:
                continue
            thresh = entry[thresh_key]
            tpr = sum(1 for s in scores if s >= thresh) / total
            parts.append(f"tpr@fpr{(1-q)*100:.3f}%={tpr:.3f}")
        if parts:
            print(f"{json.loads(key)}: " + ", ".join(parts))


if __name__ == "__main__":
    main()
