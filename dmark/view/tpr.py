"""Compute TPR at given FPR thresholds using precomputed FPR stats."""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

from dmark.view.process import process_dir, process_file


def _gen_key(gen_meta: dict | None) -> str:
    return json.dumps(gen_meta or {}, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

def _wm_key(wm_meta: dict | None) -> str:
    return json.dumps(wm_meta or {}, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

def load_fpr_stats(path: Path) -> Dict[str, dict]:
    """Index FPR thresholds by generation metadata only (consistent with fpr.py)."""
    with path.open("r") as f:
        data = json.load(f)
    stats: Dict[str, dict] = {}
    for entry in data:
        raw_gen = entry.get("generation_metadata") or {}
        key = _gen_key(raw_gen if isinstance(raw_gen, dict) else {})
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


def instance_key(instance: dict) -> Optional[str]:
    gen = instance.get("generation_metadata")
    wm = instance.get("watermark_metadata")
    if not gen and not wm:
        return None
    return _gen_key(gen) + _wm_key(wm)


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
        k = instance_key(instance)
        if k is None:
            return None
        return (
            k,
            extract_scores(instance),
            instance.get("generation_metadata") or {},
            instance.get("watermark_metadata") or {},
        )

    if args.input.is_dir():
        mapping = process_dir(args.input, aggregate)
    else:
        mapping = {args.input: process_file(args.input, aggregate)}

    # grouped by watermark key, collecting scores and last seen generation metadata for display
    grouped_scores: Dict[str, List[float]] = {}
    grouped_gen: Dict[str, dict] = {}
    grouped_wm: Dict[str, dict] = {}

    for _, results in mapping.items():
        for record in results:
            if not record:
                continue
            k, scores, gen_meta, wm_meta = record
            if k is None or not scores:
                continue
            grouped_scores.setdefault(k, []).extend(scores)
            if isinstance(gen_meta, dict):
                grouped_gen[k] = gen_meta
            if isinstance(wm_meta, dict):
                grouped_wm[k] = wm_meta

    for key, scores in grouped_scores.items():
        gen_meta = grouped_gen.get(key, {})
        wm_meta = grouped_wm.get(key, {})
        gen_key = _gen_key(gen_meta)

        if gen_key not in fpr_stats:
            continue  # no matching FPR baseline; skip

        entry = fpr_stats[gen_key]
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
            print(f"gen={gen_meta}, wm={wm_meta}: " + ", ".join(parts))


if __name__ == "__main__":
    main()
