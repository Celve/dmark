from typing import Any
from dmark.llada.gen import run_generation, parse_args
import os
import numpy as np
import argparse

from dmark.watermark.config import WatermarkConfig

def run_acc(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    no_watermark_config = WatermarkConfig(
        vocab_size=args.watermark_config.vocab_size,
        ratio=args.watermark_config.ratio,
        delta=args.watermark_config.delta,
        key=args.watermark_config.key,
        prebias=args.watermark_config.prebias,
        strategy=None,
        bitmap_path=args.watermark_config.bitmap_path
    )

    # we run the no-watermarked generation first
    non_watermarked_results = run_generation(
        model_path=args.model,
        device=args.device,
        dataset_name=args.dataset,
        num_samples=args.num_samples,
        steps=args.steps,
        gen_length=args.gen_length,
        block_length=args.block_length,
        temperature=args.temperature,
        cfg_scale=args.cfg_scale,
        remasking=args.remasking,
        watermark_config=no_watermark_config,
        output_dir=args.output_dir,
        enable_ppl=args.ppl,
    )

    # we then run the watermarked generation
    watermarked_results = run_generation(
        model_path=args.model,
        device=args.device,
        dataset_name=args.dataset,
        num_samples=args.num_samples,
        steps=args.steps,
        gen_length=args.gen_length,
        block_length=args.block_length,
        temperature=args.temperature,
        cfg_scale=args.cfg_scale,
        remasking=args.remasking,
        watermark_config=args.watermark_config,
        output_dir=args.output_dir,
        enable_ppl=args.ppl,
    )
    
    return non_watermarked_results, watermarked_results

def calc_acc(args: argparse.Namespace, non_watermarked_results: list[dict[str, Any]], watermarked_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Calculate accuracy metrics and z-score thresholds for specified false positive rates."""
    low_false_positive_rates = [0.001, 0.005, 0.01, 0.05]
    
    # Extract z-scores from results
    non_wm_z_scores = np.array([r["z_score"] for r in non_watermarked_results])
    wm_z_scores = np.array([r["z_score"] for r in watermarked_results])
    
    results = {}
    
    # For each target FPR, find the z-score threshold
    for target_fpr in low_false_positive_rates:
        # Sort non-watermarked z-scores in descending order
        sorted_non_wm = np.sort(non_wm_z_scores)[::-1]
        
        # Find threshold index based on target FPR
        threshold_idx = int(np.floor(len(sorted_non_wm) * target_fpr))
        print(threshold_idx, len(sorted_non_wm), target_fpr)
        if threshold_idx >= len(sorted_non_wm):
            threshold_idx = len(sorted_non_wm) - 1
        
        # Get z-score threshold that achieves this FPR
        z_threshold = sorted_non_wm[threshold_idx] if threshold_idx < len(sorted_non_wm) else sorted_non_wm[-1]
        
        # Calculate actual FPR and TPR at this threshold
        actual_fpr = np.mean(non_wm_z_scores > z_threshold)
        tpr = np.mean(wm_z_scores > z_threshold)  # True Positive Rate (detection rate)
        
        results[f"fpr_{target_fpr}"] = {
            "target_fpr": target_fpr,
            "z_threshold": float(z_threshold),
            "actual_fpr": float(actual_fpr),
            "tpr": float(tpr),
            "accuracy": float((tpr + (1 - actual_fpr)) / 2),  # Balanced accuracy
        }
    
    # Add general statistics
    results["statistics"] = {
        "non_wm_z_scores": {
            "mean": float(np.mean(non_wm_z_scores)),
            "std": float(np.std(non_wm_z_scores)),
            "min": float(np.min(non_wm_z_scores)),
            "max": float(np.max(non_wm_z_scores)),
            "median": float(np.median(non_wm_z_scores)),
        },
        "wm_z_scores": {
            "mean": float(np.mean(wm_z_scores)),
            "std": float(np.std(wm_z_scores)),
            "min": float(np.min(wm_z_scores)),
            "max": float(np.max(wm_z_scores)),
            "median": float(np.median(wm_z_scores)),
        },
        "num_samples": len(non_watermarked_results),
    }
    
    # Print results
    print("\n" + "="*60)
    print("Z-Score Thresholds for Target False Positive Rates")
    print("="*60)
    for target_fpr in low_false_positive_rates:
        info = results[f"fpr_{target_fpr}"]
        print(f"\nTarget FPR: {target_fpr:.3f}")
        print(f"  Z-Score Threshold: {info['z_threshold']:.3f}")
        print(f"  Actual FPR: {info['actual_fpr']:.3f}")
        print(f"  TPR (Detection Rate): {info['tpr']:.3f}")
        print(f"  Balanced Accuracy: {info['accuracy']:.3f}")
    
    print("\n" + "="*60)
    print("Z-Score Statistics")
    print("="*60)
    print(f"\nNon-Watermarked:")
    print(f"  Mean: {results['statistics']['non_wm_z_scores']['mean']:.3f}")
    print(f"  Std: {results['statistics']['non_wm_z_scores']['std']:.3f}")
    print(f"  Median: {results['statistics']['non_wm_z_scores']['median']:.3f}")
    print(f"  Range: [{results['statistics']['non_wm_z_scores']['min']:.3f}, {results['statistics']['non_wm_z_scores']['max']:.3f}]")
    
    print(f"\nWatermarked:")
    print(f"  Mean: {results['statistics']['wm_z_scores']['mean']:.3f}")
    print(f"  Std: {results['statistics']['wm_z_scores']['std']:.3f}")
    print(f"  Median: {results['statistics']['wm_z_scores']['median']:.3f}")
    print(f"  Range: [{results['statistics']['wm_z_scores']['min']:.3f}, {results['statistics']['wm_z_scores']['max']:.3f}]")

    # INSERT_YOUR_CODE
    # Dump results to acc.json
    if args.output_dir is not None:
        with open(f"{args.output_dir}/acc.json", "w") as f:
            import json
            json.dump(results, f, indent=4)
        print(f"Results saved to {args.output_dir}/acc.json")

    return results


def main():
    """Main entry point for accuracy evaluation."""
    args = parse_args()
    non_wm_results, wm_results = run_acc(args)
    calc_acc(args, non_wm_results, wm_results)


if __name__ == "__main__":
    main()