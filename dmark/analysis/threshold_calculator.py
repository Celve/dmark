import argparse
import json
import os
from typing import List, Dict
import numpy as np
from tqdm import tqdm


def calculate_thresholds_for_tpr(z_scores: List[float], target_tprs: List[float] = [0.90, 0.95, 0.99, 0.999]) -> Dict[float, float]:
    """
    Calculate z-score thresholds for target true positive rates.
    
    Args:
        z_scores: List of z-scores from watermarked samples
        target_tprs: List of target true positive rates
    
    Returns:
        Dictionary mapping TPR to threshold
    """
    if not z_scores:
        return {tpr: float('nan') for tpr in target_tprs}
    
    # Sort z-scores in ascending order
    sorted_scores = sorted(z_scores)
    n_samples = len(sorted_scores)
    
    thresholds = {}
    for tpr in target_tprs:
        # Calculate the index for the desired TPR
        # We want the threshold where at least tpr * n_samples are >= threshold
        index = int(np.floor((1 - tpr) * n_samples))
        
        if index >= n_samples:
            threshold = sorted_scores[-1] - 1  # Below minimum
        elif index < 0:
            threshold = sorted_scores[0]
        else:
            threshold = sorted_scores[index]
        
        thresholds[tpr] = threshold
    
    return thresholds


def process_watermarked_files(input_dir: str, target_tprs: List[float] = [0.90, 0.95, 0.99, 0.999]) -> None:
    """
    Process JSON files to calculate z-score thresholds for watermarked content only.
    
    Args:
        input_dir: Directory containing JSON files with z-scores
        target_tprs: List of target true positive rates
    """
    # Find all JSON files with z-scores
    json_files = [f for f in os.listdir(input_dir) if f.endswith('_zscore.json')]
    
    if not json_files:
        print(f"No *_zscore.json files found in {input_dir}")
        return
    
    # Collect z-scores from watermarked samples only
    watermark_scores = []
    
    print(f"Processing {len(json_files)} files...")
    
    for json_file in tqdm(json_files, desc="Loading watermarked samples"):
        file_path = os.path.join(input_dir, json_file)
        
        with open(file_path, 'r') as f:
            results = json.load(f)
        
        for result in results:
            # Check if this is a watermarked sample
            if result.get('watermark_metadata') is not None:
                # Try different locations for z_score
                z_score = None
                if 'watermark' in result and result['watermark'] is not None:
                    z_score = result['watermark'].get('z_score')
                elif 'z_score' in result:
                    z_score = result['z_score']
                
                if z_score is not None:
                    watermark_scores.append(z_score)
    
    print(f"\nFound {len(watermark_scores)} watermarked samples with z-scores")
    
    if not watermark_scores:
        print("Error: No watermarked samples found with z-scores")
        return
    
    # Calculate thresholds for target TPRs
    thresholds = calculate_thresholds_for_tpr(watermark_scores, target_tprs)
    
    # Calculate actual TPR for each threshold and display results
    results = []
    print("\n" + "="*70)
    print("Z-Score Threshold Analysis for Watermarked Content")
    print("="*70)
    
    for tpr_target, threshold in thresholds.items():
        # Calculate actual TPR (percentage of watermarked samples >= threshold)
        actual_tpr = sum(1 for score in watermark_scores if score >= threshold) / len(watermark_scores)
        
        results.append({
            'target_tpr': tpr_target,
            'threshold': threshold,
            'actual_tpr': actual_tpr,
            'total_samples': len(watermark_scores)
        })
        
        print(f"\nTarget TPR: {tpr_target*100:.1f}%")
        print(f"  Z-score threshold: {threshold:.4f}")
        print(f"  Actual TPR achieved: {actual_tpr*100:.2f}%")
        print(f"  Samples above threshold: {int(actual_tpr * len(watermark_scores))}/{len(watermark_scores)}")
    
    # Save results to file
    output_file = os.path.join(input_dir, 'watermark_threshold_analysis.json')
    with open(output_file, 'w') as f:
        json.dump({
            'thresholds': results,
            'statistics': {
                'total_samples': len(watermark_scores),
                'mean': float(np.mean(watermark_scores)),
                'std': float(np.std(watermark_scores)),
                'min': float(np.min(watermark_scores)),
                'max': float(np.max(watermark_scores)),
                'median': float(np.median(watermark_scores)),
                'percentiles': {
                    '1%': float(np.percentile(watermark_scores, 1)),
                    '5%': float(np.percentile(watermark_scores, 5)),
                    '10%': float(np.percentile(watermark_scores, 10)),
                    '25%': float(np.percentile(watermark_scores, 25)),
                    '50%': float(np.percentile(watermark_scores, 50)),
                    '75%': float(np.percentile(watermark_scores, 75)),
                    '90%': float(np.percentile(watermark_scores, 90)),
                    '95%': float(np.percentile(watermark_scores, 95)),
                    '99%': float(np.percentile(watermark_scores, 99)),
                    '99.9%': float(np.percentile(watermark_scores, 99.9))
                }
            }
        }, f, indent=4)
    
    print(f"\nResults saved to: {output_file}")
    
    # Print detailed statistics
    print("\n" + "="*70)
    print("Z-Score Statistics for Watermarked Samples")
    print("="*70)
    
    print(f"\nSample size: {len(watermark_scores)}")
    print(f"Mean: {np.mean(watermark_scores):.4f}")
    print(f"Std: {np.std(watermark_scores):.4f}")
    print(f"Min: {np.min(watermark_scores):.4f}")
    print(f"Max: {np.max(watermark_scores):.4f}")
    print(f"Median: {np.median(watermark_scores):.4f}")
    
    print("\nPercentiles:")
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9]
    for p in percentiles:
        value = np.percentile(watermark_scores, p)
        print(f"  {p:5.1f}%: {value:8.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Calculate z-score thresholds for watermarked content to achieve target true positive rates"
    )
    
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Directory containing *_zscore.json files"
    )
    
    parser.add_argument(
        "--tpr",
        type=float,
        nargs='+',
        default=[0.90, 0.95, 0.99, 0.999],
        help="Target true positive rates (default: 0.90 0.95 0.99 0.999)"
    )
    
    args = parser.parse_args()
    
    # Check if input directory exists
    if not os.path.exists(args.input):
        print(f"Error: Input directory '{args.input}' not found")
        return
    
    if not os.path.isdir(args.input):
        print(f"Error: '{args.input}' is not a directory")
        return
    
    # Process files
    process_watermarked_files(args.input, args.tpr)


if __name__ == "__main__":
    main()