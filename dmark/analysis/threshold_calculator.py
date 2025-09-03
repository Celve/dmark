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


def process_single_file(file_path: str, target_tprs: List[float] = [0.90, 0.95, 0.99, 0.999]) -> Dict:
    """
    Process a single JSON file to calculate z-score thresholds.
    
    Args:
        file_path: Path to JSON file with z-scores
        target_tprs: List of target true positive rates
    
    Returns:
        Dictionary with analysis results for this file
    """
    with open(file_path, 'r') as f:
        results = json.load(f)
    
    # Collect z-scores from watermarked samples only
    watermark_scores = []
    
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
    
    if not watermark_scores:
        return None
    
    # Calculate thresholds for target TPRs
    thresholds = calculate_thresholds_for_tpr(watermark_scores, target_tprs)
    
    # Calculate actual TPR for each threshold
    threshold_results = []
    for tpr_target, threshold in thresholds.items():
        actual_tpr = sum(1 for score in watermark_scores if score >= threshold) / len(watermark_scores)
        threshold_results.append({
            'target_tpr': tpr_target,
            'threshold': threshold,
            'actual_tpr': actual_tpr,
            'samples_above': sum(1 for score in watermark_scores if score >= threshold),
            'total_samples': len(watermark_scores)
        })
    
    return {
        'file': os.path.basename(file_path),
        'thresholds': threshold_results,
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
        },
        'z_scores': watermark_scores  # Include raw scores for further analysis if needed
    }


def process_watermarked_files(input_dir: str, target_tprs: List[float] = [0.90, 0.95, 0.99, 0.999]) -> None:
    """
    Process JSON files to calculate z-score thresholds for watermarked content only.
    Each file is processed separately.
    
    Args:
        input_dir: Directory containing JSON files with z-scores
        target_tprs: List of target true positive rates
    """
    # Find all JSON files with z-scores
    json_files = [f for f in os.listdir(input_dir) if f.endswith('_zscore.json')]
    
    if not json_files:
        print(f"No *_zscore.json files found in {input_dir}")
        return
    
    print(f"Found {len(json_files)} files to process")
    
    all_results = []
    
    for json_file in tqdm(json_files, desc="Processing files"):
        file_path = os.path.join(input_dir, json_file)
        file_results = process_single_file(file_path, target_tprs)
        
        if file_results:
            all_results.append(file_results)
            
            # Print results for this file
            print(f"\n{'='*70}")
            print(f"File: {json_file}")
            print(f"Samples: {file_results['statistics']['total_samples']}")
            print(f"Mean z-score: {file_results['statistics']['mean']:.4f}")
            print(f"{'='*70}")
            
            for threshold_info in file_results['thresholds']:
                print(f"TPR {threshold_info['target_tpr']*100:.1f}%: threshold={threshold_info['threshold']:.4f}, "
                      f"actual={threshold_info['actual_tpr']*100:.1f}%, "
                      f"samples={threshold_info['samples_above']}/{threshold_info['total_samples']}")
    
    if not all_results:
        print("No watermarked samples found in any files")
        return
    
    # Save all results to a single JSON file
    output_file = os.path.join(input_dir, 'threshold_analysis_per_file.json')
    
    # Prepare summary without raw z_scores for saving
    save_results = []
    for result in all_results:
        save_result = result.copy()
        save_result.pop('z_scores', None)  # Remove raw scores from saved file
        save_results.append(save_result)
    
    with open(output_file, 'w') as f:
        json.dump({
            'per_file_results': save_results,
            'summary': {
                'total_files': len(all_results),
                'target_tprs': target_tprs
            }
        }, f, indent=4)
    
    print(f"\nResults saved to: {output_file}")
    
    # Print overall summary
    print("\n" + "="*70)
    print("SUMMARY ACROSS ALL FILES")
    print("="*70)
    
    # Aggregate statistics
    all_z_scores = []
    for result in all_results:
        all_z_scores.extend(result['z_scores'])
    
    if all_z_scores:
        print(f"\nTotal samples across all files: {len(all_z_scores)}")
        print(f"Overall mean z-score: {np.mean(all_z_scores):.4f}")
        print(f"Overall std: {np.std(all_z_scores):.4f}")
        print(f"Overall min: {np.min(all_z_scores):.4f}")
        print(f"Overall max: {np.max(all_z_scores):.4f}")
        
        # Calculate overall thresholds
        overall_thresholds = calculate_thresholds_for_tpr(all_z_scores, target_tprs)
        print("\nOverall thresholds (aggregated across all files):")
        for tpr_target, threshold in overall_thresholds.items():
            actual_tpr = sum(1 for score in all_z_scores if score >= threshold) / len(all_z_scores)
            print(f"  TPR {tpr_target*100:.1f}%: threshold={threshold:.4f}, actual={actual_tpr*100:.1f}%")


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