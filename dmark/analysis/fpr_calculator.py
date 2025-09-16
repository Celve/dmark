import argparse
import csv
import json
import os
from typing import List, Dict, Optional
import numpy as np
from tqdm import tqdm


def calculate_thresholds_for_fpr(z_scores: List[float], target_fprs: List[float] = [0.001, 0.01, 0.05, 0.10]) -> Dict[float, float]:
    """
    Calculate z-score thresholds for target false positive rates.
    
    Args:
        z_scores: List of z-scores from non-watermarked samples
        target_fprs: List of target false positive rates
    
    Returns:
        Dictionary mapping FPR to threshold
    """
    if not z_scores:
        return {fpr: float('nan') for fpr in target_fprs}
    
    # Sort z-scores in descending order for FPR calculation
    sorted_scores = sorted(z_scores, reverse=True)
    n_samples = len(sorted_scores)
    
    thresholds = {}
    for fpr in target_fprs:
        # Calculate the index for the desired FPR
        # We want the threshold where at most fpr * n_samples are >= threshold
        index = int(np.floor(fpr * n_samples))
        
        if index >= n_samples:
            threshold = sorted_scores[-1] + 1  # Above maximum
        elif index < 0:
            threshold = sorted_scores[0] + 1
        else:
            # Use index-1 to ensure we have at most fpr false positives
            if index > 0:
                threshold = sorted_scores[index - 1]
            else:
                # For very low FPR, set threshold above highest score
                threshold = sorted_scores[0] + 0.01
        
        thresholds[fpr] = threshold
    
    return thresholds


def extract_metadata_from_results(results: List[Dict]) -> Dict[str, Optional[str]]:
    """
    Extract metadata from the first non-watermarked instance in results.
    
    Args:
        results: List of result dictionaries
    
    Returns:
        Dictionary with extracted metadata
    """
    metadata = {
        'dataset': None,
        'model': None,
        'has_watermark': False,
        'steps': None,
        'gen_length': None,
        'block_length': None,
        'temperature': None,
        'cfg_scale': None,
        'device': None,
        'batch_size': None,
        'remasking': None
    }
    
    # Find first result with metadata
    for result in results:
        # Check generation metadata
        if 'generation_metadata' in result:
            gen_meta = result['generation_metadata']
            metadata['model'] = gen_meta.get('model')
            metadata['dataset'] = gen_meta.get('dataset')
            metadata['steps'] = gen_meta.get('steps')
            metadata['gen_length'] = gen_meta.get('gen_length')
            metadata['block_length'] = gen_meta.get('block_length')
            metadata['temperature'] = gen_meta.get('temperature')
            metadata['cfg_scale'] = gen_meta.get('cfg_scale')
            metadata['remasking'] = gen_meta.get('remasking')
            metadata['device'] = gen_meta.get('device')
            metadata['batch_size'] = gen_meta.get('batch_size')
            break  # Found metadata, stop looking
    
    return metadata


def process_single_file(file_path: str, target_fprs: List[float] = [0.001, 0.01, 0.05, 0.10]) -> Dict:
    """
    Process a single JSON file to calculate z-score thresholds for FPR.
    
    Args:
        file_path: Path to JSON file with z-scores from non-watermarked samples
        target_fprs: List of target false positive rates
    
    Returns:
        Dictionary with analysis results for this file
    """
    with open(file_path, 'r') as f:
        results = json.load(f)
    
    # Extract metadata from the results
    metadata = extract_metadata_from_results(results)
    
    # Collect z-scores from non-watermarked samples only
    non_watermark_scores = []
    
    for result in results:
        # Check if this is a non-watermarked sample
        if result.get('watermark_metadata') is None:
            # Try different locations for z_score
            z_score = None
            if 'watermark' in result and result['watermark'] is not None:
                z_score = result['watermark'].get('z_score')
            elif 'z_score' in result:
                z_score = result['z_score']
            
            if z_score is not None:
                non_watermark_scores.append(z_score)
    
    if not non_watermark_scores:
        return None
    
    # Calculate thresholds for target FPRs
    thresholds = calculate_thresholds_for_fpr(non_watermark_scores, target_fprs)
    
    # Calculate actual FPR for each threshold
    threshold_results = []
    for fpr_target, threshold in thresholds.items():
        actual_fpr = sum(1 for score in non_watermark_scores if score >= threshold) / len(non_watermark_scores)
        threshold_results.append({
            'target_fpr': fpr_target,
            'threshold': threshold,
            'actual_fpr': actual_fpr,
            'false_positives': sum(1 for score in non_watermark_scores if score >= threshold),
            'total_samples': len(non_watermark_scores)
        })
    
    return {
        'file': os.path.basename(file_path),
        'metadata': metadata,
        'thresholds': threshold_results,
        'statistics': {
            'total_samples': len(non_watermark_scores),
            'mean': float(np.mean(non_watermark_scores)),
            'std': float(np.std(non_watermark_scores)),
            'min': float(np.min(non_watermark_scores)),
            'max': float(np.max(non_watermark_scores)),
            'median': float(np.median(non_watermark_scores)),
            'percentiles': {
                '90%': float(np.percentile(non_watermark_scores, 90)),
                '95%': float(np.percentile(non_watermark_scores, 95)),
                '99%': float(np.percentile(non_watermark_scores, 99)),
                '99.9%': float(np.percentile(non_watermark_scores, 99.9)),
                '99.99%': float(np.percentile(non_watermark_scores, 99.99))
            }
        },
        'z_scores': non_watermark_scores  # Include raw scores for further analysis if needed
    }


def save_results(all_results: List[Dict], output_path: str, target_fprs: List[float]) -> None:
    """
    Save FPR threshold analysis results to JSON and CSV files.
    
    Args:
        all_results: List of analysis results from each file
        output_path: Path to save output files (without extension)
        target_fprs: List of target false positive rates
    """
    # Save JSON with full results
    json_path = output_path + '_fpr_thresholds.json'
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nJSON results saved to: {json_path}")
    
    # Save CSV with summary
    csv_path = output_path + '_fpr_thresholds.csv'
    
    # Prepare CSV headers
    headers = [
        'file', 'dataset', 'model', 'steps', 'gen_length', 'block_length', 
        'temperature', 'cfg_scale', 'batch_size', 'mean_zscore', 'std_zscore', 
        'min_zscore', 'max_zscore', 'median_zscore', 'total_samples'
    ]
    
    # Add threshold columns for each FPR
    for fpr in target_fprs:
        headers.append(f'threshold_fpr_{fpr:.2%}')
        headers.append(f'actual_fpr_{fpr:.2%}')
    
    # Write CSV
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        
        for result in all_results:
            meta = result['metadata']
            row = {
                'file': result['file'],
                'dataset': meta.get('dataset'),
                'model': meta.get('model'),
                'steps': meta.get('steps'),
                'gen_length': meta.get('gen_length'),
                'block_length': meta.get('block_length'),
                'temperature': meta.get('temperature'),
                'cfg_scale': meta.get('cfg_scale'),
                'batch_size': meta.get('batch_size'),
                'mean_zscore': result['statistics']['mean'],
                'std_zscore': result['statistics']['std'],
                'min_zscore': result['statistics']['min'],
                'max_zscore': result['statistics']['max'],
                'median_zscore': result['statistics']['median'],
                'total_samples': result['statistics']['total_samples']
            }
            
            # Add threshold data for each FPR
            for threshold_info in result['thresholds']:
                fpr = threshold_info['target_fpr']
                row[f'threshold_fpr_{fpr:.2%}'] = threshold_info['threshold']
                row[f'actual_fpr_{fpr:.2%}'] = threshold_info['actual_fpr']
            
            writer.writerow(row)
    
    print(f"CSV summary saved to: {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Calculate z-score thresholds from non-watermarked content to achieve target false positive rates"
    )
    
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to JSON file or directory containing JSON files with non-watermarked z_score data"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path prefix for results (without extension). Defaults to input path"
    )
    
    parser.add_argument(
        "--fpr",
        type=float,
        nargs='+',
        default=[0.001, 0.01, 0.05, 0.10],
        help="Target false positive rates (default: 0.001 0.01 0.05 0.10)"
    )
    
    args = parser.parse_args()
    
    # Check if input exists
    if not os.path.exists(args.input):
        print(f"Error: Input path '{args.input}' not found")
        return
    
    # Determine if input is file or directory
    if os.path.isfile(args.input):
        json_files = [args.input]
        default_output = os.path.splitext(args.input)[0]
    else:
        # Find all JSON files in directory
        json_files = [os.path.join(args.input, f) for f in os.listdir(args.input) 
                     if f.endswith('.json')]
        default_output = os.path.join(args.input, 'fpr_analysis')
    
    if not json_files:
        print(f"No JSON files found")
        return
    
    # Set output path
    output_path = args.output if args.output else default_output
    
    # Filter files that contain z_score data from non-watermarked samples
    valid_files = []
    for file_path in json_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                # Check if it's an array and has non-watermarked samples with z_scores
                if isinstance(data, list) and len(data) > 0:
                    # Check for at least one non-watermarked sample with z_score
                    for item in data:
                        if item.get('watermark_metadata') is None:
                            has_zscore = (
                                'z_score' in item or
                                ('watermark' in item and item.get('watermark') and 'z_score' in item['watermark'])
                            )
                            if has_zscore:
                                valid_files.append(file_path)
                                break
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
    
    if not valid_files:
        print(f"No JSON files with non-watermarked z_score data found")
        return
    
    print(f"Found {len(valid_files)} files with non-watermarked z_score data to process")
    
    all_results = []
    
    for file_path in tqdm(valid_files, desc="Processing files"):
        file_results = process_single_file(file_path, args.fpr)
        
        if file_results:
            all_results.append(file_results)
            
            # Print results for this file
            print(f"\n{'='*70}")
            print(f"File: {os.path.basename(file_path)}")
            print(f"Non-watermarked samples: {file_results['statistics']['total_samples']}")
            print(f"Mean z-score: {file_results['statistics']['mean']:.4f}")
            print(f"Std z-score: {file_results['statistics']['std']:.4f}")
            print(f"{'='*70}")
            
            print("\nFPR Thresholds:")
            for threshold_info in file_results['thresholds']:
                print(f"  FPR {threshold_info['target_fpr']*100:.2f}%: threshold={threshold_info['threshold']:.4f}, "
                      f"actual={threshold_info['actual_fpr']*100:.2f}%, "
                      f"false_positives={threshold_info['false_positives']}/{threshold_info['total_samples']}")
            
            print("\nPercentiles (for reference):")
            for p, val in file_results['statistics']['percentiles'].items():
                print(f"  {p}: {val:.4f}")
    
    if not all_results:
        print("No non-watermarked samples found in any files")
        return
    
    # Save results
    save_results(all_results, output_path, args.fpr)
    
    # Print overall summary
    print("\n" + "="*70)
    print("SUMMARY ACROSS ALL FILES")
    print("="*70)
    
    # Aggregate statistics
    all_z_scores = []
    for result in all_results:
        all_z_scores.extend(result['z_scores'])
    
    if all_z_scores:
        print(f"\nTotal non-watermarked samples: {len(all_z_scores)}")
        print(f"Overall mean z-score: {np.mean(all_z_scores):.4f}")
        print(f"Overall std: {np.std(all_z_scores):.4f}")
        print(f"Overall min: {np.min(all_z_scores):.4f}")
        print(f"Overall max: {np.max(all_z_scores):.4f}")
        
        # Calculate overall thresholds
        overall_thresholds = calculate_thresholds_for_fpr(all_z_scores, args.fpr)
        print("\nOverall FPR thresholds (aggregated across all files):")
        for fpr_target, threshold in overall_thresholds.items():
            actual_fpr = sum(1 for score in all_z_scores if score >= threshold) / len(all_z_scores)
            print(f"  FPR {fpr_target*100:.2f}%: threshold={threshold:.4f}, actual={actual_fpr*100:.2f}%")
        
        print("\nKey thresholds for common FPR targets:")
        print(f"  1% FPR: z-score >= {overall_thresholds.get(0.01, 'N/A'):.4f}")
        print(f"  0.1% FPR: z-score >= {overall_thresholds.get(0.001, 'N/A'):.4f}")


if __name__ == "__main__":
    main()