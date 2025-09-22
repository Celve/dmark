import argparse
import csv
import json
import os
from typing import List, Dict, Optional, Tuple
import numpy as np
from tqdm import tqdm
from datetime import datetime
from collections import Counter


def calculate_thresholds_for_fpr(z_scores: List[float], target_fprs: List[float] = [0.001, 0.005, 0.01, 0.05]) -> Dict[float, float]:
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


def normalize_model_name(model_path: str) -> str:
    """
    Normalize model name to last two path components.
    
    Args:
        model_path: Full or partial model path
    
    Returns:
        Normalized model name (e.g., 'GSAI-ML/LLaDA-8B-Instruct')
    """
    if not model_path:
        return model_path
    
    # Split by '/' and get last two parts
    parts = model_path.split('/')
    if len(parts) >= 2:
        return '/'.join(parts[-2:])
    return model_path


def extract_generation_config(results: List[Dict]) -> Dict[str, Optional[str]]:
    """
    Extract generation configuration from results.
    
    Args:
        results: List of result dictionaries
    
    Returns:
        Dictionary with generation configuration
    """
    config = {}
    
    # Find first result with metadata
    for result in results:
        if 'generation_metadata' in result:
            gen_meta = result['generation_metadata']
            # Normalize model name to last two path components
            model = gen_meta.get('model')
            config['model'] = normalize_model_name(model) if model else None
            config['dataset'] = gen_meta.get('dataset')
            config['steps'] = gen_meta.get('steps')
            config['gen_length'] = gen_meta.get('gen_length')
            config['block_length'] = gen_meta.get('block_length')
            config['temperature'] = gen_meta.get('temperature')
            config['cfg_scale'] = gen_meta.get('cfg_scale')
            config['remasking'] = gen_meta.get('remasking')
            break
    
    return config


def extract_watermark_ratio(results: List[Dict]) -> Optional[float]:
    """
    Extract watermark ratio from non-watermarked samples that have z-score calculations.
    
    Args:
        results: List of result dictionaries
    
    Returns:
        Watermark ratio if found, None otherwise
    """
    for result in results:
        # Only check non-watermarked samples with z-score calculations
        if result.get('watermark_metadata') is None and 'watermark' in result:
            wm_data = result['watermark']
            # Check if ratio is stored in watermark data
            if 'ratio' in wm_data:
                return wm_data['ratio']
    return None


def check_repetition(output_ids: List[int], repeat_ratio: float) -> Tuple[bool, Optional[int], Optional[float]]:
    """Check if any token repeats excessively in output_ids.
    
    Args:
        output_ids: List of token IDs
        repeat_ratio: Maximum allowed repetition ratio (1.0 = no limit)
    
    Returns:
        Tuple of (passes_check, most_repeated_token, max_ratio)
    """
    if not output_ids or repeat_ratio >= 1.0:
        return True, None, None
    
    token_counts = Counter(output_ids)
    max_count = max(token_counts.values())
    num_tokens = len(output_ids)
    max_ratio = max_count / num_tokens if num_tokens > 0 else 0
    
    if max_ratio > repeat_ratio:
        most_repeated = max(token_counts, key=token_counts.get)
        return False, most_repeated, max_ratio
    
    return True, None, max_ratio


def process_single_file(file_path: str, target_fprs: List[float] = [0.001, 0.005, 0.01, 0.05], z_score_type: str = 'auto', repeat_ratio: float = 1.0) -> Dict:
    """
    Process a single JSON file to calculate z-score thresholds for FPR.
    
    Args:
        file_path: Path to JSON file with z-scores from non-watermarked samples
        target_fprs: List of target false positive rates
        z_score_type: Which z-score type to use ('original', 'truncated', 'attacked', 'auto')
        repeat_ratio: Maximum allowed token repetition ratio (1.0 = no limit)
    
    Returns:
        Dictionary with analysis results for this file
    """
    with open(file_path, 'r') as f:
        results = json.load(f)
    
    # Extract configuration from the results
    config = extract_generation_config(results)
    
    # Extract watermark ratio if available
    wm_ratio = extract_watermark_ratio(results)
    
    # Collect z-scores from non-watermarked samples only
    non_watermark_scores = []
    z_score_version = None
    skipped_repetition = 0
    
    for result in results:
        # Check if this is a non-watermarked sample
        if result.get('watermark_metadata') is None:
            # Check for excessive repetition in output_ids
            output_ids = result.get('data', {}).get('output_ids', [])
            if output_ids:
                passes_check, repeated_token, max_ratio = check_repetition(output_ids, repeat_ratio)
                if not passes_check:
                    skipped_repetition += 1
                    continue
            
            # Try different locations for z_score based on type
            z_score = None
            if 'watermark' in result and result['watermark'] is not None:
                wm_data = result['watermark']
                
                if z_score_type == 'original' and 'z_score_original' in wm_data:
                    z_score = wm_data['z_score_original']
                    z_score_version = 'original'
                elif z_score_type == 'truncated' and 'z_score_truncated' in wm_data:
                    z_score = wm_data['z_score_truncated']
                    z_score_version = 'truncated'
                elif z_score_type == 'attacked' and 'z_score_attacked' in wm_data:
                    z_score = wm_data['z_score_attacked']
                    z_score_version = 'attacked'
                elif z_score_type == 'auto':
                    # Auto-detect: prioritize original, then truncated, then attacked, then legacy
                    if 'z_score_original' in wm_data:
                        z_score = wm_data['z_score_original']
                        z_score_version = 'original'
                    elif 'z_score_truncated' in wm_data:
                        z_score = wm_data['z_score_truncated']
                        z_score_version = 'truncated'
                    elif 'z_score_attacked' in wm_data:
                        z_score = wm_data['z_score_attacked']
                        z_score_version = 'attacked'
                    elif 'z_score' in wm_data:
                        z_score = wm_data['z_score']
                        z_score_version = 'legacy'
            
            # Fallback to old location if not found in watermark dict
            if z_score is None and 'z_score' in result:
                z_score = result['z_score']
                z_score_version = 'legacy'
            
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
        'config': config,
        'watermark_ratio': wm_ratio,
        'z_score_version': z_score_version,
        'thresholds': threshold_results,
        'statistics': {
            'n_samples': len(non_watermark_scores),
            'skipped_repetition': skipped_repetition,
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
        'z_scores': non_watermark_scores
    }


def generate_threshold_json(all_results: List[Dict], target_fprs: List[float], z_score_type: str, repeat_ratio: float) -> Dict:
    """
    Generate threshold JSON structure from results.
    
    Args:
        all_results: List of analysis results from each file
        target_fprs: List of target false positive rates
        z_score_type: Which z-score type was used
    
    Returns:
        Dictionary with threshold configuration structure
    """
    configurations = []
    
    for result in all_results:
        # Extract threshold values as a simple dict
        threshold_dict = {}
        for threshold_info in result['thresholds']:
            fpr_key = str(threshold_info['target_fpr'])
            threshold_dict[fpr_key] = round(threshold_info['threshold'], 4)
        
        config_entry = {
            'config': result['config'],
            'watermark_ratio': result.get('watermark_ratio'),
            'thresholds': threshold_dict,
            'statistics': {
                'mean': round(result['statistics']['mean'], 4),
                'std': round(result['statistics']['std'], 4),
                'min': round(result['statistics']['min'], 4),
                'max': round(result['statistics']['max'], 4),
                'median': round(result['statistics']['median'], 4),
                'n_samples': result['statistics']['n_samples']
            }
        }
        configurations.append(config_entry)
    
    return {
        'version': '1.0',
        'metadata': {
            'created_at': datetime.now().isoformat(),
            'z_score_type': z_score_type,
            'target_fprs': target_fprs,
            'repeat_ratio': repeat_ratio
        },
        'configurations': configurations
    }


def save_results(all_results: List[Dict], output_path: str, target_fprs: List[float], z_score_type: str, repeat_ratio: float) -> None:
    """
    Save FPR threshold analysis results to JSON and CSV files.
    
    Args:
        all_results: List of analysis results from each file
        output_path: Path to save output files (without extension)
        target_fprs: List of target false positive rates
        z_score_type: Which z-score type was used
    """
    # Generate and save threshold configuration JSON
    threshold_config = generate_threshold_json(all_results, target_fprs, z_score_type, repeat_ratio)
    config_json_path = output_path + '_threshold_config.json'
    with open(config_json_path, 'w') as f:
        json.dump(threshold_config, f, indent=2)
    print(f"\nThreshold configuration saved to: {config_json_path}")
    
    # Save full results JSON for reference
    full_json_path = output_path + '_fpr_analysis.json'
    with open(full_json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Full analysis saved to: {full_json_path}")
    
    # Save CSV with summary
    csv_path = output_path + '_fpr_thresholds.csv'
    
    # Prepare CSV headers
    headers = [
        'file', 'dataset', 'model', 'steps', 'gen_length', 'block_length', 
        'temperature', 'cfg_scale', 'batch_size', 'watermark_ratio', 'z_score_version', 'mean_zscore', 'std_zscore', 
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
            config = result['config']
            row = {
                'file': result['file'],
                'dataset': config.get('dataset'),
                'model': config.get('model'),
                'steps': config.get('steps'),
                'gen_length': config.get('gen_length'),
                'block_length': config.get('block_length'),
                'temperature': config.get('temperature'),
                'cfg_scale': config.get('cfg_scale'),
                'batch_size': config.get('batch_size'),
                'watermark_ratio': result.get('watermark_ratio'),
                'z_score_version': result.get('z_score_version', 'unknown'),
                'mean_zscore': result['statistics']['mean'],
                'std_zscore': result['statistics']['std'],
                'min_zscore': result['statistics']['min'],
                'max_zscore': result['statistics']['max'],
                'median_zscore': result['statistics']['median'],
                'total_samples': result['statistics']['n_samples']
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
        default=[0.001, 0.005, 0.01, 0.05],
        help="Target false positive rates (default: 0.001 0.005 0.01 0.05)"
    )
    
    parser.add_argument(
        "--z-score-type",
        type=str,
        default="auto",
        choices=["original", "truncated", "attacked", "auto"],
        help="Which z-score type to analyze (default: auto - uses original if available, then truncated, then attacked)"
    )
    
    parser.add_argument(
        "--repeat-ratio",
        type=float,
        default=1.0,
        help="Maximum allowed token repetition ratio, skip instances exceeding this (default: 1.0 = no filtering)"
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
                            has_zscore = False
                            if 'watermark' in item and item.get('watermark'):
                                wm = item['watermark']
                                has_zscore = any(key in wm for key in ['z_score', 'z_score_original', 'z_score_truncated', 'z_score_attacked'])
                            elif 'z_score' in item:
                                has_zscore = True
                            
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
        file_results = process_single_file(file_path, args.fpr, args.z_score_type, args.repeat_ratio)
        
        if file_results:
            all_results.append(file_results)
            
            # Print results for this file
            print(f"\n{'='*70}")
            print(f"File: {os.path.basename(file_path)}")
            print(f"Z-score version: {file_results.get('z_score_version', 'unknown')}")
            print(f"Watermark ratio: {file_results.get('watermark_ratio', 'N/A')}")
            print(f"Non-watermarked samples: {file_results['statistics']['n_samples']}")
            if file_results['statistics']['skipped_repetition'] > 0:
                print(f"Skipped due to repetition: {file_results['statistics']['skipped_repetition']}")
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
    save_results(all_results, output_path, args.fpr, args.z_score_type, args.repeat_ratio)
    
    # Print overall summary
    print("\n" + "="*70)
    print("SUMMARY ACROSS ALL FILES")
    print("="*70)
    
    # Aggregate statistics
    all_z_scores = []
    total_skipped = 0
    for result in all_results:
        all_z_scores.extend(result['z_scores'])
        total_skipped += result['statistics'].get('skipped_repetition', 0)
    
    if all_z_scores:
        print(f"\nTotal non-watermarked samples: {len(all_z_scores)}")
        if total_skipped > 0:
            print(f"Total skipped due to repetition: {total_skipped}")
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