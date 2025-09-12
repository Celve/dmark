import argparse
import csv
import json
import os
from typing import List, Dict, Optional
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


def extract_metadata_from_results(results: List[Dict]) -> Dict[str, Optional[str]]:
    """
    Extract metadata from the first watermarked instance in results.
    
    Args:
        results: List of result dictionaries
    
    Returns:
        Dictionary with extracted metadata
    """
    metadata = {
        'dataset': None,
        'model': None,
        'has_watermark': False,
        'ratio': None,
        'delta': None,
        'key': None,
        'strategy': None,
        'prebias': None,
        'remasking': None,
        'steps': None,
        'gen_length': None,
        'block_length': None,
        'temperature': None,
        'cfg_scale': None,
        'device': None,
        'batch_size': None
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
        
        # Check watermark metadata
        if 'watermark_metadata' in result and result['watermark_metadata'] is not None:
            metadata['has_watermark'] = True
            wm_meta = result['watermark_metadata']
            metadata['strategy'] = wm_meta.get('strategy')
            metadata['ratio'] = wm_meta.get('ratio')
            metadata['delta'] = wm_meta.get('delta')
            metadata['key'] = wm_meta.get('key')
            metadata['prebias'] = wm_meta.get('prebias')
            metadata['vocab_size'] = wm_meta.get('vocab_size')
            break  # Found metadata, stop looking
    
    return metadata


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
    
    # Extract metadata from the results
    metadata = extract_metadata_from_results(results)
    
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
        'metadata': metadata,
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


def save_csv_results(all_results: List[Dict], input_dir: str, output_dir: str, target_tprs: List[float]) -> None:
    """
    Save threshold analysis results to CSV files.
    
    Args:
        all_results: List of analysis results from each file
        input_dir: Directory containing input files (used for naming)
        output_dir: Directory to save CSV files
        target_tprs: List of target true positive rates
    """
    # Use input directory name as base for CSV filenames
    dir_name = os.path.basename(os.path.normpath(input_dir))
    
    # Per-file CSV with thresholds
    csv_path = os.path.join(output_dir, f'{dir_name}_threshold_analysis_per_file.csv')
    
    # Prepare CSV headers
    headers = [
        'file', 'dataset', 'model', 'strategy', 'ratio', 'delta', 'key', 'prebias',
        'vocab_size', 'remasking', 'steps', 'gen_length', 'block_length', 'temperature',
        'cfg_scale', 'batch_size', 'mean_zscore', 'std_zscore', 'min_zscore', 'max_zscore',
        'median_zscore', 'total_samples'
    ]
    
    # Add threshold columns for each TPR
    for tpr in target_tprs:
        headers.append(f'threshold_tpr_{tpr:.1%}')
        headers.append(f'actual_tpr_{tpr:.1%}')
    
    # Write per-file CSV
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        
        for result in all_results:
            meta = result['metadata']
            row = {
                'file': result['file'],
                'dataset': meta.get('dataset'),
                'model': meta.get('model'),
                'strategy': meta.get('strategy'),
                'ratio': meta.get('ratio'),
                'delta': meta.get('delta'),
                'key': meta.get('key'),
                'prebias': meta.get('prebias'),
                'vocab_size': meta.get('vocab_size'),
                'remasking': meta.get('remasking'),
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
            
            # Add threshold data for each TPR
            for threshold_info in result['thresholds']:
                tpr = threshold_info['target_tpr']
                row[f'threshold_tpr_{tpr:.1%}'] = threshold_info['threshold']
                row[f'actual_tpr_{tpr:.1%}'] = threshold_info['actual_tpr']
            
            writer.writerow(row)
    
    print(f"\nPer-file CSV saved to: {csv_path}")
    
    # Summary CSV with aggregated thresholds by configuration
    summary_csv_path = os.path.join(output_dir, f'{dir_name}_threshold_summary_by_config.csv')
    
    # Group results by configuration
    config_groups = {}
    for result in all_results:
        meta = result['metadata']
        # Create a configuration key
        config_key = (
            meta.get('strategy'),
            meta.get('ratio'),
            meta.get('delta'),
            meta.get('key'),
            meta.get('prebias'),
            meta.get('remasking'),
            meta.get('model'),
            meta.get('dataset')
        )
        
        if config_key not in config_groups:
            config_groups[config_key] = []
        config_groups[config_key].append(result)
    
    # Write summary CSV
    summary_headers = [
        'strategy', 'ratio', 'delta', 'key', 'prebias', 'remasking',
        'model', 'dataset', 'num_files', 'total_samples', 'mean_zscore', 'std_zscore'
    ]
    
    for tpr in target_tprs:
        summary_headers.append(f'threshold_tpr_{tpr:.1%}')
        summary_headers.append(f'actual_tpr_{tpr:.1%}')
    
    with open(summary_csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=summary_headers)
        writer.writeheader()
        
        for config_key, group_results in config_groups.items():
            # Aggregate z-scores for this configuration
            all_scores = []
            for result in group_results:
                all_scores.extend(result['z_scores'])
            
            if all_scores:
                # Calculate aggregated thresholds
                agg_thresholds = calculate_thresholds_for_tpr(all_scores, target_tprs)
                
                row = {
                    'strategy': config_key[0],
                    'ratio': config_key[1],
                    'delta': config_key[2],
                    'key': config_key[3],
                    'prebias': config_key[4],
                    'remasking': config_key[5],
                    'model': config_key[6],
                    'dataset': config_key[7],
                    'num_files': len(group_results),
                    'total_samples': len(all_scores),
                    'mean_zscore': np.mean(all_scores),
                    'std_zscore': np.std(all_scores)
                }
                
                # Add threshold data
                for tpr, threshold in agg_thresholds.items():
                    actual_tpr = sum(1 for score in all_scores if score >= threshold) / len(all_scores)
                    row[f'threshold_tpr_{tpr:.1%}'] = threshold
                    row[f'actual_tpr_{tpr:.1%}'] = actual_tpr
                
                writer.writerow(row)
    
    print(f"Summary CSV saved to: {summary_csv_path}")


def process_watermarked_files(input_dir: str, output_dir: Optional[str] = None, target_tprs: List[float] = [0.90, 0.95, 0.99, 0.999]) -> None:
    """
    Process JSON files to calculate z-score thresholds for watermarked content only.
    Each file is processed separately.
    
    Args:
        input_dir: Directory containing JSON files with z-scores
        output_dir: Directory to save output CSV files (defaults to input_dir)
        target_tprs: List of target true positive rates
    """
    # Default output directory to input directory if not specified
    if output_dir is None:
        output_dir = input_dir
    else:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
    # Find all JSON files
    json_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
    
    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return
    
    # Filter files that contain z_score data
    valid_files = []
    for json_file in json_files:
        file_path = os.path.join(input_dir, json_file)
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                # Check if it's an array and first element has z_score
                if isinstance(data, list) and len(data) > 0:
                    first_item = data[0]
                    # Check for z_score in various possible locations
                    has_zscore = (
                        'z_score' in first_item or
                        ('watermark' in first_item and first_item.get('watermark') and 'z_score' in first_item['watermark'])
                    )
                    if has_zscore:
                        valid_files.append(json_file)
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
    
    if not valid_files:
        print(f"No JSON files with z_score data found in {input_dir}")
        return
    
    print(f"Found {len(valid_files)} files with z_score data to process")
    json_files = valid_files
    
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
    
    # Save CSV results only
    save_csv_results(all_results, input_dir, output_dir, target_tprs)
    
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
        
        # Save overall thresholds to a simple CSV for quick reference
        dir_name = os.path.basename(os.path.normpath(input_dir))
        quick_ref_csv = os.path.join(output_dir, f'{dir_name}_threshold_quick_reference.csv')
        with open(quick_ref_csv, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Target TPR (%)', 'Z-Score Threshold', 'Actual TPR (%)', 'Total Samples'])
            for tpr_target, threshold in overall_thresholds.items():
                actual_tpr = sum(1 for score in all_z_scores if score >= threshold) / len(all_z_scores)
                writer.writerow([f'{tpr_target*100:.1f}', f'{threshold:.4f}', f'{actual_tpr*100:.1f}', len(all_z_scores)])
        print(f"\nQuick reference CSV saved to: {quick_ref_csv}")


def main():
    parser = argparse.ArgumentParser(
        description="Calculate z-score thresholds for watermarked content to achieve target true positive rates"
    )
    
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Directory containing JSON files with z_score data"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Directory to save output CSV files (defaults to input directory)"
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
    process_watermarked_files(args.input, args.output, args.tpr)


if __name__ == "__main__":
    main()