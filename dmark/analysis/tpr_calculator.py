import argparse
import csv
import json
import os
from typing import List, Dict, Optional, Tuple
import numpy as np
from tqdm import tqdm
from datetime import datetime
from collections import Counter
from dmark.analysis.threshold_loader import ThresholdLoader


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


def extract_generation_config(results: List[Dict]) -> Dict[str, any]:
    """
    Extract generation configuration from the first instance in results.
    
    Args:
        results: List of result dictionaries
    
    Returns:
        Dictionary with generation configuration
    """
    if not results:
        return {}
    
    # Get metadata from first result
    first_result = results[0]
    
    if 'generation_metadata' in first_result:
        gen_meta = first_result['generation_metadata']
        # Normalize model name to last two path components
        model = gen_meta.get('model')
        return {
            'model': normalize_model_name(model) if model else None,
            'dataset': gen_meta.get('dataset'),
            'steps': gen_meta.get('steps'),
            'gen_length': gen_meta.get('gen_length'),
            'block_length': gen_meta.get('block_length'),
            'temperature': gen_meta.get('temperature'),
            'cfg_scale': gen_meta.get('cfg_scale'),
            'remasking': gen_meta.get('remasking')
        }
    
    return {}


def extract_watermark_config(results: List[Dict]) -> Dict[str, any]:
    """
    Extract watermark configuration from the first instance in results.

    Args:
        results: List of result dictionaries

    Returns:
        Dictionary with watermark configuration
    """
    if not results:
        return {}

    # Get metadata from first result
    first_result = results[0]

    if 'watermark_metadata' in first_result and first_result['watermark_metadata']:
        wm_meta = first_result['watermark_metadata']
        return {
            'vocab_size': wm_meta.get('vocab_size'),
            'ratio': wm_meta.get('ratio'),
            'delta': wm_meta.get('delta'),
            'key': wm_meta.get('key'),
            'prebias': wm_meta.get('prebias'),
            'strategy': wm_meta.get('strategy')
        }

    return {}


def extract_attack_config(results: List[Dict]) -> Dict[str, any]:
    """
    Extract attack configuration from the first instance in results.

    Args:
        results: List of result dictionaries

    Returns:
        Dictionary with attack configuration
    """
    if not results:
        return {}

    # Get metadata from first result
    first_result = results[0]

    if 'attack_metadata' in first_result and first_result['attack_metadata']:
        attack_meta = first_result['attack_metadata']
        config = {
            'type': attack_meta.get('type'),
            'ratio': attack_meta.get('ratio'),
            'seed': attack_meta.get('seed'),
            'source_field': attack_meta.get('source_field')
        }

        # Add pre-attack truncation info if present
        if 'pre_attack_truncation' in attack_meta:
            truncation = attack_meta['pre_attack_truncation']
            config['pre_attack_truncation'] = {
                'applied': truncation.get('applied'),
                'min_output_length': truncation.get('min_output_length'),
                'original_full_length': truncation.get('original_full_length'),
                'truncated_to': truncation.get('truncated_to')
            }

        # Add paraphrase-specific fields if present
        if attack_meta.get('type') == 'paraphrase':
            config['api_provider'] = attack_meta.get('api_provider')
            config['api_model'] = attack_meta.get('api_model')
            config['temperature'] = attack_meta.get('temperature')

        # Add DIPPER-specific fields if present
        elif attack_meta.get('type') == 'dipper_paraphrase':
            # Extract from paraphrase_params
            paraphrase_params = attack_meta.get('paraphrase_params', {})
            config['lex_diversity'] = paraphrase_params.get('lex_diversity')
            config['order_diversity'] = paraphrase_params.get('order_diversity')
            config['sent_interval'] = paraphrase_params.get('sent_interval')
            config['dipper_model'] = paraphrase_params.get('dipper_model')
            config['max_length'] = paraphrase_params.get('max_length')
            config['top_p'] = paraphrase_params.get('top_p')
            config['top_k'] = paraphrase_params.get('top_k')
            config['do_sample'] = paraphrase_params.get('do_sample')

        return config

    return {}


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


def get_z_scores_from_results(results: List[Dict], z_score_type: str = 'auto', repeat_ratio: float = 1.0) -> Tuple[List[float], str, int]:
    """
    Extract z-scores from watermarked samples.
    
    Args:
        results: List of result dictionaries
        z_score_type: Which z-score type to use ('original', 'truncated', 'attacked', 'auto')
        repeat_ratio: Maximum allowed token repetition ratio (1.0 = no limit)
    
    Returns:
        Tuple of (list of z-scores, z-score version used, skipped count)
    """
    z_scores = []
    z_score_version = None
    skipped_repetition = 0
    
    for result in results:
        # Only process watermarked samples
        if result.get('watermark_metadata') is not None:
            # Check for excessive repetition in output_ids
            output_ids = result.get('data', {}).get('output_ids', [])
            if output_ids:
                passes_check, _, _ = check_repetition(output_ids, repeat_ratio)
                if not passes_check:
                    skipped_repetition += 1
                    continue
            z_score = None
            
            # Try different locations for z_score
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
                    # Auto-detect: prioritize original, then truncated, then attacked
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
            
            # Fallback to old location if not found
            if z_score is None and 'z_score' in result:
                z_score = result['z_score']
                z_score_version = 'legacy'
            
            if z_score is not None:
                z_scores.append(z_score)
    
    return z_scores, z_score_version, skipped_repetition


def calculate_tpr(z_scores: List[float], threshold: float) -> float:
    """
    Calculate true positive rate for a given threshold.
    
    Args:
        z_scores: List of z-scores from watermarked samples
        threshold: Z-score threshold
    
    Returns:
        True positive rate (fraction of samples with z-score >= threshold)
    """
    if not z_scores:
        return 0.0
    
    detected = sum(1 for z in z_scores if z >= threshold)
    return detected / len(z_scores)


def process_single_file(
    file_path: str,
    threshold_loader: ThresholdLoader,
    z_score_type: str = 'auto',
    repeat_ratio: float = 1.0,
    extract_attack: bool = False
) -> Optional[Dict]:
    """
    Process a single JSON file to calculate TPR for different FPR thresholds.
    
    Args:
        file_path: Path to JSON file with watermarked samples
        threshold_loader: ThresholdLoader instance with threshold configurations
        z_score_type: Which z-score type to use
        repeat_ratio: Maximum allowed token repetition ratio (1.0 = no limit)
    
    Returns:
        Dictionary with analysis results or None if processing failed
    """
    try:
        with open(file_path, 'r') as f:
            results = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Error reading {file_path}: {e}")
        return None
    
    if not isinstance(results, list) or not results:
        return None
    
    # Extract configurations
    gen_config = extract_generation_config(results)
    wm_config = extract_watermark_config(results)
    attack_config = extract_attack_config(results) if extract_attack else None
    
    # Skip if no watermark metadata (non-watermarked file)
    if not wm_config:
        return None
    
    # Get z-scores from watermarked samples
    z_scores, z_score_version, skipped_repetition = get_z_scores_from_results(results, z_score_type, repeat_ratio)
    
    if not z_scores:
        return None
    
    # Find matching threshold configuration
    # Include watermark ratio for accurate matching
    matching_configs = threshold_loader.find_configurations(
        model=gen_config.get('model'),
        dataset=gen_config.get('dataset'),
        steps=gen_config.get('steps'),
        gen_length=gen_config.get('gen_length'),
        block_length=gen_config.get('block_length'),
        temperature=gen_config.get('temperature'),
        cfg_scale=gen_config.get('cfg_scale'),
        remasking=gen_config.get('remasking'),
        watermark_ratio=wm_config.get('ratio')
    )
    
    if not matching_configs:
        print(f"Warning: No matching threshold configuration found for {os.path.basename(file_path)}")
        print(f"  Generation config: {gen_config}")
        print(f"  Watermark ratio: {wm_config.get('ratio')}")
        return None
    
    # Use first matching configuration
    threshold_config = matching_configs[0]
    thresholds = threshold_config['thresholds']
    
    # Calculate TPR for each FPR threshold
    tpr_results = {}
    for fpr_key, threshold in thresholds.items():
        tpr = calculate_tpr(z_scores, threshold)
        tpr_results[fpr_key] = {
            'threshold': threshold,
            'tpr': tpr,
            'detected': sum(1 for z in z_scores if z >= threshold),
            'total': len(z_scores)
        }
    
    # Calculate statistics
    statistics = {
        'n_samples': len(z_scores),
        'skipped_repetition': skipped_repetition,
        'mean': float(np.mean(z_scores)),
        'std': float(np.std(z_scores)),
        'min': float(np.min(z_scores)),
        'max': float(np.max(z_scores)),
        'median': float(np.median(z_scores)),
        'percentiles': {
            '90%': float(np.percentile(z_scores, 90)),
            '95%': float(np.percentile(z_scores, 95)),
            '99%': float(np.percentile(z_scores, 99))
        }
    }
    
    result_dict = {
        'file': os.path.basename(file_path),
        'gen_config': gen_config,
        'wm_config': wm_config,
        'z_score_version': z_score_version,
        'tpr_results': tpr_results,
        'statistics': statistics,
        'z_scores': z_scores  # Include raw scores if needed
    }

    if extract_attack and attack_config:
        result_dict['attack_config'] = attack_config

    return result_dict


def save_results(
    all_results: List[Dict],
    output_path: str,
    fprs: List[float],
    threshold_config_path: str,
    z_score_type: str,
    repeat_ratio: float,
    extract_attack: bool = False
) -> None:
    """
    Save TPR analysis results to CSV and JSON files.
    
    Args:
        all_results: List of analysis results from each file
        output_path: Path to save output files (without extension)
        fprs: List of FPR values used
        threshold_config_path: Path to the threshold configuration file used
        z_score_type: Which z-score type was used
        repeat_ratio: Maximum allowed token repetition ratio
    """
    # Save full JSON results
    json_path = output_path + '_tpr_analysis.json'
    full_results = {
        'metadata': {
            'created_at': datetime.now().isoformat(),
            'threshold_config': threshold_config_path,
            'z_score_type': z_score_type,
            'fprs': fprs,
            'repeat_ratio': repeat_ratio,
            'extract_attack': extract_attack
        },
        'results': all_results
    }
    
    with open(json_path, 'w') as f:
        json.dump(full_results, f, indent=2)
    print(f"\nFull results saved to: {json_path}")
    
    # Prepare CSV with comprehensive results
    csv_path = output_path + '_tpr_analysis.csv'
    
    # Build CSV headers
    headers = [
        'file',
        # Generation config
        'model', 'dataset', 'steps', 'gen_length', 'block_length',
        'temperature', 'cfg_scale', 'remasking',
        # Watermark config
        'wm_strategy', 'wm_ratio', 'wm_delta', 'wm_key', 'wm_prebias',
    ]

    # Add attack config headers if extracting attacks
    if extract_attack:
        headers.extend([
            'attack_type', 'attack_ratio', 'attack_seed', 'attack_source_field',
            'attack_truncation_applied', 'attack_min_output_length',
            # API-based paraphrase fields
            'attack_api_provider', 'attack_api_model', 'attack_temperature',
            # DIPPER-specific fields
            'dipper_lex_diversity', 'dipper_order_diversity', 'dipper_sent_interval',
            'dipper_model', 'dipper_max_length', 'dipper_top_p', 'dipper_top_k', 'dipper_do_sample'
        ])

    # Add statistics headers
    headers.extend([
        'z_score_version', 'n_samples', 'mean_zscore', 'std_zscore',
        'min_zscore', 'max_zscore', 'median_zscore'
    ])
    
    # Add TPR columns for each FPR
    for fpr in fprs:
        fpr_str = f"{fpr:.1%}" if fpr >= 0.01 else f"{fpr:.2%}"
        headers.extend([
            f'tpr_at_fpr_{fpr_str}',
            f'threshold_at_fpr_{fpr_str}',
            f'detected_at_fpr_{fpr_str}'
        ])
    
    # Write CSV
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        
        for result in all_results:
            gen_cfg = result['gen_config']
            wm_cfg = result['wm_config']
            
            row = {
                'file': result['file'],
                # Generation config
                'model': gen_cfg.get('model'),
                'dataset': gen_cfg.get('dataset'),
                'steps': gen_cfg.get('steps'),
                'gen_length': gen_cfg.get('gen_length'),
                'block_length': gen_cfg.get('block_length'),
                'temperature': gen_cfg.get('temperature'),
                'cfg_scale': gen_cfg.get('cfg_scale'),
                'remasking': gen_cfg.get('remasking'),
                # Watermark config
                'wm_strategy': wm_cfg.get('strategy'),
                'wm_ratio': wm_cfg.get('ratio'),
                'wm_delta': wm_cfg.get('delta'),
                'wm_key': wm_cfg.get('key'),
                'wm_prebias': wm_cfg.get('prebias'),
            }

            # Add attack config if available
            if extract_attack and 'attack_config' in result:
                attack_cfg = result['attack_config']
                row.update({
                    'attack_type': attack_cfg.get('type'),
                    'attack_ratio': attack_cfg.get('ratio'),
                    'attack_seed': attack_cfg.get('seed'),
                    'attack_source_field': attack_cfg.get('source_field'),
                    'attack_truncation_applied': attack_cfg.get('pre_attack_truncation', {}).get('applied') if attack_cfg.get('pre_attack_truncation') else None,
                    'attack_min_output_length': attack_cfg.get('pre_attack_truncation', {}).get('min_output_length') if attack_cfg.get('pre_attack_truncation') else None,
                    'attack_api_provider': attack_cfg.get('api_provider'),
                    'attack_api_model': attack_cfg.get('api_model'),
                    'attack_temperature': attack_cfg.get('temperature'),
                    # DIPPER-specific fields
                    'dipper_lex_diversity': attack_cfg.get('lex_diversity'),
                    'dipper_order_diversity': attack_cfg.get('order_diversity'),
                    'dipper_sent_interval': attack_cfg.get('sent_interval'),
                    'dipper_model': attack_cfg.get('dipper_model'),
                    'dipper_max_length': attack_cfg.get('max_length'),
                    'dipper_top_p': attack_cfg.get('top_p'),
                    'dipper_top_k': attack_cfg.get('top_k'),
                    'dipper_do_sample': attack_cfg.get('do_sample')
                })
            elif extract_attack:
                # Fill with None if no attack config found
                row.update({
                    'attack_type': None,
                    'attack_ratio': None,
                    'attack_seed': None,
                    'attack_source_field': None,
                    'attack_truncation_applied': None,
                    'attack_min_output_length': None,
                    'attack_api_provider': None,
                    'attack_api_model': None,
                    'attack_temperature': None,
                    # DIPPER-specific fields
                    'dipper_lex_diversity': None,
                    'dipper_order_diversity': None,
                    'dipper_sent_interval': None,
                    'dipper_model': None,
                    'dipper_max_length': None,
                    'dipper_top_p': None,
                    'dipper_top_k': None,
                    'dipper_do_sample': None
                })

            # Add statistics
            row.update({
                'z_score_version': result['z_score_version'],
                'n_samples': result['statistics']['n_samples'],
                'mean_zscore': f"{result['statistics']['mean']:.4f}",
                'std_zscore': f"{result['statistics']['std']:.4f}",
                'min_zscore': f"{result['statistics']['min']:.4f}",
                'max_zscore': f"{result['statistics']['max']:.4f}",
                'median_zscore': f"{result['statistics']['median']:.4f}"
            })
            
            # Add TPR data for each FPR
            for fpr_key, tpr_data in result['tpr_results'].items():
                fpr = float(fpr_key)
                fpr_str = f"{fpr:.1%}" if fpr >= 0.01 else f"{fpr:.2%}"
                row[f'tpr_at_fpr_{fpr_str}'] = f"{tpr_data['tpr']:.4f}"
                row[f'threshold_at_fpr_{fpr_str}'] = f"{tpr_data['threshold']:.4f}"
                row[f'detected_at_fpr_{fpr_str}'] = f"{tpr_data['detected']}/{tpr_data['total']}"
            
            writer.writerow(row)
    
    print(f"CSV results saved to: {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Calculate TPR (True Positive Rate) using thresholds from FPR analysis"
    )
    
    parser.add_argument(
        "--threshold-config",
        type=str,
        required=True,
        help="Path to threshold configuration JSON file (generated by fpr_calculator.py)"
    )
    
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing JSON files with watermarked samples"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results. Defaults to input directory"
    )
    
    parser.add_argument(
        "--z-score-type",
        type=str,
        default="auto",
        choices=["original", "truncated", "attacked", "auto"],
        help="Which z-score type to analyze (default: auto)"
    )
    
    parser.add_argument(
        "--file-pattern",
        type=str,
        default="*.json",
        help="File pattern to match (default: *.json)"
    )
    
    parser.add_argument(
        "--repeat-ratio",
        type=float,
        default=1.0,
        help="Maximum allowed token repetition ratio, skip instances exceeding this (default: 1.0 = no filtering)"
    )

    parser.add_argument(
        "--extract-attack",
        action="store_true",
        help="Extract and include attack configuration in the output"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.threshold_config):
        print(f"Error: Threshold configuration file not found: {args.threshold_config}")
        return
    
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory not found: {args.input_dir}")
        return
    
    # Load threshold configuration
    print(f"Loading threshold configuration from: {args.threshold_config}")
    try:
        threshold_loader = ThresholdLoader(args.threshold_config)
    except Exception as e:
        print(f"Error loading threshold configuration: {e}")
        return
    
    # Get available FPRs from configuration
    fprs = threshold_loader.get_available_fprs()
    if not fprs:
        print("Error: No FPR values found in threshold configuration")
        return
    
    print(f"Available FPRs in configuration: {fprs}")
    print(f"Available watermark ratios: {threshold_loader.get_available_watermark_ratios()}")
    
    # Find JSON files in input directory
    import glob
    pattern = os.path.join(args.input_dir, args.file_pattern)
    json_files = glob.glob(pattern)
    
    if not json_files:
        print(f"No JSON files found matching pattern: {pattern}")
        return
    
    print(f"Found {len(json_files)} JSON files to process")
    
    # Process each file
    all_results = []
    skipped_files = []
    
    for file_path in tqdm(json_files, desc="Processing files"):
        result = process_single_file(file_path, threshold_loader, args.z_score_type, args.repeat_ratio, args.extract_attack)
        
        if result:
            all_results.append(result)
        else:
            skipped_files.append(os.path.basename(file_path))
    
    if not all_results:
        print("No valid watermarked files found to process")
        if skipped_files:
            print(f"Skipped {len(skipped_files)} files (non-watermarked or no matching config)")
            print("\nNote: Make sure the threshold configuration contains thresholds for the same watermark ratio")
            print("      used in the watermarked files you're trying to analyze.")
        return
    
    # Set output directory and filename
    if args.output_dir:
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = args.input_dir
    
    # Generate output filename based on input directory name
    input_dir_name = os.path.basename(os.path.normpath(args.input_dir))
    output_path = os.path.join(output_dir, f"{input_dir_name}_tpr_analysis")
    
    # Save results
    save_results(all_results, output_path, fprs, args.threshold_config, args.z_score_type, args.repeat_ratio, args.extract_attack)
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total files processed: {len(all_results)}")
    if skipped_files:
        print(f"Files skipped: {len(skipped_files)}")
    
    # Print repetition statistics
    total_skipped = sum(r['statistics'].get('skipped_repetition', 0) for r in all_results)
    if total_skipped > 0:
        print(f"Total instances skipped due to repetition: {total_skipped}")
    
    # Calculate aggregate TPR across all files
    print("\nAggregate TPR across all files:")
    all_z_scores = []
    for result in all_results:
        all_z_scores.extend(result['z_scores'])
    
    if all_z_scores:
        print(f"Total watermarked samples: {len(all_z_scores)}")
        
        # Get one set of thresholds (they should be the same for matching configs)
        if all_results:
            sample_thresholds = all_results[0]['tpr_results']
            for fpr_key, tpr_data in sample_thresholds.items():
                threshold = tpr_data['threshold']
                aggregate_tpr = calculate_tpr(all_z_scores, threshold)
                fpr = float(fpr_key)
                print(f"  FPR {fpr*100:.2f}%: TPR = {aggregate_tpr:.4f} (threshold = {threshold:.4f})")
    
    # Group results by watermark configuration
    print("\nTPR by watermark configuration:")
    config_groups = {}
    for result in all_results:
        wm_key = (
            result['wm_config'].get('strategy'),
            result['wm_config'].get('ratio'),
            result['wm_config'].get('delta'),
            result['wm_config'].get('key')
        )
        if wm_key not in config_groups:
            config_groups[wm_key] = []
        config_groups[wm_key].append(result)

    for (strategy, ratio, delta, key), group_results in config_groups.items():
        print(f"\n  Strategy={strategy}, Ratio={ratio}, Delta={delta}, Key={key}:")
        print(f"    Files: {len(group_results)}")

        # Calculate average TPR for this configuration
        for fpr in fprs:
            fpr_key = str(fpr)
            tprs = [r['tpr_results'][fpr_key]['tpr'] for r in group_results if fpr_key in r['tpr_results']]
            if tprs:
                avg_tpr = np.mean(tprs)
                std_tpr = np.std(tprs)
                print(f"    FPR {fpr*100:.2f}%: TPR = {avg_tpr:.4f} ± {std_tpr:.4f}")

    # Group by attack configuration if extracted
    if args.extract_attack:
        print("\nTPR by attack configuration:")
        attack_groups = {}
        for result in all_results:
            if 'attack_config' in result and result['attack_config']:
                attack_cfg = result['attack_config']
                # Include DIPPER-specific parameters in the key if present
                if attack_cfg.get('type') == 'dipper_paraphrase':
                    attack_key = (
                        attack_cfg.get('type'),
                        attack_cfg.get('lex_diversity'),
                        attack_cfg.get('order_diversity'),
                        attack_cfg.get('sent_interval')
                    )
                else:
                    attack_key = (
                        attack_cfg.get('type'),
                        attack_cfg.get('ratio'),
                        attack_cfg.get('seed'),
                        None  # placeholder to keep tuple length consistent
                    )
                if attack_key not in attack_groups:
                    attack_groups[attack_key] = []
                attack_groups[attack_key].append(result)

        if attack_groups:
            for attack_key, group_results in attack_groups.items():
                if attack_key[0] == 'dipper_paraphrase':
                    attack_type, lex_div, order_div, sent_int = attack_key
                    print(f"\n  Attack Type={attack_type}, Lex={lex_div}, Order={order_div}, SentInterval={sent_int}:")
                else:
                    attack_type, attack_ratio, attack_seed, _ = attack_key
                    print(f"\n  Attack Type={attack_type}, Ratio={attack_ratio}, Seed={attack_seed}:")
                print(f"    Files: {len(group_results)}")

                # Calculate average TPR for this attack configuration
                for fpr in fprs:
                    fpr_key = str(fpr)
                    tprs = [r['tpr_results'][fpr_key]['tpr'] for r in group_results if fpr_key in r['tpr_results']]
                    if tprs:
                        avg_tpr = np.mean(tprs)
                        std_tpr = np.std(tprs)
                        print(f"    FPR {fpr*100:.2f}%: TPR = {avg_tpr:.4f} ± {std_tpr:.4f}")
        else:
            print("  No attack configurations found in results")


if __name__ == "__main__":
    main()