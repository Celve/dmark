#!/usr/bin/env python3
"""
Merge two CSV files with different headers but matching generation and watermark configurations.
One CSV contains PPL (perplexity) metrics, the other contains z-score and TPR metrics.
"""

import argparse
import csv
import os
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


def normalize_dataset_name(dataset_path: str) -> str:
    """
    Normalize dataset name to last two path components.
    Special case: 'gsm8k' becomes 'openai/gsm8k'.

    Args:
        dataset_path: Full or partial dataset path

    Returns:
        Normalized dataset name (e.g., 'sentence-transformers/eli5', 'openai/gsm8k')
    """
    if not dataset_path:
        return dataset_path

    # Special case for gsm8k
    if dataset_path == 'gsm8k':
        return 'openai/gsm8k'

    # Split by '/' and get last two parts
    parts = dataset_path.split('/')
    if len(parts) >= 2:
        normalized = '/'.join(parts[-2:])
        # Check if result is gsm8k after normalization
        if normalized == 'gsm8k':
            return 'openai/gsm8k'
        return normalized

    # If only one part and it's gsm8k
    if parts[0] == 'gsm8k':
        return 'openai/gsm8k'

    return dataset_path


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


def normalize_key_names(row: Dict[str, str], key_mapping: Dict[str, str]) -> Dict[str, str]:
    """
    Normalize key names in a row according to the mapping.

    Args:
        row: Dictionary representing a CSV row
        key_mapping: Mapping from old keys to new keys

    Returns:
        Row with normalized keys
    """
    normalized = {}
    for old_key, value in row.items():
        new_key = key_mapping.get(old_key, old_key)
        normalized[new_key] = value
    return normalized


def create_config_key(row: Dict[str, str], config_fields: List[str]) -> Tuple:
    """
    Create a hashable key from configuration fields.

    Args:
        row: Dictionary representing a CSV row
        config_fields: List of field names to include in the key

    Returns:
        Tuple of configuration values
    """
    return tuple(row.get(field, '') for field in config_fields)


def extract_filename_base(filename: str) -> str:
    """
    Extract base filename without _zscore, _ppl, or other suffixes.

    Args:
        filename: Full filename

    Returns:
        Base filename
    """
    # Remove common suffixes
    for suffix in ['_zscore', '_ppl', '_attack', '_truncated']:
        if suffix in filename:
            filename = filename.replace(suffix, '')

    # Remove .json extension if present
    if filename.endswith('.json'):
        filename = filename[:-5]

    return filename


def merge_csv_files(
    ppl_file: str,
    zscore_file: str,
    output_file: str,
    match_by_file: bool = True,
    verbose: bool = False
) -> None:
    """
    Merge two CSV files based on matching configurations.

    Args:
        ppl_file: Path to CSV file with PPL metrics
        zscore_file: Path to CSV file with z-score metrics
        output_file: Path to output merged CSV
        match_by_file: Whether to match by filename in addition to configs
        verbose: Print detailed matching information
    """

    # Define key mappings for normalization
    ppl_key_mapping = {
        'strategy': 'wm_strategy',
        'ratio': 'wm_ratio',
        'delta': 'wm_delta',
        'key': 'wm_key',
        'prebias': 'wm_prebias',
        'vocab_size': 'wm_vocab_size',
        'has_watermark': 'wm_enabled'
    }

    zscore_key_mapping = {
        # Already uses wm_ prefix for watermark fields
    }

    # Configuration fields for matching (in order of importance)
    gen_config_fields = [
        'dataset', 'model', 'steps', 'gen_length',
        'block_length', 'temperature', 'cfg_scale', 'remasking'
    ]

    wm_config_fields = [
        'wm_strategy', 'wm_ratio', 'wm_delta', 'wm_key', 'wm_prebias'
    ]

    # Read PPL CSV (allow duplicates by using list)
    ppl_data = defaultdict(list)
    with open(ppl_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Normalize keys
            row = normalize_key_names(row, ppl_key_mapping)

            # Normalize dataset and model names
            if 'dataset' in row:
                row['dataset'] = normalize_dataset_name(row['dataset'])
            if 'model' in row:
                row['model'] = normalize_model_name(row['model'])

            # Create config key
            gen_key = create_config_key(row, gen_config_fields)
            wm_key = create_config_key(row, wm_config_fields)

            if match_by_file:
                file_base = extract_filename_base(row.get('file', ''))
                full_key = (file_base, gen_key, wm_key)
            else:
                full_key = (gen_key, wm_key)

            ppl_data[full_key].append(row)

    # Read z-score CSV (allow duplicates by using list)
    zscore_data = defaultdict(list)
    with open(zscore_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Normalize keys
            row = normalize_key_names(row, zscore_key_mapping)

            # Normalize dataset and model names
            if 'dataset' in row:
                row['dataset'] = normalize_dataset_name(row['dataset'])
            if 'model' in row:
                row['model'] = normalize_model_name(row['model'])

            # Create config key
            gen_key = create_config_key(row, gen_config_fields)
            wm_key = create_config_key(row, wm_config_fields)

            if match_by_file:
                file_base = extract_filename_base(row.get('file', ''))
                full_key = (file_base, gen_key, wm_key)
            else:
                full_key = (gen_key, wm_key)

            zscore_data[full_key].append(row)

    # Find matches and merge (generate all combinations)
    merged_data = []
    matched_keys = set()
    combination_count = 0

    for key, ppl_rows in ppl_data.items():
        if key in zscore_data:
            zscore_rows = zscore_data[key]
            matched_keys.add(key)

            # Generate all combinations of PPL and z-score rows
            for ppl_row in ppl_rows:
                for zscore_row in zscore_rows:
                    combination_count += 1

                    # Merge rows (PPL data + z-score specific data)
                    merged_row = ppl_row.copy()

                    # Add z-score specific fields
                    zscore_specific_fields = [
                        'z_score_version', 'n_samples',
                        'mean_zscore', 'std_zscore', 'min_zscore', 'max_zscore', 'median_zscore'
                    ]

                    # Add TPR fields
                    tpr_fields = [k for k in zscore_row.keys() if 'tpr_at_fpr' in k or
                                 'threshold_at_fpr' in k or 'detected_at_fpr' in k]

                    # Add attack fields if present
                    attack_fields = [k for k in zscore_row.keys() if k.startswith('attack_')]

                    for field in zscore_specific_fields + tpr_fields + attack_fields:
                        if field in zscore_row:
                            merged_row[field] = zscore_row[field]

                    # Create unique identifier for this combination if files have same name
                    if ppl_row.get('file') == zscore_row.get('file') and len(ppl_rows) * len(zscore_rows) > 1:
                        # Add combination index to filename
                        base_file = merged_row.get('file', '')
                        if base_file:
                            merged_row['original_file'] = base_file
                            merged_row['file'] = f"{base_file}_combo_{len(merged_data) + 1}"

                    merged_data.append(merged_row)

    # Report unmatched entries
    unmatched_ppl = set(ppl_data.keys()) - matched_keys
    unmatched_zscore = set(zscore_data.keys()) - matched_keys

    # Count total entries
    total_ppl_entries = sum(len(rows) for rows in ppl_data.values())
    total_zscore_entries = sum(len(rows) for rows in zscore_data.values())

    if verbose or (unmatched_ppl or unmatched_zscore):
        print(f"\nMatching Summary:")
        print(f"  PPL unique keys: {len(ppl_data)} ({total_ppl_entries} total entries)")
        print(f"  Z-score unique keys: {len(zscore_data)} ({total_zscore_entries} total entries)")
        print(f"  Matched keys: {len(matched_keys)}")
        print(f"  Unmatched PPL keys: {len(unmatched_ppl)}")
        print(f"  Unmatched z-score keys: {len(unmatched_zscore)}")
        if combination_count > len(matched_keys):
            print(f"  Generated combinations: {combination_count} (from {len(matched_keys)} matches)")

        if unmatched_ppl and verbose:
            print("\n  Sample unmatched PPL entries:")
            for key in list(unmatched_ppl)[:5]:
                if match_by_file:
                    print(f"    File: {key[0]}, Gen: {key[1][:3]}..., WM: {key[2][:3]}...")
                else:
                    print(f"    Gen: {key[0][:3]}..., WM: {key[1][:3]}...")

        if unmatched_zscore and verbose:
            print("\n  Sample unmatched z-score entries:")
            for key in list(unmatched_zscore)[:5]:
                if match_by_file:
                    print(f"    File: {key[0]}, Gen: {key[1][:3]}..., WM: {key[2][:3]}...")
                else:
                    print(f"    Gen: {key[0][:3]}..., WM: {key[1][:3]}...")

    # Write merged CSV
    if merged_data:
        # Determine field order
        all_fields = set()
        for row in merged_data:
            all_fields.update(row.keys())

        # Order fields logically
        field_order = ['file']  # File first
        if 'original_file' in all_fields:
            field_order.append('original_file')  # Original file if renamed

        # Generation config
        field_order.extend([f for f in gen_config_fields if f in all_fields])

        # Watermark config
        field_order.extend([f for f in wm_config_fields if f in all_fields])
        field_order.extend([f for f in ['wm_vocab_size', 'wm_enabled'] if f in all_fields])

        # PPL metrics
        ppl_fields = ['total_samples', 'mean_ppl', 'std_ppl', 'min_ppl', 'max_ppl',
                     'median_ppl', 'p10_ppl', 'p25_ppl', 'p50_ppl', 'p75_ppl', 'p90_ppl']
        field_order.extend([f for f in ppl_fields if f in all_fields])

        # Z-score metrics
        zscore_fields = ['z_score_version', 'n_samples', 'mean_zscore', 'std_zscore',
                        'min_zscore', 'max_zscore', 'median_zscore']
        field_order.extend([f for f in zscore_fields if f in all_fields])

        # Attack fields
        attack_fields = sorted([f for f in all_fields if f.startswith('attack_')])
        field_order.extend(attack_fields)

        # TPR metrics (sorted by FPR value)
        tpr_fields = sorted([f for f in all_fields if 'tpr_at_fpr' in f or
                           'threshold_at_fpr' in f or 'detected_at_fpr' in f])
        field_order.extend(tpr_fields)

        # Add any remaining fields
        field_order.extend([f for f in sorted(all_fields) if f not in field_order])

        # Write CSV
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=field_order)
            writer.writeheader()

            # Sort by file name for consistent output
            merged_data.sort(key=lambda x: x.get('file', ''))
            writer.writerows(merged_data)

        print(f"\nMerged CSV saved to: {output_file}")
        print(f"Total merged entries: {len(merged_data)}")

        # Print dataset summary
        datasets = set(row.get('dataset', '') for row in merged_data)
        if datasets:
            print(f"Datasets in merged file: {', '.join(sorted(filter(None, datasets)))}")

        # Print combination statistics if there were multiple matches
        if combination_count > len(matched_keys):
            avg_combinations = combination_count / len(matched_keys) if matched_keys else 0
            print(f"Average combinations per match: {avg_combinations:.1f}")
    else:
        print("\nNo matching entries found to merge!")


def main():
    parser = argparse.ArgumentParser(
        description="Merge PPL and z-score CSV files based on matching configurations"
    )

    parser.add_argument(
        "--ppl",
        type=str,
        required=True,
        help="Path to CSV file with PPL metrics"
    )

    parser.add_argument(
        "--zscore",
        type=str,
        required=True,
        help="Path to CSV file with z-score metrics"
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to output merged CSV (default: merged_ppl_zscore.csv)"
    )

    parser.add_argument(
        "--no-match-file",
        action="store_true",
        help="Don't match by filename, only by configurations"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed matching information"
    )

    parser.add_argument(
        "--normalize-dataset",
        action="store_true",
        default=True,
        help="Normalize dataset names to last two segments (default: True)"
    )

    parser.add_argument(
        "--no-normalize-dataset",
        dest="normalize_dataset",
        action="store_false",
        help="Don't normalize dataset names"
    )

    args = parser.parse_args()

    # Validate input files
    if not os.path.exists(args.ppl):
        print(f"Error: PPL file not found: {args.ppl}")
        return

    if not os.path.exists(args.zscore):
        print(f"Error: Z-score file not found: {args.zscore}")
        return

    # Set default output path
    if args.output is None:
        output_dir = os.path.dirname(args.ppl) or '.'
        args.output = os.path.join(output_dir, "merged_ppl_zscore.csv")

    # Merge files
    merge_csv_files(
        ppl_file=args.ppl,
        zscore_file=args.zscore,
        output_file=args.output,
        match_by_file=not args.no_match_file,
        verbose=args.verbose
    )


if __name__ == "__main__":
    main()