#!/usr/bin/env python3
"""
Convert threshold JSON files to CSV format.
Extracts filename and threshold values for different TPR targets.
"""

import argparse
import json
import csv
from pathlib import Path
import sys


def extract_filename_parts(filename):
    """Extract key parameters from filename for better organization."""
    # Remove .json and _zscore suffix
    name = filename.replace('.json', '').replace('_zscore', '')
    
    # Extract key parameters
    parts = {}
    
    # Extract dataset
    if 'openai_gsm8k' in name:
        parts['dataset'] = 'gsm8k'
    elif 'sentence-transformers_eli5' in name:
        parts['dataset'] = 'eli5'
    elif 'allenai_c4' in name:
        parts['dataset'] = 'c4'
    elif 'openai_humaneval' in name:
        parts['dataset'] = 'humaneval'
    elif 'wmt16' in name:
        parts['dataset'] = 'wmt16'
    else:
        parts['dataset'] = 'unknown'
    
    # Extract remasking strategy
    for mask in ['low_confidence', 'random', 'right_to_left', 'left_to_right']:
        if f'mask_{mask}' in name:
            parts['remasking'] = mask
            break
    else:
        parts['remasking'] = 'unknown'
    
    # Extract watermark strategy
    for strategy in ['normal', 'predict', 'reverse']:
        if f'-{strategy}-' in name:
            parts['strategy'] = strategy
            break
    else:
        if 'nowm' in name:
            parts['strategy'] = 'none'
        else:
            parts['strategy'] = 'unknown'
    
    # Extract delta value
    if '-d' in name:
        try:
            delta_start = name.index('-d') + 2
            delta_end = name.index('-', delta_start)
            parts['delta'] = float(name[delta_start:delta_end])
        except:
            parts['delta'] = 0.0
    else:
        parts['delta'] = 0.0
    
    # Extract ratio
    if '-r' in name:
        try:
            ratio_start = name.index('-r') + 2
            ratio_end = name.index('-', ratio_start)
            parts['ratio'] = float(name[ratio_start:ratio_end])
        except:
            parts['ratio'] = 0.5
    else:
        parts['ratio'] = 0.5
    
    # Extract key
    if '-k' in name:
        try:
            key_start = name.index('-k') + 2
            key_end = name.index('-', key_start)
            parts['key'] = int(name[key_start:key_end])
        except:
            parts['key'] = 42
    else:
        parts['key'] = 42
    
    return parts


def json_to_csv(json_file, output_file=None, include_metadata=False):
    """Convert JSON threshold data to CSV format."""
    
    # Read JSON file
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Prepare CSV data
    csv_data = []
    
    # Define headers based on whether to include metadata
    if include_metadata:
        headers = [
            'filename', 
            'dataset', 
            'remasking', 
            'strategy', 
            'delta', 
            'ratio',
            'key',
            'threshold_90', 
            'threshold_95', 
            'threshold_99', 
            'threshold_999',
            'actual_tpr_90',
            'actual_tpr_95', 
            'actual_tpr_99',
            'actual_tpr_999',
            'mean',
            'std',
            'min',
            'max',
            'median',
            'total_samples'
        ]
    else:
        headers = [
            'filename',
            'threshold_90',
            'threshold_95', 
            'threshold_99',
            'threshold_999'
        ]
    
    # Process each file result
    for file_result in data.get('per_file_results', []):
        filename = file_result['file']
        row = {'filename': filename}
        
        # Add metadata if requested
        if include_metadata:
            parts = extract_filename_parts(filename)
            row.update(parts)
            
            # Add statistics
            stats = file_result.get('statistics', {})
            row['mean'] = stats.get('mean', '')
            row['std'] = stats.get('std', '')
            row['min'] = stats.get('min', '')
            row['max'] = stats.get('max', '')
            row['median'] = stats.get('median', '')
            row['total_samples'] = stats.get('total_samples', '')
        
        # Extract thresholds for each TPR target
        thresholds = file_result.get('thresholds', [])
        for threshold_data in thresholds:
            tpr = threshold_data['target_tpr']
            threshold_value = threshold_data['threshold']
            actual_tpr = threshold_data.get('actual_tpr', tpr)
            
            if tpr == 0.9:
                row['threshold_90'] = threshold_value
                if include_metadata:
                    row['actual_tpr_90'] = actual_tpr
            elif tpr == 0.95:
                row['threshold_95'] = threshold_value
                if include_metadata:
                    row['actual_tpr_95'] = actual_tpr
            elif tpr == 0.99:
                row['threshold_99'] = threshold_value
                if include_metadata:
                    row['actual_tpr_99'] = actual_tpr
            elif tpr == 0.999:
                row['threshold_999'] = threshold_value
                if include_metadata:
                    row['actual_tpr_999'] = actual_tpr
        
        csv_data.append(row)
    
    # Sort by filename or by metadata columns if available
    if include_metadata:
        csv_data.sort(key=lambda x: (
            x.get('dataset', ''),
            x.get('strategy', ''),
            x.get('remasking', ''),
            x.get('delta', 0),
            x.get('filename', '')
        ))
    else:
        csv_data.sort(key=lambda x: x.get('filename', ''))
    
    # Determine output file
    if output_file is None:
        output_file = Path(json_file).with_suffix('.csv')
    
    # Write CSV
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(csv_data)
    
    return output_file, len(csv_data)


def main():
    parser = argparse.ArgumentParser(
        description='Convert threshold JSON files to CSV format'
    )
    parser.add_argument(
        'input',
        type=str,
        help='Input JSON file path'
    )
    parser.add_argument(
        '--output',
        '-o',
        type=str,
        default=None,
        help='Output CSV file path (default: input_file.csv)'
    )
    parser.add_argument(
        '--metadata',
        '-m',
        action='store_true',
        help='Include metadata columns (dataset, strategy, etc.)'
    )
    parser.add_argument(
        '--simple',
        '-s',
        action='store_true',
        help='Simple output with only filename and thresholds (default)'
    )
    
    args = parser.parse_args()
    
    # Check if input file exists
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file '{args.input}' not found")
        sys.exit(1)
    
    if not input_path.is_file():
        print(f"Error: '{args.input}' is not a file")
        sys.exit(1)
    
    # Convert to CSV
    try:
        include_metadata = args.metadata and not args.simple
        output_file, num_rows = json_to_csv(
            args.input,
            args.output,
            include_metadata=include_metadata
        )
        
        print(f"Successfully converted {num_rows} entries to CSV")
        print(f"Output saved to: {output_file}")
        
        if include_metadata:
            print("Included metadata columns: dataset, remasking, strategy, delta, ratio, key, statistics")
        else:
            print("Simple format: filename and threshold values only")
            
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in input file: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()