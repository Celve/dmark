import argparse
import csv
import json
import os
from typing import List, Dict, Optional
import numpy as np
from tqdm import tqdm


def extract_metadata_from_results(results: List[Dict]) -> Dict[str, Optional[str]]:
    """
    Extract metadata from the first instance in results.
    
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
        'batch_size': None,
        'vocab_size': None
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


def process_single_file(file_path: str) -> Dict:
    """
    Process a single JSON file to calculate average perplexity.
    
    Args:
        file_path: Path to JSON file with generation results
    
    Returns:
        Dictionary with analysis results for this file
    """
    with open(file_path, 'r') as f:
        results = json.load(f)
    
    # Extract metadata from the results
    metadata = extract_metadata_from_results(results)
    
    # Collect perplexity scores
    perplexities = []
    
    for result in results:
        # Try different locations for perplexity
        ppl = None
        
        # Check if perplexity is in text_quality
        if 'text_quality' in result and result['text_quality'] is not None:
            ppl = result['text_quality'].get('perplexity')
        
        # Check if perplexity is at top level
        elif 'perplexity' in result:
            ppl = result['perplexity']
        
        # Check if perplexity is in watermark section
        elif 'watermark' in result and result['watermark'] is not None:
            if 'perplexity' in result['watermark']:
                ppl = result['watermark']['perplexity']
        
        if ppl is not None and not np.isnan(ppl) and not np.isinf(ppl):
            perplexities.append(ppl)
    
    if not perplexities:
        return None
    
    # Calculate statistics
    return {
        'file': os.path.basename(file_path),
        'metadata': metadata,
        'statistics': {
            'total_samples': len(perplexities),
            'mean': float(np.mean(perplexities)),
            'std': float(np.std(perplexities)),
            'min': float(np.min(perplexities)),
            'max': float(np.max(perplexities)),
            'median': float(np.median(perplexities)),
            'percentiles': {
                '10%': float(np.percentile(perplexities, 10)),
                '25%': float(np.percentile(perplexities, 25)),
                '50%': float(np.percentile(perplexities, 50)),
                '75%': float(np.percentile(perplexities, 75)),
                '90%': float(np.percentile(perplexities, 90))
            }
        }
    }


def save_csv_results(all_results: List[Dict], input_dir: str, output_dir: str) -> None:
    """
    Save perplexity analysis results to a CSV file.
    
    Args:
        all_results: List of analysis results from each file
        input_dir: Directory containing input files (used for naming)
        output_dir: Directory to save CSV file
    """
    # Use input directory name as base for CSV filename
    dir_name = os.path.basename(os.path.normpath(input_dir))
    csv_path = os.path.join(output_dir, f'{dir_name}_ppl_analysis.csv')
    
    # Prepare CSV headers
    headers = [
        'file', 'dataset', 'model', 'has_watermark', 'strategy', 'ratio', 'delta', 'key', 
        'prebias', 'vocab_size', 'remasking', 'steps', 'gen_length', 'block_length', 
        'temperature', 'cfg_scale', 'batch_size', 'total_samples',
        'mean_ppl', 'std_ppl', 'min_ppl', 'max_ppl', 'median_ppl',
        'p10_ppl', 'p25_ppl', 'p50_ppl', 'p75_ppl', 'p90_ppl'
    ]
    
    # Write CSV
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        
        for result in all_results:
            meta = result['metadata']
            stats = result['statistics']
            percentiles = stats['percentiles']
            
            row = {
                'file': result['file'],
                'dataset': meta.get('dataset'),
                'model': meta.get('model'),
                'has_watermark': meta.get('has_watermark'),
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
                'total_samples': stats['total_samples'],
                'mean_ppl': stats['mean'],
                'std_ppl': stats['std'],
                'min_ppl': stats['min'],
                'max_ppl': stats['max'],
                'median_ppl': stats['median'],
                'p10_ppl': percentiles['10%'],
                'p25_ppl': percentiles['25%'],
                'p50_ppl': percentiles['50%'],
                'p75_ppl': percentiles['75%'],
                'p90_ppl': percentiles['90%']
            }
            
            writer.writerow(row)
    
    print(f"\nCSV saved to: {csv_path}")


def load_existing_csv(csv_path: str) -> set:
    """
    Load existing CSV file and return set of processed files.
    
    Args:
        csv_path: Path to CSV file
    
    Returns:
        Set of processed file names
    """
    processed_files = set()
    if os.path.exists(csv_path):
        try:
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if 'file' in row:
                        processed_files.add(row['file'])
        except Exception as e:
            print(f"Warning: Could not read existing CSV: {e}")
    return processed_files


def process_ppl_files(input_dir: str, output_dir: Optional[str] = None, delta_mode: bool = False) -> None:
    """
    Process JSON files to calculate average perplexity for each file.
    
    Args:
        input_dir: Directory containing JSON files with perplexity data
        output_dir: Directory to save output CSV file (defaults to input_dir)
        delta_mode: If True, only process files not already in the output CSV
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
    
    # Filter files that contain perplexity data or generation data
    valid_files = []
    for json_file in json_files:
        file_path = os.path.join(input_dir, json_file)
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                # Check if it's an array and first element has relevant data
                if isinstance(data, list) and len(data) > 0:
                    first_item = data[0]
                    # Check for perplexity in various possible locations
                    has_ppl = (
                        'perplexity' in first_item or
                        ('text_quality' in first_item and first_item.get('text_quality') and 'perplexity' in first_item['text_quality']) or
                        ('watermark' in first_item and first_item.get('watermark') and 'perplexity' in first_item['watermark'])
                    )
                    # Also accept files with generation data (even without PPL yet)
                    has_gen_data = 'data' in first_item and 'output' in first_item['data']
                    
                    if has_ppl or has_gen_data:
                        valid_files.append(json_file)
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
    
    if not valid_files:
        print(f"No JSON files with perplexity or generation data found in {input_dir}")
        return
    
    print(f"Found {len(valid_files)} files to process")
    
    # Check for existing CSV in delta mode
    processed_files = set()
    if delta_mode:
        dir_name = os.path.basename(os.path.normpath(input_dir))
        csv_path = os.path.join(output_dir, f'{dir_name}_ppl_analysis.csv')
        processed_files = load_existing_csv(csv_path)
        if processed_files:
            print(f"Delta mode: Found {len(processed_files)} already processed files")
            # Filter out already processed files
            valid_files = [f for f in valid_files if f not in processed_files]
            print(f"Remaining files to process: {len(valid_files)}")
    
    json_files = valid_files
    
    if not json_files:
        print("No new files to process")
        return
    
    all_results = []
    processed_count = 0
    skipped_count = len(processed_files) if delta_mode else 0
    
    for json_file in tqdm(json_files, desc="Processing files"):
        file_path = os.path.join(input_dir, json_file)
        file_results = process_single_file(file_path)
        
        if file_results:
            all_results.append(file_results)
            processed_count += 1
            
            # Print results for this file
            print(f"\n{'='*70}")
            print(f"File: {json_file}")
            print(f"Samples: {file_results['statistics']['total_samples']}")
            print(f"Mean PPL: {file_results['statistics']['mean']:.2f}")
            print(f"Std PPL: {file_results['statistics']['std']:.2f}")
            print(f"Min PPL: {file_results['statistics']['min']:.2f}")
            print(f"Max PPL: {file_results['statistics']['max']:.2f}")
            print(f"Median PPL: {file_results['statistics']['median']:.2f}")
    
    if not all_results:
        print("No perplexity data found in any files")
        print("Note: Files need to have perplexity already calculated.")
        print("You can use 'python -m dmark.eval.ppl' to add perplexity scores first.")
        return
    
    # Save CSV results (append mode for delta)
    if delta_mode and processed_files:
        # Append to existing CSV
        dir_name = os.path.basename(os.path.normpath(input_dir))
        csv_path = os.path.join(output_dir, f'{dir_name}_ppl_analysis.csv')
        
        # Read existing data
        existing_data = []
        if os.path.exists(csv_path):
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                existing_data = list(reader)
        
        # Combine with new results
        headers = [
            'file', 'dataset', 'model', 'has_watermark', 'strategy', 'ratio', 'delta', 'key', 
            'prebias', 'vocab_size', 'remasking', 'steps', 'gen_length', 'block_length', 
            'temperature', 'cfg_scale', 'batch_size', 'total_samples',
            'mean_ppl', 'std_ppl', 'min_ppl', 'max_ppl', 'median_ppl',
            'p10_ppl', 'p25_ppl', 'p50_ppl', 'p75_ppl', 'p90_ppl'
        ]
        
        # Write combined data
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()
            
            # Write existing data
            for row in existing_data:
                writer.writerow(row)
            
            # Write new data
            for result in all_results:
                meta = result['metadata']
                stats = result['statistics']
                percentiles = stats['percentiles']
                
                row = {
                    'file': result['file'],
                    'dataset': meta.get('dataset'),
                    'model': meta.get('model'),
                    'has_watermark': meta.get('has_watermark'),
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
                    'total_samples': stats['total_samples'],
                    'mean_ppl': stats['mean'],
                    'std_ppl': stats['std'],
                    'min_ppl': stats['min'],
                    'max_ppl': stats['max'],
                    'median_ppl': stats['median'],
                    'p10_ppl': percentiles['10%'],
                    'p25_ppl': percentiles['25%'],
                    'p50_ppl': percentiles['50%'],
                    'p75_ppl': percentiles['75%'],
                    'p90_ppl': percentiles['90%']
                }
                
                writer.writerow(row)
        
        print(f"\nCSV updated: {csv_path}")
    else:
        # Normal save (overwrite)
        save_csv_results(all_results, input_dir, output_dir)
    
    # Print overall summary
    print("\n" + "="*70)
    print("SUMMARY ACROSS ALL FILES")
    print("="*70)
    
    # Aggregate statistics
    all_means = [result['statistics']['mean'] for result in all_results]
    all_samples = sum(result['statistics']['total_samples'] for result in all_results)
    
    print(f"\nFiles processed in this run: {processed_count}")
    if delta_mode and skipped_count > 0:
        print(f"Files skipped (already processed): {skipped_count}")
    print(f"Total samples across new files: {all_samples}")
    if all_means:
        print(f"Average mean PPL across new files: {np.mean(all_means):.2f}")
        print(f"Std of mean PPL across new files: {np.std(all_means):.2f}")
        print(f"Min mean PPL: {np.min(all_means):.2f}")
        print(f"Max mean PPL: {np.max(all_means):.2f}")


def main():
    parser = argparse.ArgumentParser(
        description="Calculate average perplexity for each JSON file in a directory and generate CSV report"
    )
    
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Directory containing JSON files with perplexity data"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Directory to save output CSV file (defaults to input directory)"
    )
    
    parser.add_argument(
        "--delta",
        action="store_true",
        help="Delta mode: only process files not already in the output CSV"
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
    process_ppl_files(args.input, args.output, args.delta)


if __name__ == "__main__":
    main()