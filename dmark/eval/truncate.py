import argparse
import json
import os
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from datetime import datetime
import numpy as np
from transformers import AutoTokenizer


def get_min_length_from_metadata(results: List[Dict[str, Any]]) -> int:
    """
    Extract minimum output length from experiment metadata.
    
    Args:
        results: List of result dictionaries
    
    Returns:
        Minimum output length (default 200 if not found)
    """
    # Look for experiment metadata in first result that has it
    for result in results:
        if 'expr_metadata' in result and result['expr_metadata']:
            exp_meta = result['expr_metadata']
            if 'minimum_output_token' in exp_meta and exp_meta['minimum_output_token'] is not None:
                return exp_meta['minimum_output_token']
        
        # Also check generation_metadata for gen_length as fallback
        if 'generation_metadata' in result and result['generation_metadata']:
            gen_meta = result['generation_metadata']
            if 'gen_length' in gen_meta and gen_meta['gen_length'] is not None:
                # Use gen_length as a reasonable truncation length
                return gen_meta['gen_length']
    
    # Default to 200 if no metadata found
    return 200


def get_model_name_from_metadata(results: List[Dict[str, Any]]) -> Optional[str]:
    """
    Extract model name from generation metadata.
    
    Args:
        results: List of result dictionaries
    
    Returns:
        Model name or None if not found
    """
    for result in results:
        if 'generation_metadata' in result and result['generation_metadata']:
            gen_meta = result['generation_metadata']
            if 'model' in gen_meta:
                return gen_meta['model']
    return None


def truncate_output(
    result: Dict[str, Any], 
    min_length: int,
    tokenizer: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Truncate output and output_ids to specified minimum length.
    
    Args:
        result: Single result dictionary
        min_length: Minimum output length to truncate to
        tokenizer: Tokenizer for proper text decoding (optional)
    
    Returns:
        Modified result with truncated fields added
    """
    # Create a deep copy to avoid modifying original
    import copy
    truncated_result = copy.deepcopy(result)
    
    # Check if data field exists and contains output_ids
    if 'data' in result and 'output_ids' in result['data']:
        output_ids = result['data']['output_ids']
        
        # Truncate output_ids to min_length
        truncated_ids = output_ids[:min_length]
        
        # Add truncated_output_ids field
        truncated_result['data']['truncated_output_ids'] = truncated_ids
        
        # Decode truncated IDs to get proper text
        if tokenizer is not None and len(truncated_ids) > 0:
            # Use tokenizer to properly decode the truncated token IDs
            truncated_text = tokenizer.decode(truncated_ids, skip_special_tokens=True)
        else:
            # Fallback: if no tokenizer, do character-based truncation
            if 'output' in result['data']:
                original_output = result['data']['output']
                if len(output_ids) > 0:
                    # Estimate truncation ratio based on token count
                    truncation_ratio = min(1.0, min_length / len(output_ids))
                    char_limit = int(len(original_output) * truncation_ratio)
                    truncated_text = original_output[:char_limit]
                else:
                    truncated_text = original_output
            else:
                truncated_text = ""
        
        truncated_result['data']['truncated_output'] = truncated_text
        
        # Add truncation metadata
        truncated_result['truncation_metadata'] = {
            'original_length': len(output_ids),
            'truncated_length': len(truncated_ids),
            'min_length_used': min_length,
            'was_truncated': len(output_ids) > min_length,
            'tokenizer_used': tokenizer is not None
        }
    
    return truncated_result


def process_json_file(
    input_file: str,
    output_file: str,
    min_length: Optional[int] = None,
    model_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Process a single JSON file and add truncated fields.
    
    Args:
        input_file: Path to input JSON file
        output_file: Path to output JSON file
        min_length: Minimum output length (auto-detected if None)
        model_name: Model name for tokenizer (auto-detected if None)
    
    Returns:
        Processing statistics
    """
    with open(input_file, 'r') as f:
        results = json.load(f)
    
    # Auto-detect min_length from metadata if not provided
    if min_length is None:
        min_length = get_min_length_from_metadata(results)
    
    # Auto-detect model name from metadata if not provided
    if model_name is None:
        model_name = get_model_name_from_metadata(results)
    
    # Load tokenizer if model name is available
    tokenizer = None
    if model_name:
        try:
            print(f"  Loading tokenizer: {model_name}")
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        except Exception as e:
            print(f"  Warning: Could not load tokenizer for {model_name}: {e}")
            print(f"  Falling back to character-based truncation")
    
    # Process each result
    truncated_results = []
    stats = {
        'total_samples': len(results),
        'truncated_samples': 0,
        'min_length_used': min_length,
        'model_name': model_name,
        'tokenizer_loaded': tokenizer is not None,
        'original_lengths': [],
        'truncated_lengths': []
    }
    
    for result in results:
        truncated_result = truncate_output(result, min_length, tokenizer)
        truncated_results.append(truncated_result)
        
        # Update statistics
        if 'truncation_metadata' in truncated_result:
            meta = truncated_result['truncation_metadata']
            stats['original_lengths'].append(meta['original_length'])
            stats['truncated_lengths'].append(meta['truncated_length'])
            if meta['was_truncated']:
                stats['truncated_samples'] += 1
    
    # Save truncated results
    with open(output_file, 'w') as f:
        json.dump(truncated_results, f, indent=4)
    
    return stats


def process_directory(
    input_dir: str,
    output_dir: Optional[str] = None,
    min_length: Optional[int] = None,
    model_name: Optional[str] = None
) -> None:
    """
    Process all JSON files in a directory and save truncated versions.
    
    Args:
        input_dir: Directory containing input JSON files
        output_dir: Directory to save truncated files (auto-generated if None)
        min_length: Minimum output length (auto-detected per file if None)
        model_name: Model name for tokenizer (auto-detected per file if None)
    """
    # Auto-generate output directory name if not provided
    if output_dir is None:
        base_dir = os.path.dirname(input_dir.rstrip('/'))
        dir_name = os.path.basename(input_dir.rstrip('/'))
        output_dir = os.path.join(base_dir, f"{dir_name}_truncated")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all JSON files
    json_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.json')])
    
    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return
    
    print(f"\n{'='*70}")
    print(f"Truncation Configuration")
    print(f"{'='*70}")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Min length: {'auto-detect from metadata' if min_length is None else min_length}")
    print(f"Model: {'auto-detect from metadata' if model_name is None else model_name}")
    print(f"Files to process: {len(json_files)}")
    print(f"{'='*70}\n")
    
    # Track overall statistics
    overall_stats = {
        'files_processed': 0,
        'total_samples': 0,
        'total_truncated': 0,
        'min_lengths_used': {},
        'models_used': {},
        'all_original_lengths': [],
        'all_truncated_lengths': []
    }
    
    # Process each file
    for json_file in tqdm(json_files, desc="Processing files"):
        input_path = os.path.join(input_dir, json_file)
        output_path = os.path.join(output_dir, json_file)  # Keep original filename
        
        try:
            stats = process_json_file(input_path, output_path, min_length, model_name)
            
            # Update overall statistics
            overall_stats['files_processed'] += 1
            overall_stats['total_samples'] += stats['total_samples']
            overall_stats['total_truncated'] += stats['truncated_samples']
            overall_stats['all_original_lengths'].extend(stats['original_lengths'])
            overall_stats['all_truncated_lengths'].extend(stats['truncated_lengths'])
            
            # Track min_length used for this file
            min_length_used = stats['min_length_used']
            if min_length_used not in overall_stats['min_lengths_used']:
                overall_stats['min_lengths_used'][min_length_used] = 0
            overall_stats['min_lengths_used'][min_length_used] += 1
            
            # Track model used
            model_used = stats['model_name'] or 'unknown'
            if model_used not in overall_stats['models_used']:
                overall_stats['models_used'][model_used] = 0
            overall_stats['models_used'][model_used] += 1
            
            tokenizer_info = "✓ tokenizer" if stats['tokenizer_loaded'] else "✗ char-based"
            print(f"✓ {json_file}: {stats['truncated_samples']}/{stats['total_samples']} truncated (min={min_length_used}, {tokenizer_info})")
            
        except Exception as e:
            print(f"✗ {json_file}: Error - {e}")
    
    # Save truncation summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'input_directory': input_dir,
        'output_directory': output_dir,
        'configuration': {
            'min_length': 'auto-detected' if min_length is None else min_length,
            'model': 'auto-detected' if model_name is None else model_name
        },
        'statistics': {
            'files_processed': overall_stats['files_processed'],
            'total_samples': overall_stats['total_samples'],
            'samples_truncated': overall_stats['total_truncated'],
            'truncation_rate': overall_stats['total_truncated'] / overall_stats['total_samples'] if overall_stats['total_samples'] > 0 else 0,
            'min_lengths_used': overall_stats['min_lengths_used'],
            'models_used': overall_stats['models_used']
        }
    }
    
    if overall_stats['all_original_lengths']:
        summary['statistics']['length_statistics'] = {
            'avg_original_length': float(np.mean(overall_stats['all_original_lengths'])),
            'avg_truncated_length': float(np.mean(overall_stats['all_truncated_lengths'])),
            'std_original_length': float(np.std(overall_stats['all_original_lengths'])),
            'std_truncated_length': float(np.std(overall_stats['all_truncated_lengths'])),
            'min_original_length': int(np.min(overall_stats['all_original_lengths'])),
            'max_original_length': int(np.max(overall_stats['all_original_lengths'])),
            'min_truncated_length': int(np.min(overall_stats['all_truncated_lengths'])),
            'max_truncated_length': int(np.max(overall_stats['all_truncated_lengths']))
        }
    
    summary_path = os.path.join(output_dir, '_truncation_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print final summary
    print(f"\n{'='*70}")
    print(f"Truncation Summary")
    print(f"{'='*70}")
    print(f"Files processed: {overall_stats['files_processed']}/{len(json_files)}")
    print(f"Total samples: {overall_stats['total_samples']}")
    print(f"Samples truncated: {overall_stats['total_truncated']} ({overall_stats['total_truncated']/overall_stats['total_samples']*100:.1f}%)")
    
    if overall_stats['min_lengths_used']:
        print(f"\nMin lengths used:")
        for length, count in sorted(overall_stats['min_lengths_used'].items()):
            print(f"  {length} tokens: {count} files")
    
    if overall_stats['models_used']:
        print(f"\nModels used:")
        for model, count in overall_stats['models_used'].items():
            print(f"  {model}: {count} files")
    
    if overall_stats['all_original_lengths']:
        print(f"\nLength Statistics:")
        print(f"  Original: {np.mean(overall_stats['all_original_lengths']):.1f} ± {np.std(overall_stats['all_original_lengths']):.1f} tokens")
        print(f"  Truncated: {np.mean(overall_stats['all_truncated_lengths']):.1f} ± {np.std(overall_stats['all_truncated_lengths']):.1f} tokens")
        print(f"  Range: [{np.min(overall_stats['all_original_lengths'])}, {np.max(overall_stats['all_original_lengths'])}] → [{np.min(overall_stats['all_truncated_lengths'])}, {np.max(overall_stats['all_truncated_lengths'])}]")
    
    print(f"\nOutput directory: {output_dir}")
    print(f"Summary saved to: {summary_path}")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(
        description="Truncate outputs in JSON files to minimum length from metadata or specified value"
    )
    
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input directory containing JSON files or single JSON file"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (auto-generated with '_truncated' suffix if not specified)"
    )
    
    parser.add_argument(
        "--min_length",
        type=int,
        default=None,
        help="Minimum output length in tokens (auto-detected from expr_metadata if not specified, default: 200)"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name for tokenizer (auto-detected from generation_metadata if not specified)"
    )
    
    args = parser.parse_args()
    
    # Check if input exists
    if not os.path.exists(args.input):
        print(f"Error: Input path '{args.input}' not found")
        return
    
    # Process based on input type
    if os.path.isfile(args.input):
        # Single file processing
        if args.output is None:
            # Auto-generate output filename
            dir_path = os.path.dirname(args.input)
            base_name = os.path.splitext(os.path.basename(args.input))[0]
            output_file = os.path.join(dir_path, f"{base_name}_truncated.json")
        else:
            output_file = args.output
        
        # Ensure output directory exists
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        print(f"Processing single file: {args.input}")
        print(f"Output: {output_file}")
        
        stats = process_json_file(args.input, output_file, args.min_length, args.model)
        
        print(f"\nTruncation complete!")
        print(f"Samples processed: {stats['total_samples']}")
        print(f"Samples truncated: {stats['truncated_samples']}")
        print(f"Min length used: {stats['min_length_used']}")
        print(f"Model: {stats['model_name'] or 'unknown'}")
        print(f"Tokenizer: {'loaded' if stats['tokenizer_loaded'] else 'not loaded (using char-based truncation)'}")
        print(f"Output saved to: {output_file}")
        
    elif os.path.isdir(args.input):
        # Directory processing
        process_directory(args.input, args.output, args.min_length, args.model)
    else:
        print(f"Error: '{args.input}' is neither a file nor a directory")


if __name__ == "__main__":
    main()