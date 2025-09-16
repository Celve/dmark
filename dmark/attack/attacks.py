import argparse
import json
import os
import random
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer
from datetime import datetime


def random_swap_attack(token_ids: List[int], ratio: float = 0.2, seed: int = None) -> List[int]:
    """
    Randomly swap pairs of tokens in the text.
    
    Args:
        token_ids: List of token IDs to attack
        ratio: Ratio of tokens to swap (default: 0.2 = 20%)
        seed: Random seed for reproducibility
    
    Returns:
        List of token IDs after swapping
    """
    if seed is not None:
        random.seed(seed)
    
    attacked_ids = token_ids.copy()
    n_tokens = len(attacked_ids)
    
    if n_tokens < 2:
        return attacked_ids
    
    # Calculate number of swaps (each swap affects 2 tokens)
    n_swaps = int(n_tokens * ratio / 2)
    
    # Perform swaps
    for _ in range(n_swaps):
        # Select two different random positions
        pos1 = random.randint(0, n_tokens - 1)
        pos2 = random.randint(0, n_tokens - 1)
        
        # Ensure positions are different
        while pos2 == pos1 and n_tokens > 1:
            pos2 = random.randint(0, n_tokens - 1)
        
        # Swap tokens
        attacked_ids[pos1], attacked_ids[pos2] = attacked_ids[pos2], attacked_ids[pos1]
    
    return attacked_ids


def random_delete_attack(token_ids: List[int], ratio: float = 0.2, seed: int = None) -> List[int]:
    """
    Randomly delete tokens from the text.
    
    Args:
        token_ids: List of token IDs to attack
        ratio: Ratio of tokens to delete (default: 0.2 = 20%)
        seed: Random seed for reproducibility
    
    Returns:
        List of token IDs after deletion
    """
    if seed is not None:
        random.seed(seed)
    
    n_tokens = len(token_ids)
    
    if n_tokens == 0:
        return token_ids
    
    # Calculate number of tokens to keep
    n_keep = int(n_tokens * (1 - ratio))
    
    # Randomly select indices to keep
    all_indices = list(range(n_tokens))
    keep_indices = sorted(random.sample(all_indices, min(n_keep, n_tokens)))
    
    # Return tokens at kept indices
    attacked_ids = [token_ids[i] for i in keep_indices]
    
    return attacked_ids


def random_insert_attack(token_ids: List[int], vocab_size: int = 126464, ratio: float = 0.2, seed: int = None) -> List[int]:
    """
    Randomly insert tokens into the text.
    
    Args:
        token_ids: List of token IDs to attack
        vocab_size: Size of vocabulary for random token selection
        ratio: Ratio of tokens to insert (default: 0.2 = 20%)
        seed: Random seed for reproducibility
    
    Returns:
        List of token IDs after insertion
    """
    if seed is not None:
        random.seed(seed)
    
    n_tokens = len(token_ids)
    n_insert = int(n_tokens * ratio)
    
    attacked_ids = token_ids.copy()
    
    # Insert tokens at random positions
    for _ in range(n_insert):
        position = random.randint(0, len(attacked_ids))
        random_token = random.randint(0, vocab_size - 1)
        attacked_ids.insert(position, random_token)
    
    return attacked_ids


def synonym_attack(token_ids: List[int], tokenizer: Any, ratio: float = 0.2, seed: int = None) -> List[int]:
    """
    Replace tokens with random tokens (simple synonym attack).
    
    Args:
        token_ids: List of token IDs to attack
        tokenizer: Tokenizer for vocabulary size
        ratio: Ratio of tokens to replace (default: 0.2 = 20%)
        seed: Random seed for reproducibility
    
    Returns:
        List of token IDs after replacement
    """
    if seed is not None:
        random.seed(seed)
    
    attacked_ids = token_ids.copy()
    n_tokens = len(attacked_ids)
    vocab_size = len(tokenizer)
    if n_tokens == 0:
        return attacked_ids
    
    # Calculate number of tokens to replace
    n_replace = int(n_tokens * ratio)
    
    # Randomly select positions to replace
    positions = random.sample(range(n_tokens), min(n_replace, n_tokens))
    
    # Replace tokens at selected positions
    for pos in positions:
        attacked_ids[pos] = random.randint(0, vocab_size - 1)
    
    return attacked_ids


def apply_attack(
    token_ids: List[int],
    attack_type: str,
    ratio: float,
    seed: int = None,
    tokenizer: Any = None
) -> Tuple[List[int], Dict[str, Any]]:
    """
    Apply the specified attack to token IDs.
    
    Args:
        token_ids: List of token IDs to attack
        attack_type: Type of attack to apply
        ratio: Attack ratio
        seed: Random seed
        tokenizer: Tokenizer (needed for some attacks)
    
    Returns:
        Tuple of (attacked token IDs, attack metadata)
    """
    original_length = len(token_ids)
    
    if attack_type == 'swap':
        attacked_ids = random_swap_attack(token_ids, ratio=ratio, seed=seed)
        tokens_affected = int(original_length * ratio)
    elif attack_type == 'delete':
        attacked_ids = random_delete_attack(token_ids, ratio=ratio, seed=seed)
        tokens_affected = original_length - len(attacked_ids)
    elif attack_type == 'insert':
        vocab_size = len(tokenizer) if tokenizer else 126464
        print(f"Vocab size: {vocab_size}")
        attacked_ids = random_insert_attack(token_ids, vocab_size=vocab_size, ratio=ratio, seed=seed)
        tokens_affected = len(attacked_ids) - original_length
    elif attack_type == 'synonym':
        attacked_ids = synonym_attack(token_ids, tokenizer=tokenizer, ratio=ratio, seed=seed)
        tokens_affected = int(original_length * ratio)
    else:
        raise ValueError(f"Unknown attack type: {attack_type}")
    
    metadata = {
        'type': attack_type,
        'ratio': ratio,
        'original_length': original_length,
        'attacked_length': len(attacked_ids),
        'tokens_affected': tokens_affected,
        'seed': seed
    }
    
    return attacked_ids, metadata


def process_json_file(
    input_file: str,
    output_file: str,
    attack_type: str,
    ratio: float = 0.2,
    seed: int = None,
    model_name: str = "GSAI-ML/LLaDA-8B-Instruct",
    tokenizer: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Process a JSON file and apply the specified attack.
    
    Args:
        input_file: Path to input JSON file
        output_file: Path to output JSON file
        attack_type: Type of attack
        ratio: Attack ratio
        seed: Random seed
        model_name: Model name for tokenizer
        tokenizer: Pre-initialized tokenizer (optional)
    
    Returns:
        Statistics about the attack
    """
    # Initialize tokenizer if not provided
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    with open(input_file, 'r') as f:
        results = json.load(f)
    
    attacked_results = []
    stats = {
        'total_samples': 0,
        'successful_attacks': 0,
        'failed_attacks': 0,
        'original_lengths': [],
        'attacked_lengths': [],
        'tokens_affected': []
    }
    
    for idx, result in enumerate(results):
        stats['total_samples'] += 1
        
        # Create a copy of the result
        attacked_result = result.copy()
        
        # Get output IDs - prioritize truncated_output_ids if available
        if 'data' in result:
            # Use truncated_output_ids if available, otherwise use output_ids
            if 'truncated_output_ids' in result['data']:
                output_ids = result['data']['truncated_output_ids']
            elif 'output_ids' in result['data']:
                output_ids = result['data']['output_ids']
            else:
                output_ids = None
            
            if output_ids is not None:
                try:
                    # Apply attack
                    attacked_ids, attack_metadata = apply_attack(
                        output_ids,
                        attack_type,
                        ratio,
                        seed=seed + idx if seed is not None else None,
                        tokenizer=tokenizer
                    )
                    
                    # Update the same field we read from
                    if 'truncated_output_ids' in result['data']:
                        # We used truncated, so update truncated fields
                        attacked_result['data']['truncated_output_ids'] = attacked_ids
                        # Decode attacked IDs to get new output text
                        attacked_text = tokenizer.decode(attacked_ids, skip_special_tokens=True)
                        # Store original truncated output if it exists
                        if 'truncated_output' in result['data']:
                            attacked_result['data']['original_truncated_output'] = result['data']['truncated_output']
                        attacked_result['data']['truncated_output'] = attacked_text
                    else:
                        # We used regular output_ids, so update regular fields
                        attacked_result['data']['output_ids'] = attacked_ids
                        # Decode attacked IDs to get new output text
                        attacked_text = tokenizer.decode(attacked_ids, skip_special_tokens=True)
                        # Store original output if it exists
                        if 'output' in result['data']:
                            attacked_result['data']['original_output'] = result['data']['output']
                        attacked_result['data']['output'] = attacked_text
                    
                    # Add attack metadata
                    attacked_result['attack_metadata'] = attack_metadata
                    
                    # Update statistics
                    stats['successful_attacks'] += 1
                    stats['original_lengths'].append(attack_metadata['original_length'])
                    stats['attacked_lengths'].append(attack_metadata['attacked_length'])
                    stats['tokens_affected'].append(attack_metadata['tokens_affected'])
                    
                except Exception as e:
                    print(f"  Warning: Failed to attack sample {idx}: {e}")
                    stats['failed_attacks'] += 1
            else:
                stats['failed_attacks'] += 1
        else:
            stats['failed_attacks'] += 1
        
        attacked_results.append(attacked_result)
    
    # Save results
    with open(output_file, 'w') as f:
        json.dump(attacked_results, f, indent=4)
    
    return stats


def process_directory(
    input_dir: str,
    output_dir: Optional[str] = None,
    attack_type: str = 'swap',
    ratio: float = 0.2,
    seed: int = None,
    model_name: str = "GSAI-ML/LLaDA-8B-Instruct"
) -> None:
    """
    Process all JSON files in a directory and save attacked versions.
    
    Args:
        input_dir: Directory containing input JSON files
        output_dir: Directory to save attacked files (auto-generated if None)
        attack_type: Type of attack to apply
        ratio: Attack ratio
        seed: Random seed
        model_name: Model name for tokenizer
    """
    # Auto-generate output directory name if not provided
    if output_dir is None:
        base_dir = os.path.dirname(input_dir.rstrip('/'))
        dir_name = os.path.basename(input_dir.rstrip('/'))
        # Include attack type and ratio in directory name
        ratio_percent = int(ratio * 100)
        output_dir = os.path.join(base_dir, f"{dir_name}_attack_{attack_type}_{ratio_percent}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all JSON files
    json_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.json')])
    
    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return
    
    print(f"\n{'='*70}")
    print(f"Attack Configuration")
    print(f"{'='*70}")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Attack type: {attack_type}")
    print(f"Attack ratio: {ratio:.1%}")
    print(f"Files to process: {len(json_files)}")
    print(f"Random seed: {seed}")
    print(f"Model: {model_name}")
    print(f"{'='*70}\n")
    
    # Initialize tokenizer once for all files
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Track overall statistics
    overall_stats = {
        'files_processed': 0,
        'total_samples': 0,
        'successful_attacks': 0,
        'failed_attacks': 0,
        'all_original_lengths': [],
        'all_attacked_lengths': [],
        'all_tokens_affected': []
    }
    
    # Process each file
    for json_file in tqdm(json_files, desc="Processing files"):
        input_path = os.path.join(input_dir, json_file)
        output_path = os.path.join(output_dir, json_file)  # Keep original filename
        
        try:
            stats = process_json_file(
                input_path,
                output_path,
                attack_type,
                ratio,
                seed,
                model_name,
                tokenizer
            )
            
            # Update overall statistics
            overall_stats['files_processed'] += 1
            overall_stats['total_samples'] += stats['total_samples']
            overall_stats['successful_attacks'] += stats['successful_attacks']
            overall_stats['failed_attacks'] += stats['failed_attacks']
            overall_stats['all_original_lengths'].extend(stats['original_lengths'])
            overall_stats['all_attacked_lengths'].extend(stats['attacked_lengths'])
            overall_stats['all_tokens_affected'].extend(stats['tokens_affected'])
            
            print(f"✓ {json_file}: {stats['successful_attacks']}/{stats['total_samples']} samples attacked")
            
        except Exception as e:
            print(f"✗ {json_file}: Error - {e}")
    
    # Save attack summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'input_directory': input_dir,
        'output_directory': output_dir,
        'attack_configuration': {
            'type': attack_type,
            'ratio': ratio,
            'seed': seed,
            'model': model_name
        },
        'statistics': {
            'files_processed': overall_stats['files_processed'],
            'total_samples': overall_stats['total_samples'],
            'successful_attacks': overall_stats['successful_attacks'],
            'failed_attacks': overall_stats['failed_attacks'],
            'success_rate': overall_stats['successful_attacks'] / overall_stats['total_samples'] if overall_stats['total_samples'] > 0 else 0
        }
    }
    
    if overall_stats['all_original_lengths']:
        summary['statistics']['length_statistics'] = {
            'avg_original_length': float(np.mean(overall_stats['all_original_lengths'])),
            'avg_attacked_length': float(np.mean(overall_stats['all_attacked_lengths'])),
            'avg_tokens_affected': float(np.mean(overall_stats['all_tokens_affected'])),
            'std_original_length': float(np.std(overall_stats['all_original_lengths'])),
            'std_attacked_length': float(np.std(overall_stats['all_attacked_lengths'])),
            'std_tokens_affected': float(np.std(overall_stats['all_tokens_affected']))
        }
        
        if attack_type == 'delete':
            summary['statistics']['length_statistics']['actual_deletion_ratio'] = 1 - np.mean(overall_stats['all_attacked_lengths']) / np.mean(overall_stats['all_original_lengths'])
        elif attack_type == 'insert':
            summary['statistics']['length_statistics']['actual_insertion_ratio'] = (np.mean(overall_stats['all_attacked_lengths']) - np.mean(overall_stats['all_original_lengths'])) / np.mean(overall_stats['all_original_lengths'])
    
    summary_path = os.path.join(output_dir, '_attack_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print final summary
    print(f"\n{'='*70}")
    print(f"Attack Summary")
    print(f"{'='*70}")
    print(f"Files processed: {overall_stats['files_processed']}/{len(json_files)}")
    print(f"Total samples: {overall_stats['total_samples']}")
    print(f"Successful attacks: {overall_stats['successful_attacks']}")
    print(f"Failed attacks: {overall_stats['failed_attacks']}")
    
    if overall_stats['all_original_lengths']:
        print(f"\nLength Statistics:")
        print(f"  Average original length: {np.mean(overall_stats['all_original_lengths']):.1f} ± {np.std(overall_stats['all_original_lengths']):.1f}")
        print(f"  Average attacked length: {np.mean(overall_stats['all_attacked_lengths']):.1f} ± {np.std(overall_stats['all_attacked_lengths']):.1f}")
        print(f"  Average tokens affected: {np.mean(overall_stats['all_tokens_affected']):.1f} ± {np.std(overall_stats['all_tokens_affected']):.1f}")
        
        if attack_type == 'delete':
            actual_ratio = 1 - np.mean(overall_stats['all_attacked_lengths']) / np.mean(overall_stats['all_original_lengths'])
            print(f"  Actual deletion ratio: {actual_ratio:.1%}")
        elif attack_type == 'insert':
            actual_ratio = (np.mean(overall_stats['all_attacked_lengths']) - np.mean(overall_stats['all_original_lengths'])) / np.mean(overall_stats['all_original_lengths'])
            print(f"  Actual insertion ratio: {actual_ratio:.1%}")
        elif attack_type in ['swap', 'synonym']:
            print(f"  Actual modification ratio: {np.mean(overall_stats['all_tokens_affected']) / np.mean(overall_stats['all_original_lengths']):.1%}")
    
    print(f"\nOutput directory: {output_dir}")
    print(f"Attack summary saved to: {summary_path}")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(
        description="Apply attacks to watermarked text in JSON files"
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
        help="Output directory (auto-generated with '_attack' suffix if not specified)"
    )
    
    parser.add_argument(
        "--attack",
        type=str,
        default='swap',
        choices=['swap', 'delete', 'insert', 'synonym'],
        help="Type of attack to apply (default: swap)"
    )
    
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.2,
        help="Attack ratio - portion of tokens to affect (default: 0.2 = 20%%)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="GSAI-ML/LLaDA-8B-Instruct",
        help="Model name for tokenizer (default: GSAI-ML/LLaDA-8B-Instruct)"
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
            output_file = os.path.join(dir_path, f"{base_name}_attack.json")
        else:
            output_file = args.output
        
        # Ensure output directory exists
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        print(f"Processing single file: {args.input}")
        print(f"Output: {output_file}")
        
        # Initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        
        stats = process_json_file(
            args.input,
            output_file,
            args.attack,
            args.ratio,
            args.seed,
            args.model,
            tokenizer
        )
        
        print(f"\nAttack complete!")
        print(f"Samples processed: {stats['total_samples']}")
        print(f"Successful attacks: {stats['successful_attacks']}")
        print(f"Output saved to: {output_file}")
        
    elif os.path.isdir(args.input):
        # Directory processing
        process_directory(
            args.input,
            args.output,
            args.attack,
            args.ratio,
            args.seed,
            args.model
        )
    else:
        print(f"Error: '{args.input}' is neither a file nor a directory")


if __name__ == "__main__":
    main()