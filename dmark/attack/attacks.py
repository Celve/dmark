import argparse
import json
import os
import random
from typing import List, Dict, Any
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer


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


def process_json_file(
    input_file: str,
    output_file: str,
    attack_type: str,
    ratio: float = 0.2,
    seed: int = None,
    model_name: str = "GSAI-ML/LLaDA-8B-Instruct"
) -> None:
    """
    Process a JSON file and apply the specified attack to output_ids and output text.
    
    Args:
        input_file: Path to input JSON file
        output_file: Path to output JSON file
        attack_type: Type of attack ('swap' or 'delete')
        ratio: Attack ratio (default: 0.2)
        seed: Random seed for reproducibility
        model_name: Model name for tokenizer
    """
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    with open(input_file, 'r') as f:
        results = json.load(f)
    
    attacked_results = []
    
    for idx, result in enumerate(tqdm(results, desc=f"Applying {attack_type} attack")):
        # Create a copy of the result
        attacked_result = result.copy()
        
        # Get output IDs
        if 'data' in result and 'output_ids' in result['data']:
            output_ids = result['data']['output_ids']
            
            # Apply attack based on type
            if attack_type == 'swap':
                attacked_ids = random_swap_attack(
                    output_ids, 
                    ratio=ratio, 
                    seed=seed + idx if seed is not None else None
                )
            elif attack_type == 'delete':
                attacked_ids = random_delete_attack(
                    output_ids, 
                    ratio=ratio, 
                    seed=seed + idx if seed is not None else None
                )
            else:
                raise ValueError(f"Unknown attack type: {attack_type}")
            
            # Update the output_ids
            attacked_result['data']['output_ids'] = attacked_ids
            
            # Decode attacked IDs to get new output text
            attacked_text = tokenizer.decode(attacked_ids, skip_special_tokens=True)
            
            # Update the output text field
            attacked_result['data']['output'] = attacked_text
            
            # Store original output for reference if it exists
            if 'output' in result['data']:
                attacked_result['data']['original_output'] = result['data']['output']
            
            # Add attack metadata
            attacked_result['attack_metadata'] = {
                'type': attack_type,
                'ratio': ratio,
                'original_length': len(output_ids),
                'attacked_length': len(attacked_ids),
                'tokens_affected': abs(len(output_ids) - len(attacked_ids)) if attack_type == 'delete' else int(len(output_ids) * ratio),
                'seed': seed + idx if seed is not None else None
            }
        
        attacked_results.append(attacked_result)
    
    # Save results
    with open(output_file, 'w') as f:
        json.dump(attacked_results, f, indent=4)
    
    print(f"Attacked results saved to: {output_file}")
    
    # Print statistics
    original_lengths = [len(r['data']['output_ids']) for r in results if 'data' in r and 'output_ids' in r['data']]
    attacked_lengths = [r['attack_metadata']['attacked_length'] for r in attacked_results if 'attack_metadata' in r]
    
    if original_lengths and attacked_lengths:
        print(f"\nStatistics:")
        print(f"  Average original length: {np.mean(original_lengths):.1f}")
        print(f"  Average attacked length: {np.mean(attacked_lengths):.1f}")
        if attack_type == 'delete':
            print(f"  Average deletion ratio: {1 - np.mean(attacked_lengths) / np.mean(original_lengths):.2%}")
        elif attack_type == 'swap':
            tokens_swapped = [r['attack_metadata']['tokens_affected'] for r in attacked_results if 'attack_metadata' in r]
            print(f"  Average tokens swapped: {np.mean(tokens_swapped):.1f}")


def process_directory(
    input_dir: str,
    output_dir: str,
    attack_type: str,
    ratio: float = 0.2,
    seed: int = None,
    model_name: str = "GSAI-ML/LLaDA-8B-Instruct"
) -> None:
    """
    Process all JSON files in a directory.
    
    Args:
        input_dir: Directory containing input JSON files
        output_dir: Directory to save attacked JSON files
        attack_type: Type of attack ('swap' or 'delete')
        ratio: Attack ratio (default: 0.2)
        seed: Random seed for reproducibility
        model_name: Model name for tokenizer
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all JSON files
    json_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
    
    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return
    
    print(f"Found {len(json_files)} JSON files to process")
    
    for json_file in json_files:
        input_path = os.path.join(input_dir, json_file)
        
        # Create output filename with attack suffix
        base_name = json_file.replace('.json', '')
        output_name = f"{base_name}_{attack_type}_attack_{int(ratio*100)}.json"
        output_path = os.path.join(output_dir, output_name)
        
        print(f"\nProcessing: {json_file}")
        process_json_file(input_path, output_path, attack_type, ratio, seed, model_name)


def main():
    parser = argparse.ArgumentParser(description="Apply attacks to watermarked text")
    
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input directory or JSON file"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory or JSON file"
    )
    
    parser.add_argument(
        "--attack",
        type=str,
        required=True,
        choices=['swap', 'delete'],
        help="Type of attack to apply"
    )
    
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.2,
        help="Attack ratio (default: 0.2 = 20%%)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
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
        # Process single file
        if os.path.isdir(args.output):
            # If output is directory, create filename
            base_name = os.path.basename(args.input).replace('.json', '')
            output_file = os.path.join(
                args.output, 
                f"{base_name}_{args.attack}_attack_{int(args.ratio*100)}.json"
            )
        else:
            output_file = args.output
        
        # Ensure output directory exists
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        process_json_file(args.input, output_file, args.attack, args.ratio, args.seed, args.model)
        
    elif os.path.isdir(args.input):
        # Process directory
        if not os.path.isdir(args.output):
            # Create output directory if it doesn't exist
            os.makedirs(args.output, exist_ok=True)
        
        process_directory(args.input, args.output, args.attack, args.ratio, args.seed, args.model)
    else:
        print(f"Error: '{args.input}' is neither a file nor a directory")


if __name__ == "__main__":
    main()