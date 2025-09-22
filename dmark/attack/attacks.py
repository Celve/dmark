import argparse
import json
import os
import random
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer
from datetime import datetime
import time


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


def paraphrase_attack(
    token_ids: List[int],
    tokenizer: Any,
    api_key: str = None,
    model: str = "gpt-3.5-turbo",
    temperature: float = 0.7,
    max_retries: int = 3
) -> List[int]:
    """
    Paraphrase text using OpenAI API.

    Args:
        token_ids: List of token IDs to paraphrase
        tokenizer: Tokenizer for text conversion
        api_key: OpenAI API key (if None, uses environment variable)
        model: OpenAI model to use
        temperature: Temperature for generation (higher = more creative)
        max_retries: Maximum number of retry attempts on API failure

    Returns:
        List of token IDs after paraphrasing
    """
    try:
        import openai
    except ImportError:
        raise ImportError("openai package not installed. Install with: pip install openai")

    # Get API key from environment if not provided
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY environment variable or pass api_key parameter.")

    # Initialize OpenAI client
    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    # Decode tokens to text
    original_text = tokenizer.decode(token_ids, skip_special_tokens=True)

    if not original_text.strip():
        return token_ids

    # Create paraphrasing prompt
    prompt = f"""Please paraphrase the following text while preserving its meaning.
Make the paraphrase natural and fluent but different from the original:

Text: {original_text}

Paraphrased version:"""

    # Try to get paraphrase with retries
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that paraphrases text while preserving meaning."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=len(token_ids) * 2,  # Allow some expansion
                n=1
            )

            paraphrased_text = response.choices[0].message.content.strip()

            # Encode paraphrased text back to tokens
            paraphrased_ids = tokenizer.encode(paraphrased_text, add_special_tokens=False)

            return paraphrased_ids

        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                print(f"  API error (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {wait_time:.1f}s...")
                time.sleep(wait_time)
            else:
                print(f"  Failed to paraphrase after {max_retries} attempts: {e}")
                # Return original tokens on failure
                return token_ids

    return token_ids


def apply_attack(
    token_ids: List[int],
    attack_type: str,
    ratio: float,
    seed: int = None,
    tokenizer: Any = None,
    api_key: str = None,
    api_model: str = "gpt-3.5-turbo",
    temperature: float = 0.7
) -> Tuple[List[int], Dict[str, Any]]:
    """
    Apply the specified attack to token IDs.

    Args:
        token_ids: List of token IDs to attack
        attack_type: Type of attack to apply
        ratio: Attack ratio
        seed: Random seed
        tokenizer: Tokenizer (needed for some attacks)
        api_key: OpenAI API key (for paraphrase attack)
        api_model: OpenAI model to use (for paraphrase attack)
        temperature: Temperature for paraphrase generation

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
    elif attack_type == 'paraphrase':
        if tokenizer is None:
            raise ValueError("Tokenizer is required for paraphrase attack")
        attacked_ids = paraphrase_attack(
            token_ids,
            tokenizer=tokenizer,
            api_key=api_key,
            model=api_model,
            temperature=temperature
        )
        # For paraphrase, all tokens are potentially affected
        tokens_affected = abs(len(attacked_ids) - original_length) + min(len(attacked_ids), original_length)
    else:
        raise ValueError(f"Unknown attack type: {attack_type}")

    metadata = {
        'type': attack_type,
        'ratio': ratio if attack_type != 'paraphrase' else None,
        'original_length': original_length,
        'attacked_length': len(attacked_ids),
        'tokens_affected': tokens_affected,
        'seed': seed
    }

    # Add paraphrase-specific metadata
    if attack_type == 'paraphrase':
        metadata['api_model'] = api_model
        metadata['temperature'] = temperature

    return attacked_ids, metadata


def process_json_file(
    input_file: str,
    output_file: str,
    attack_type: str,
    ratio: float = 0.2,
    seed: int = None,
    model_name: str = "GSAI-ML/LLaDA-8B-Instruct",
    tokenizer: Optional[Any] = None,
    min_output_length: int = 200,
    api_key: str = None,
    api_model: str = "gpt-3.5-turbo",
    temperature: float = 0.7
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
        min_output_length: Minimum output length to truncate before attack (default: 200)
        api_key: OpenAI API key for paraphrase attack
        api_model: OpenAI model for paraphrase attack
        temperature: Temperature for paraphrase generation

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
                    # Truncate to min_output_length before attack
                    original_full_length = len(output_ids)
                    if len(output_ids) > min_output_length:
                        output_ids = output_ids[:min_output_length]
                        truncation_applied = True
                    else:
                        truncation_applied = False

                    # Apply attack
                    attacked_ids, attack_metadata = apply_attack(
                        output_ids,
                        attack_type,
                        ratio,
                        seed=seed + idx if seed is not None else None,
                        tokenizer=tokenizer,
                        api_key=api_key,
                        api_model=api_model,
                        temperature=temperature
                    )

                    # Decode attacked IDs to get text
                    attacked_text = tokenizer.decode(attacked_ids, skip_special_tokens=True)

                    # Add new fields for attacked data (don't replace original fields)
                    attacked_result['data']['attacked_ids'] = attacked_ids
                    attacked_result['data']['attacked_text'] = attacked_text

                    # Track which field was used as source
                    if 'truncated_output_ids' in result['data']:
                        attack_metadata['source_field'] = 'truncated_output_ids'
                    else:
                        attack_metadata['source_field'] = 'output_ids'

                    # Add truncation info to metadata
                    attack_metadata['pre_attack_truncation'] = {
                        'applied': truncation_applied,
                        'original_full_length': original_full_length,
                        'truncated_to': len(output_ids) if truncation_applied else original_full_length,
                        'min_output_length': min_output_length
                    }

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
    model_name: str = "GSAI-ML/LLaDA-8B-Instruct",
    min_output_length: int = 200,
    api_key: str = None,
    api_model: str = "gpt-3.5-turbo",
    temperature: float = 0.7
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
        min_output_length: Minimum output length to truncate before attack (default: 200)
        api_key: OpenAI API key for paraphrase attack
        api_model: OpenAI model for paraphrase attack
        temperature: Temperature for paraphrase generation
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
    if attack_type != 'paraphrase':
        print(f"Attack ratio: {ratio:.1%}")
    else:
        print(f"API model: {api_model}")
        print(f"Temperature: {temperature}")
    print(f"Min output length (truncate before attack): {min_output_length}")
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
                tokenizer,
                min_output_length,
                api_key,
                api_model,
                temperature
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
            'ratio': ratio if attack_type != 'paraphrase' else None,
            'seed': seed,
            'model': model_name,
            'min_output_length': min_output_length,
            'api_model': api_model if attack_type == 'paraphrase' else None,
            'temperature': temperature if attack_type == 'paraphrase' else None
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
        choices=['swap', 'delete', 'insert', 'synonym', 'paraphrase'],
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

    parser.add_argument(
        "--min-output-length",
        type=int,
        default=200,
        help="Minimum output length to truncate to before applying attack (default: 200)"
    )

    # OpenAI API arguments for paraphrase attack
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="OpenAI API key (can also be set via OPENAI_API_KEY environment variable)"
    )

    parser.add_argument(
        "--api-model",
        type=str,
        default="gpt-3.5-turbo",
        help="OpenAI model to use for paraphrase attack (default: gpt-3.5-turbo)"
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for paraphrase generation (default: 0.7)"
    )

    args = parser.parse_args()

    # Check if input exists
    if not os.path.exists(args.input):
        print(f"Error: Input path '{args.input}' not found")
        return

    # Check API key for paraphrase attack
    if args.attack == 'paraphrase':
        if not args.api_key and not os.getenv("OPENAI_API_KEY"):
            print(f"Error: OpenAI API key required for paraphrase attack.")
            print(f"Either set OPENAI_API_KEY environment variable or use --api-key argument")
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
            tokenizer,
            args.min_output_length,
            args.api_key,
            args.api_model,
            args.temperature
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
            args.model,
            args.min_output_length,
            args.api_key,
            args.api_model,
            args.temperature
        )
    else:
        print(f"Error: '{args.input}' is neither a file nor a directory")


if __name__ == "__main__":
    main()