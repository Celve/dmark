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
from dataclasses import dataclass


@dataclass
class AttackConfig:
    """Configuration for attack operations."""
    attack_type: str = 'swap'
    ratio: float = 0.2
    seed: Optional[int] = None
    min_output_length: int = 200
    model_name: str = "GSAI-ML/LLaDA-8B-Instruct"

    # OpenAI API settings for paraphrase
    api_key: Optional[str] = None
    api_model: str = "gpt-3.5-turbo"
    temperature: float = 0.7
    max_retries: int = 3

    def __post_init__(self):
        """Validate and set up configuration."""
        if self.attack_type == 'paraphrase':
            # Check for API key in environment if not provided
            if self.api_key is None:
                self.api_key = os.getenv("OPENAI_API_KEY")

            if not self.api_key:
                raise ValueError(
                    "OpenAI API key required for paraphrase attack. "
                    "Set OPENAI_API_KEY environment variable or provide api_key."
                )


class BaseAttack:
    """Base class for all attack types."""

    def __init__(self, config: AttackConfig, tokenizer: Any = None):
        self.config = config
        self.tokenizer = tokenizer

    def apply(self, token_ids: List[int], idx: int = 0) -> Tuple[List[int], Dict[str, Any]]:
        """Apply the attack to token IDs.

        Args:
            token_ids: List of token IDs to attack
            idx: Sample index (used for seed variation)

        Returns:
            Tuple of (attacked token IDs, attack metadata)
        """
        raise NotImplementedError

    def get_metadata(self, original_length: int, attacked_length: int, tokens_affected: int) -> Dict[str, Any]:
        """Generate metadata for the attack."""
        return {
            'type': self.config.attack_type,
            'ratio': self.config.ratio if self.config.attack_type != 'paraphrase' else None,
            'original_length': original_length,
            'attacked_length': attacked_length,
            'tokens_affected': tokens_affected,
            'seed': self.config.seed
        }


class SwapAttack(BaseAttack):
    """Randomly swap pairs of tokens."""

    def apply(self, token_ids: List[int], idx: int = 0) -> Tuple[List[int], Dict[str, Any]]:
        if self.config.seed is not None:
            random.seed(self.config.seed + idx)

        attacked_ids = token_ids.copy()
        n_tokens = len(attacked_ids)

        if n_tokens < 2:
            return attacked_ids, self.get_metadata(n_tokens, n_tokens, 0)

        # Calculate number of swaps (each swap affects 2 tokens)
        n_swaps = int(n_tokens * self.config.ratio / 2)

        # Perform swaps
        for _ in range(n_swaps):
            pos1 = random.randint(0, n_tokens - 1)
            pos2 = random.randint(0, n_tokens - 1)

            while pos2 == pos1 and n_tokens > 1:
                pos2 = random.randint(0, n_tokens - 1)

            attacked_ids[pos1], attacked_ids[pos2] = attacked_ids[pos2], attacked_ids[pos1]

        tokens_affected = min(n_swaps * 2, n_tokens)
        return attacked_ids, self.get_metadata(len(token_ids), len(attacked_ids), tokens_affected)


class DeleteAttack(BaseAttack):
    """Randomly delete tokens."""

    def apply(self, token_ids: List[int], idx: int = 0) -> Tuple[List[int], Dict[str, Any]]:
        if self.config.seed is not None:
            random.seed(self.config.seed + idx)

        n_tokens = len(token_ids)

        if n_tokens == 0:
            return token_ids, self.get_metadata(0, 0, 0)

        # Calculate number of tokens to keep
        n_keep = int(n_tokens * (1 - self.config.ratio))

        # Randomly select indices to keep
        all_indices = list(range(n_tokens))
        keep_indices = sorted(random.sample(all_indices, min(n_keep, n_tokens)))

        # Return tokens at kept indices
        attacked_ids = [token_ids[i] for i in keep_indices]
        tokens_affected = n_tokens - len(attacked_ids)

        return attacked_ids, self.get_metadata(n_tokens, len(attacked_ids), tokens_affected)


class InsertAttack(BaseAttack):
    """Randomly insert tokens."""

    def apply(self, token_ids: List[int], idx: int = 0) -> Tuple[List[int], Dict[str, Any]]:
        if self.config.seed is not None:
            random.seed(self.config.seed + idx)

        n_tokens = len(token_ids)
        n_insert = int(n_tokens * self.config.ratio)
        vocab_size = len(self.tokenizer) if self.tokenizer else 126464

        attacked_ids = token_ids.copy()

        # Insert tokens at random positions
        for _ in range(n_insert):
            position = random.randint(0, len(attacked_ids))
            random_token = random.randint(0, vocab_size - 1)
            attacked_ids.insert(position, random_token)

        return attacked_ids, self.get_metadata(n_tokens, len(attacked_ids), n_insert)


class SynonymAttack(BaseAttack):
    """Replace tokens with random tokens."""

    def apply(self, token_ids: List[int], idx: int = 0) -> Tuple[List[int], Dict[str, Any]]:
        if self.config.seed is not None:
            random.seed(self.config.seed + idx)

        attacked_ids = token_ids.copy()
        n_tokens = len(attacked_ids)

        if n_tokens == 0 or self.tokenizer is None:
            return attacked_ids, self.get_metadata(n_tokens, n_tokens, 0)

        vocab_size = len(self.tokenizer)
        n_replace = int(n_tokens * self.config.ratio)

        # Randomly select positions to replace
        positions = random.sample(range(n_tokens), min(n_replace, n_tokens))

        # Replace tokens at selected positions
        for pos in positions:
            attacked_ids[pos] = random.randint(0, vocab_size - 1)

        return attacked_ids, self.get_metadata(n_tokens, len(attacked_ids), n_replace)


class ParaphraseAttack(BaseAttack):
    """Paraphrase text using OpenAI API."""

    def apply(self, token_ids: List[int], idx: int = 0) -> Tuple[List[int], Dict[str, Any]]:
        if self.tokenizer is None:
            raise ValueError("Tokenizer is required for paraphrase attack")

        try:
            import openai
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package not installed. Install with: pip install openai")

        # Initialize OpenAI client
        client = OpenAI(api_key=self.config.api_key)

        # Decode tokens to text
        original_text = self.tokenizer.decode(token_ids, skip_special_tokens=True)

        if not original_text.strip():
            return token_ids, self.get_metadata(len(token_ids), len(token_ids), 0)

        # Create paraphrasing prompt
        prompt = f"""Please paraphrase the following text while preserving its meaning.
Make the paraphrase natural and fluent but different from the original:

Text: {original_text}

Paraphrased version:"""

        # Try to get paraphrase with retries
        for attempt in range(self.config.max_retries):
            try:
                response = client.responses.create(
                    model="gpt-5",
                    input=[
                        {"role": "system", "content": "You are a helpful assistant that paraphrases text while preserving meaning."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=self.config.temperature,
                    max_output_tokens=1024,  # or another safe cap
                )

                paraphrased_text = response.output_text
                paraphrased_ids = self.tokenizer.encode(paraphrased_text, add_special_tokens=False)

                # Calculate tokens affected (approximation for paraphrase)
                tokens_affected = abs(len(paraphrased_ids) - len(token_ids)) + min(len(paraphrased_ids), len(token_ids))

                metadata = self.get_metadata(len(token_ids), len(paraphrased_ids), tokens_affected)
                metadata['api_model'] = self.config.api_model
                metadata['temperature'] = self.config.temperature

                return paraphrased_ids, metadata

            except Exception as e:
                if attempt < self.config.max_retries - 1:
                    wait_time = (2 ** attempt) + random.uniform(0, 1)
                    print(f"  API error (attempt {attempt + 1}/{self.config.max_retries}): {e}. Retrying in {wait_time:.1f}s...")
                    time.sleep(wait_time)
                else:
                    print(f"  Failed to paraphrase after {self.config.max_retries} attempts: {e}")
                    return token_ids, self.get_metadata(len(token_ids), len(token_ids), 0)

        return token_ids, self.get_metadata(len(token_ids), len(token_ids), 0)


class AttackFactory:
    """Factory for creating attack instances."""

    ATTACK_CLASSES = {
        'swap': SwapAttack,
        'delete': DeleteAttack,
        'insert': InsertAttack,
        'synonym': SynonymAttack,
        'paraphrase': ParaphraseAttack
    }

    @classmethod
    def create(cls, config: AttackConfig, tokenizer: Any = None) -> BaseAttack:
        """Create an attack instance based on configuration."""
        attack_class = cls.ATTACK_CLASSES.get(config.attack_type)
        if not attack_class:
            raise ValueError(f"Unknown attack type: {config.attack_type}")
        return attack_class(config, tokenizer)


class AttackProcessor:
    """Main processor for applying attacks to JSON files."""

    def __init__(self, config: AttackConfig):
        self.config = config
        self.tokenizer = None
        self.attack = None

    def initialize(self):
        """Initialize tokenizer and attack instance."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name, trust_remote_code=True)
        self.attack = AttackFactory.create(self.config, self.tokenizer)

    def truncate_tokens(self, token_ids: List[int]) -> Tuple[List[int], bool, int]:
        """Truncate tokens to min_output_length if needed."""
        original_length = len(token_ids)
        if original_length > self.config.min_output_length:
            return token_ids[:self.config.min_output_length], True, original_length
        return token_ids, False, original_length

    def process_sample(self, result: dict, idx: int) -> dict:
        """Process a single sample."""
        attacked_result = result.copy()

        # Get output IDs - prioritize truncated_output_ids if available
        if 'data' not in result:
            return attacked_result

        output_ids = None
        source_field = None

        if 'truncated_output_ids' in result['data']:
            output_ids = result['data']['truncated_output_ids']
            source_field = 'truncated_output_ids'
        elif 'output_ids' in result['data']:
            output_ids = result['data']['output_ids']
            source_field = 'output_ids'

        if output_ids is None:
            return attacked_result

        # Truncate to min_output_length before attack
        output_ids, truncation_applied, original_full_length = self.truncate_tokens(output_ids)

        # Apply attack
        attacked_ids, attack_metadata = self.attack.apply(output_ids, idx)

        # Decode attacked IDs to get text
        attacked_text = self.tokenizer.decode(attacked_ids, skip_special_tokens=True)

        # Add attacked data to result
        attacked_result['data']['attacked_ids'] = attacked_ids
        attacked_result['data']['attacked_text'] = attacked_text

        # Add metadata
        attack_metadata['source_field'] = source_field
        attack_metadata['pre_attack_truncation'] = {
            'applied': truncation_applied,
            'original_full_length': original_full_length,
            'truncated_to': len(output_ids) if truncation_applied else original_full_length,
            'min_output_length': self.config.min_output_length
        }

        attacked_result['attack_metadata'] = attack_metadata

        return attacked_result

    def process_file(self, input_file: str, output_file: str) -> Dict[str, Any]:
        """Process a JSON file and apply attacks."""
        with open(input_file, 'r') as f:
            results = json.load(f)

        attacked_results = []
        stats = {
            'total_samples': len(results),
            'successful_attacks': 0,
            'failed_attacks': 0,
            'original_lengths': [],
            'attacked_lengths': [],
            'tokens_affected': []
        }

        for idx, result in enumerate(results):
            try:
                attacked_result = self.process_sample(result, idx)

                # Update statistics if attack was successful
                if 'attack_metadata' in attacked_result:
                    metadata = attacked_result['attack_metadata']
                    stats['successful_attacks'] += 1
                    stats['original_lengths'].append(metadata['original_length'])
                    stats['attacked_lengths'].append(metadata['attacked_length'])
                    stats['tokens_affected'].append(metadata['tokens_affected'])
                else:
                    stats['failed_attacks'] += 1

                attacked_results.append(attacked_result)

            except Exception as e:
                print(f"  Warning: Failed to attack sample {idx}: {e}")
                stats['failed_attacks'] += 1
                attacked_results.append(result)

        # Save results
        with open(output_file, 'w') as f:
            json.dump(attacked_results, f, indent=4)

        return stats

    def process_directory(self, input_dir: str, output_dir: str) -> None:
        """Process all JSON files in a directory."""
        # Auto-generate output directory name if not provided
        if output_dir is None:
            base_dir = os.path.dirname(input_dir.rstrip('/'))
            dir_name = os.path.basename(input_dir.rstrip('/'))
            ratio_percent = int(self.config.ratio * 100)
            output_dir = os.path.join(base_dir, f"{dir_name}_attack_{self.config.attack_type}_{ratio_percent}")

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Find all JSON files
        json_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.json')])

        if not json_files:
            print(f"No JSON files found in {input_dir}")
            return

        self.print_configuration(input_dir, output_dir, len(json_files))

        # Initialize once for all files
        print("Loading tokenizer...")
        self.initialize()

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
            output_path = os.path.join(output_dir, json_file)

            try:
                stats = self.process_file(input_path, output_path)

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

        # Save and print summary
        self.save_summary(output_dir, input_dir, overall_stats)
        self.print_summary(overall_stats, len(json_files), output_dir)

    def print_configuration(self, input_dir: str, output_dir: str, num_files: int):
        """Print attack configuration."""
        print(f"\n{'='*70}")
        print(f"Attack Configuration")
        print(f"{'='*70}")
        print(f"Input directory: {input_dir}")
        print(f"Output directory: {output_dir}")
        print(f"Attack type: {self.config.attack_type}")

        if self.config.attack_type != 'paraphrase':
            print(f"Attack ratio: {self.config.ratio:.1%}")
        else:
            print(f"API model: {self.config.api_model}")
            print(f"Temperature: {self.config.temperature}")

        print(f"Min output length (truncate before attack): {self.config.min_output_length}")
        print(f"Files to process: {num_files}")
        print(f"Random seed: {self.config.seed}")
        print(f"Model: {self.config.model_name}")
        print(f"{'='*70}\n")

    def save_summary(self, output_dir: str, input_dir: str, stats: Dict[str, Any]):
        """Save attack summary to JSON file."""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'input_directory': input_dir,
            'output_directory': output_dir,
            'attack_configuration': {
                'type': self.config.attack_type,
                'ratio': self.config.ratio if self.config.attack_type != 'paraphrase' else None,
                'seed': self.config.seed,
                'model': self.config.model_name,
                'min_output_length': self.config.min_output_length,
                'api_model': self.config.api_model if self.config.attack_type == 'paraphrase' else None,
                'temperature': self.config.temperature if self.config.attack_type == 'paraphrase' else None
            },
            'statistics': {
                'files_processed': stats['files_processed'],
                'total_samples': stats['total_samples'],
                'successful_attacks': stats['successful_attacks'],
                'failed_attacks': stats['failed_attacks'],
                'success_rate': stats['successful_attacks'] / stats['total_samples'] if stats['total_samples'] > 0 else 0
            }
        }

        if stats['all_original_lengths']:
            summary['statistics']['length_statistics'] = {
                'avg_original_length': float(np.mean(stats['all_original_lengths'])),
                'avg_attacked_length': float(np.mean(stats['all_attacked_lengths'])),
                'avg_tokens_affected': float(np.mean(stats['all_tokens_affected'])),
                'std_original_length': float(np.std(stats['all_original_lengths'])),
                'std_attacked_length': float(np.std(stats['all_attacked_lengths'])),
                'std_tokens_affected': float(np.std(stats['all_tokens_affected']))
            }

            if self.config.attack_type == 'delete':
                summary['statistics']['length_statistics']['actual_deletion_ratio'] = \
                    1 - np.mean(stats['all_attacked_lengths']) / np.mean(stats['all_original_lengths'])
            elif self.config.attack_type == 'insert':
                summary['statistics']['length_statistics']['actual_insertion_ratio'] = \
                    (np.mean(stats['all_attacked_lengths']) - np.mean(stats['all_original_lengths'])) / np.mean(stats['all_original_lengths'])

        summary_path = os.path.join(output_dir, '_attack_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

    def print_summary(self, stats: Dict[str, Any], num_files: int, output_dir: str):
        """Print final summary."""
        print(f"\n{'='*70}")
        print(f"Attack Summary")
        print(f"{'='*70}")
        print(f"Files processed: {stats['files_processed']}/{num_files}")
        print(f"Total samples: {stats['total_samples']}")
        print(f"Successful attacks: {stats['successful_attacks']}")
        print(f"Failed attacks: {stats['failed_attacks']}")

        if stats['all_original_lengths']:
            print(f"\nLength Statistics:")
            print(f"  Average original length: {np.mean(stats['all_original_lengths']):.1f} ± {np.std(stats['all_original_lengths']):.1f}")
            print(f"  Average attacked length: {np.mean(stats['all_attacked_lengths']):.1f} ± {np.std(stats['all_attacked_lengths']):.1f}")
            print(f"  Average tokens affected: {np.mean(stats['all_tokens_affected']):.1f} ± {np.std(stats['all_tokens_affected']):.1f}")

            if self.config.attack_type == 'delete':
                actual_ratio = 1 - np.mean(stats['all_attacked_lengths']) / np.mean(stats['all_original_lengths'])
                print(f"  Actual deletion ratio: {actual_ratio:.1%}")
            elif self.config.attack_type == 'insert':
                actual_ratio = (np.mean(stats['all_attacked_lengths']) - np.mean(stats['all_original_lengths'])) / np.mean(stats['all_original_lengths'])
                print(f"  Actual insertion ratio: {actual_ratio:.1%}")
            elif self.config.attack_type in ['swap', 'synonym']:
                print(f"  Actual modification ratio: {np.mean(stats['all_tokens_affected']) / np.mean(stats['all_original_lengths']):.1%}")

        print(f"\nOutput directory: {output_dir}")
        print(f"Attack summary saved to: {os.path.join(output_dir, '_attack_summary.json')}")
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
        default="gpt-5",
        help="OpenAI model to use for paraphrase attack (default: gpt-5)"
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=1,
        help="Temperature for paraphrase generation (default: 0.7)"
    )

    args = parser.parse_args()

    # Check if input exists
    if not os.path.exists(args.input):
        print(f"Error: Input path '{args.input}' not found")
        return

    # Create configuration
    try:
        config = AttackConfig(
            attack_type=args.attack,
            ratio=args.ratio,
            seed=args.seed,
            min_output_length=args.min_output_length,
            model_name=args.model,
            api_key=args.api_key,
            api_model=args.api_model,
            temperature=args.temperature
        )
    except ValueError as e:
        print(f"Configuration error: {e}")
        return

    # Create processor
    processor = AttackProcessor(config)

    # Process based on input type
    if os.path.isfile(args.input):
        # Single file processing
        if args.output is None:
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

        # Process file
        processor.initialize()
        stats = processor.process_file(args.input, output_file)

        print(f"\nAttack complete!")
        print(f"Samples processed: {stats['total_samples']}")
        print(f"Successful attacks: {stats['successful_attacks']}")
        print(f"Output saved to: {output_file}")

    elif os.path.isdir(args.input):
        # Directory processing
        processor.process_directory(args.input, args.output)
    else:
        print(f"Error: '{args.input}' is neither a file nor a directory")


if __name__ == "__main__":
    main()