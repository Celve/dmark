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
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


@dataclass
class AttackConfig:
    """Configuration for attack operations."""
    attack_type: str = 'swap'
    ratio: float = 0.2
    seed: Optional[int] = None
    min_output_length: int = 200
    model_name: str = "GSAI-ML/LLaDA-8B-Instruct"

    # Increment mode - skip if results already exist
    increment_mode: bool = False

    # API settings for paraphrase
    api_provider: str = "openai"  # "openai" or "gemini"
    api_base: Optional[str] = None  
    api_key: Optional[str] = None
    api_model: str = "gpt-3.5-turbo"  # Or "gemini-pro" for Gemini
    temperature: float = 0.7
    max_retries: int = 3
    max_concurrent: int = 10  # Maximum concurrent paraphrase tasks

    def __post_init__(self):
        """Validate and set up configuration."""
        if self.attack_type == 'paraphrase':
            # Check for API key in environment if not provided
            if self.api_key is None:
                if self.api_provider == "openai":
                    self.api_key = os.getenv("OPENAI_API_KEY")
                elif self.api_provider == "gemini":
                    self.api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

            if not self.api_key:
                provider_name = self.api_provider.upper()
                env_var = "OPENAI_API_KEY" if self.api_provider == "openai" else "GEMINI_API_KEY or GOOGLE_API_KEY"
                raise ValueError(
                    f"{provider_name} API key required for paraphrase attack. "
                    f"Set {env_var} environment variable or provide api_key."
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
    """Paraphrase text using OpenAI or Gemini API."""

    def apply(self, token_ids: List[int], idx: int = 0) -> Tuple[List[int], Dict[str, Any]]:
        if self.tokenizer is None:
            raise ValueError("Tokenizer is required for paraphrase attack")

        # Decode tokens to text
        original_text = self.tokenizer.decode(token_ids, skip_special_tokens=True)

        if not original_text.strip():
            return token_ids, self.get_metadata(len(token_ids), len(token_ids), 0)

        # Route to appropriate API
        if self.config.api_provider == "openai":
            return self._apply_openai(token_ids, original_text, idx)
        elif self.config.api_provider == "gemini":
            return self._apply_gemini(token_ids, original_text, idx)
        else:
            raise ValueError(f"Unsupported API provider: {self.config.api_provider}")

    def _apply_openai(self, token_ids: List[int], original_text: str, idx: int) -> Tuple[List[int], Dict[str, Any]]:
        """Apply paraphrase using OpenAI API."""
        try:
            import openai
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package not installed. Install with: pip install openai")

        # Initialize OpenAI client
        if self.config.api_base is None:
            client = OpenAI(api_key=self.config.api_key)
        else:
            client = OpenAI(api_key=self.config.api_key, api_base=self.config.api_base)

        # Create paraphrasing prompt
        prompt = f"""Please paraphrase the following text while preserving its meaning. Output only the rewritten text, nothing else:

{original_text}"""

        # Try to get paraphrase with retries
        for attempt in range(self.config.max_retries):
            try:
                if self.config.api_model.startswith("gpt"):
                    response = client.responses.create(
                        model=self.config.api_model,
                        input=[
                            {"role": "system", "content": "You are a paraphrasing assistant. Output only the paraphrased text without any additional comments, explanations, or formatting."},
                            {"role": "user", "content": prompt},
                        ],
                        temperature=self.config.temperature,
                    )
                else: 
                    response = client.chat.completions.create(
                        model=self.config.api_model,
                        messages=[
                            {"role": "system", "content": "You are a paraphrasing assistant. Output only the paraphrased text without any additional comments, explanations, or formatting."},
                            {"role": "user", "content": prompt},
                        ]
                    )

                paraphrased_text = response.output_text
                paraphrased_ids = self.tokenizer.encode(paraphrased_text, add_special_tokens=False)

                # Calculate tokens affected (approximation for paraphrase)
                tokens_affected = abs(len(paraphrased_ids) - len(token_ids)) + min(len(paraphrased_ids), len(token_ids))

                metadata = self.get_metadata(len(token_ids), len(paraphrased_ids), tokens_affected)
                metadata['api_provider'] = self.config.api_provider
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

    def _apply_gemini(self, token_ids: List[int], original_text: str, idx: int) -> Tuple[List[int], Dict[str, Any]]:
        """Apply paraphrase using Gemini API."""
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            raise ImportError("google-generativeai package not installed. Install with: pip install google-generativeai")

        # Initialize model
        client = genai.Client()

        # Create paraphrasing prompt
        prompt = f"""Please paraphrase the following text while preserving its meaning. Output only the rewritten text, nothing else:

{original_text}"""

        # Try to get paraphrase with retries
        for attempt in range(self.config.max_retries):
            try:
                response = client.models.generate_content(
                    model=self.config.api_model,
                    config=types.GenerateContentConfig(
                        system_instruction="You are a paraphrasing assistant. Output only the paraphrased text without any additional comments, explanations, or formatting."
                    ),
                    contents=prompt
                )

                paraphrased_text = response.text.strip()
                paraphrased_ids = self.tokenizer.encode(paraphrased_text, add_special_tokens=False)

                # Calculate tokens affected (approximation for paraphrase)
                tokens_affected = abs(len(paraphrased_ids) - len(token_ids)) + min(len(paraphrased_ids), len(token_ids))

                metadata = self.get_metadata(len(token_ids), len(paraphrased_ids), tokens_affected)
                metadata['api_provider'] = self.config.api_provider
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

    def process_sample_concurrent(self, args):
        """Process a sample for concurrent execution."""
        result, idx = args
        try:
            return self.process_sample(result, idx)
        except Exception as e:
            print(f"  Warning: Failed to attack sample {idx}: {e}")
            return result

    def process_file(self, input_file: str, output_file: str) -> Dict[str, Any]:
        """Process a JSON file and apply attacks."""
        # Check if output already exists in increment mode
        if self.config.increment_mode and os.path.exists(output_file):
            print(f"  → Skipping (already exists): {output_file}")

            # Try to load existing results to get statistics
            try:
                with open(output_file, 'r') as f:
                    existing_results = json.load(f)

                # Extract statistics from existing results
                stats = {
                    'total_samples': len(existing_results),
                    'successful_attacks': 0,
                    'failed_attacks': 0,
                    'original_lengths': [],
                    'attacked_lengths': [],
                    'tokens_affected': [],
                    'skipped': True
                }

                for result in existing_results:
                    if 'attack_metadata' in result:
                        metadata = result['attack_metadata']
                        stats['successful_attacks'] += 1
                        stats['original_lengths'].append(metadata.get('original_length', 0))
                        stats['attacked_lengths'].append(metadata.get('attacked_length', 0))
                        stats['tokens_affected'].append(metadata.get('tokens_affected', 0))
                    else:
                        stats['failed_attacks'] += 1

                return stats
            except Exception as e:
                print(f"    Warning: Could not read existing results: {e}")
                # Continue with processing if we can't read existing results

        with open(input_file, 'r') as f:
            results = json.load(f)

        stats = {
            'total_samples': len(results),
            'successful_attacks': 0,
            'failed_attacks': 0,
            'original_lengths': [],
            'attacked_lengths': [],
            'tokens_affected': []
        }

        # Use concurrent processing for paraphrase attacks
        if self.config.attack_type == 'paraphrase' and len(results) > 1:
            attacked_results = self._process_concurrent(results, stats)
        else:
            # Sequential processing for other attacks or single sample
            attacked_results = self._process_sequential(results, stats)

        # Save results
        with open(output_file, 'w') as f:
            json.dump(attacked_results, f, indent=4)

        return stats

    def _process_sequential(self, results: List[dict], stats: Dict[str, Any]) -> List[dict]:
        """Process samples sequentially."""
        attacked_results = []

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

        return attacked_results

    def _process_concurrent(self, results: List[dict], stats: Dict[str, Any]) -> List[dict]:
        """Process samples concurrently using thread pool."""
        attacked_results = [None] * len(results)
        stats_lock = threading.Lock()  # Thread safety for statistics

        # Prepare tasks
        tasks = [(result, idx) for idx, result in enumerate(results)]

        # Process with progress bar
        with ThreadPoolExecutor(max_workers=self.config.max_concurrent) as executor:
            # Submit all tasks
            future_to_idx = {
                executor.submit(self.process_sample_concurrent, task): task[1]
                for task in tasks
            }

            # Process completed futures with progress bar
            with tqdm(total=len(tasks), desc=f"Paraphrasing (concurrent={self.config.max_concurrent})", leave=False) as pbar:
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]

                    try:
                        attacked_result = future.result()
                        attacked_results[idx] = attacked_result

                        # Update statistics with thread safety
                        with stats_lock:
                            if 'attack_metadata' in attacked_result:
                                metadata = attacked_result['attack_metadata']
                                stats['successful_attacks'] += 1
                                stats['original_lengths'].append(metadata['original_length'])
                                stats['attacked_lengths'].append(metadata['attacked_length'])
                                stats['tokens_affected'].append(metadata['tokens_affected'])
                            else:
                                stats['failed_attacks'] += 1

                    except Exception as e:
                        print(f"  Warning: Failed to attack sample {idx}: {e}")
                        with stats_lock:
                            stats['failed_attacks'] += 1
                        attacked_results[idx] = results[idx]

                    pbar.update(1)
                    pbar.set_postfix({
                        '✅': stats['successful_attacks'],
                        '❌': stats['failed_attacks']
                    })

        return attacked_results

    def process_directory(self, input_dir: str, output_dir: str) -> None:
        """Process all JSON files in a directory."""
        # Auto-generate output directory name if not provided
        if output_dir is None:
            base_dir = os.path.dirname(input_dir.rstrip('/'))
            dir_name = os.path.basename(input_dir.rstrip('/'))

            # Generate directory suffix
            if self.config.attack_type == 'paraphrase':
                # For paraphrase, use model name (sanitize it for filename)
                model_suffix = self.config.api_model.replace('/', '_').replace('\\', '_')
                suffix = f"_attack_paraphrase_{model_suffix}"
            else:
                # For other attacks, use type and ratio
                ratio_percent = int(self.config.ratio * 100)
                suffix = f"_attack_{self.config.attack_type}_{ratio_percent}"

            output_dir = os.path.join(base_dir, f"{dir_name}{suffix}")

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

        # Count files to skip in increment mode
        skipped_count = 0
        processed_count = 0

        # Process each file
        for json_file in tqdm(json_files, desc="Processing files"):
            input_path = os.path.join(input_dir, json_file)

            # Generate file suffix
            if self.config.attack_type == 'paraphrase':
                # For paraphrase, use model name (sanitize it for filename)
                model_suffix = self.config.api_model.replace('/', '_').replace('\\', '_')
                suffix = f"_attack_paraphrase_{model_suffix}"
            else:
                # For other attacks, use type and ratio
                ratio_percent = int(self.config.ratio * 100)
                suffix = f"_attack_{self.config.attack_type}_{ratio_percent}"

            # Add suffix to filename before .json extension
            base_name = json_file[:-5] if json_file.endswith('.json') else json_file
            output_filename = f"{base_name}{suffix}.json"
            output_path = os.path.join(output_dir, output_filename)

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

                if stats.get('skipped', False):
                    skipped_count += 1
                    print(f"⊸ {json_file}: Skipped (already exists)")
                else:
                    processed_count += 1
                    print(f"✓ {json_file}: {stats['successful_attacks']}/{stats['total_samples']} samples attacked")

            except Exception as e:
                print(f"✗ {json_file}: Error - {e}")

        # Add skip/process counts to overall stats
        if self.config.increment_mode:
            overall_stats['skipped_count'] = skipped_count
            overall_stats['processed_count'] = processed_count

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
        if self.config.increment_mode:
            print(f"Increment mode: ENABLED (skip existing results)")

        if self.config.attack_type != 'paraphrase':
            print(f"Attack ratio: {self.config.ratio:.1%}")
        else:
            print(f"API provider: {self.config.api_provider}")
            print(f"API model: {self.config.api_model}")
            print(f"Temperature: {self.config.temperature}")
            print(f"Max concurrent tasks: {self.config.max_concurrent}")

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
                'increment_mode': self.config.increment_mode,
                'api_provider': self.config.api_provider if self.config.attack_type == 'paraphrase' else None,
                'api_model': self.config.api_model if self.config.attack_type == 'paraphrase' else None,
                'temperature': self.config.temperature if self.config.attack_type == 'paraphrase' else None
            },
            'statistics': {
                'files_processed': stats['files_processed'],
                'total_samples': stats['total_samples'],
                'successful_attacks': stats['successful_attacks'],
                'failed_attacks': stats['failed_attacks'],
                'success_rate': stats['successful_attacks'] / stats['total_samples'] if stats['total_samples'] > 0 else 0,
                'files_skipped': stats.get('skipped_count', 0) if self.config.increment_mode else None,
                'files_newly_processed': stats.get('processed_count', stats['files_processed']) if self.config.increment_mode else None
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
        if 'skipped_count' in stats:
            print(f"Files skipped (already exist): {stats['skipped_count']}")
            print(f"Files newly processed: {stats['processed_count']}")
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

    # API arguments for paraphrase attack
    parser.add_argument(
        "--api-provider",
        type=str,
        default="openai",
        choices=["openai", "gemini"],
        help="API provider for paraphrase attack (default: openai)"
    )

    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for paraphrase (can also be set via OPENAI_API_KEY or GEMINI_API_KEY environment variable)"
    )

    parser.add_argument(
        "--api-model",
        type=str,
        default="gpt-3.5-turbo",
        help="Model to use for paraphrase attack (default: gpt-3.5-turbo for OpenAI, gemini-pro for Gemini)"
    )

    parser.add_argument(
        "--api-base",
        type=str,
        default=None,
        help="API base for paraphrase attack (default: None)"
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for paraphrase generation (default: 0.7)"
    )

    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=10,
        help="Maximum concurrent tasks for paraphrase attack (default: 10)"
    )

    parser.add_argument(
        "--increment",
        action="store_true",
        help="Enable increment mode - skip files that already exist in output directory"
    )

    args = parser.parse_args()

    # Check if input exists
    if not os.path.exists(args.input):
        print(f"Error: Input path '{args.input}' not found")
        return

    # Auto-adjust api_model if not specified and using Gemini
    api_model = args.api_model
    if args.attack == 'paraphrase' and args.api_provider == 'gemini' and api_model == 'gpt-3.5-turbo':
        api_model = 'gemini-pro'

    # Create configuration
    try:
        config = AttackConfig(
            attack_type=args.attack,
            ratio=args.ratio,
            seed=args.seed,
            min_output_length=args.min_output_length,
            model_name=args.model,
            increment_mode=args.increment,
            api_provider=args.api_provider,
            api_key=args.api_key,
            api_model=api_model,
            temperature=args.temperature,
            max_concurrent=args.max_concurrent
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

            # Generate file suffix
            if config.attack_type == 'paraphrase':
                # For paraphrase, use model name (sanitize it for filename)
                model_suffix = config.api_model.replace('/', '_').replace('\\', '_')
                suffix = f"_attack_paraphrase_{model_suffix}"
            else:
                # For other attacks, use type and ratio
                ratio_percent = int(config.ratio * 100)
                suffix = f"_attack_{config.attack_type}_{ratio_percent}"

            output_file = os.path.join(dir_path, f"{base_name}{suffix}.json")
        else:
            output_file = args.output

        # Check if output exists in increment mode
        if config.increment_mode and os.path.exists(output_file):
            print(f"\nIncrement mode: Skipping {args.input}")
            print(f"Output already exists: {output_file}")
            return

        # Ensure output directory exists
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        print(f"Processing single file: {args.input}")
        print(f"Output: {output_file}")
        if config.increment_mode:
            print(f"Increment mode: ENABLED")

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