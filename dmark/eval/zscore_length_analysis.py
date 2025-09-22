#!/usr/bin/env python
"""
Calculate z-scores for various generation lengths to analyze watermark detection strength over time.

This script processes JSON files containing watermarked text generation results and calculates:
1. Z-scores for each generation length from 1 to max_length
2. Average z-scores across all samples for each generation length
3. Detection rates and confidence intervals

The output shows how watermark detection strength evolves as generation length increases.
"""

import argparse
import json
import math
import os
from typing import List, Dict, Tuple
from tqdm import tqdm
import numpy as np

from dmark.watermark.persistent_bitmap import PersistentBitmap
from transformers import AutoTokenizer
from dmark.watermark.config import WatermarkConfig
from dmark.watermark.watermark import Watermark


# Constants
EOS_TOKENS = {126081, 126348}  # Special tokens to skip


def get_previous_token(index: int, output_ids: List[int], prompt_ids: List[int]) -> int:
    """Get the previous token for green list generation.

    Args:
        index: Current position in output_ids
        output_ids: List of generated token IDs
        prompt_ids: List of prompt token IDs

    Returns:
        Previous token ID
    """
    if index > 0:
        return output_ids[index - 1]
    elif prompt_ids:
        return prompt_ids[-1]
    else:
        return 0


def calculate_zscore_at_length(
    output_ids: List[int],
    watermark: Watermark,
    prompt_ids: List[int],
    target_length: int
) -> Tuple[float, float, int, int]:
    """Calculate z-score for watermark detection at a specific generation length.

    Args:
        output_ids: List of token IDs to check for watermark
        watermark: Watermark instance for efficient green list generation
        prompt_ids: List of prompt token IDs
        target_length: Target generation length to analyze

    Returns:
        Tuple of (detection_rate, z_score, detected_count, actual_length)
    """
    detected = 0
    gen_len = 0

    # Limit to target_length or available tokens
    tokens_to_check = min(len(output_ids), target_length)

    for index in range(tokens_to_check):
        curr_token = output_ids[index]

        # Skip special tokens (EOS tokens)
        if curr_token in EOS_TOKENS:
            break

        # Get previous token for green list generation
        prev_token = get_previous_token(index, output_ids, prompt_ids)

        # Check if current token is in green list
        green_list = watermark.gen_green_list(prev_token).bool()
        if green_list[curr_token]:
            detected += 1
        gen_len += 1

    if gen_len == 0:
        return 0.0, 0.0, 0, 0

    # Calculate statistics
    detection_rate = detected / gen_len
    ratio = watermark.watermark_config.ratio
    expected = gen_len * ratio
    variance = gen_len * ratio * (1 - ratio)
    z_score = (detected - expected) / math.sqrt(variance) if variance > 0 else 0.0

    return detection_rate, z_score, detected, gen_len


def analyze_length_ranges(
    results: List[dict],
    watermark: Watermark,
    tokenizer,
    max_length: int = 200,
    min_length: int = 1
) -> Dict[int, Dict]:
    """Analyze z-scores for various generation lengths.

    Args:
        results: List of result dictionaries from JSON
        watermark: Watermark instance for detection
        tokenizer: Tokenizer for encoding prompts
        max_length: Maximum generation length to analyze
        min_length: Minimum generation length to analyze

    Returns:
        Dictionary mapping generation length to statistics
    """
    length_stats = {}

    # Initialize statistics for each length
    for length in range(min_length, max_length + 1):
        length_stats[length] = {
            'z_scores': [],
            'detection_rates': [],
            'sample_count': 0
        }

    # Process each result
    for result in tqdm(results, desc="Processing samples"):
        # Get prompt IDs
        prompt_text = result["data"].get("prompt", "")
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False) if prompt_text else []

        # Get output IDs (prefer original, fallback to truncated)
        if "output_ids" in result.get("data", {}):
            output_ids = result["data"]["output_ids"]
        elif "truncated_output_ids" in result.get("data", {}):
            output_ids = result["data"]["truncated_output_ids"]
        else:
            continue

        # Calculate z-scores for each generation length
        for length in range(min_length, min(len(output_ids), max_length) + 1):
            detection_rate, z_score, detected, actual_len = calculate_zscore_at_length(
                output_ids, watermark, prompt_ids, length
            )

            if actual_len > 0:  # Only include valid calculations
                length_stats[length]['z_scores'].append(z_score)
                length_stats[length]['detection_rates'].append(detection_rate)
                length_stats[length]['sample_count'] += 1

    # Calculate aggregate statistics
    for length in length_stats:
        z_scores = length_stats[length]['z_scores']
        detection_rates = length_stats[length]['detection_rates']

        if z_scores:
            length_stats[length]['avg_z_score'] = np.mean(z_scores)
            length_stats[length]['std_z_score'] = np.std(z_scores)
            length_stats[length]['min_z_score'] = np.min(z_scores)
            length_stats[length]['max_z_score'] = np.max(z_scores)
            length_stats[length]['median_z_score'] = np.median(z_scores)

            # Calculate confidence intervals (95%)
            n = len(z_scores)
            if n > 1:
                se = length_stats[length]['std_z_score'] / np.sqrt(n)
                length_stats[length]['ci_lower'] = length_stats[length]['avg_z_score'] - 1.96 * se
                length_stats[length]['ci_upper'] = length_stats[length]['avg_z_score'] + 1.96 * se
            else:
                length_stats[length]['ci_lower'] = length_stats[length]['avg_z_score']
                length_stats[length]['ci_upper'] = length_stats[length]['avg_z_score']

        if detection_rates:
            length_stats[length]['avg_detection_rate'] = np.mean(detection_rates)
            length_stats[length]['std_detection_rate'] = np.std(detection_rates)

        # Remove raw data lists to save space in output
        del length_stats[length]['z_scores']
        del length_stats[length]['detection_rates']

    return length_stats


def generate_bitmap_filename(ratio: float, vocab_size: int, key: int) -> str:
    """Generate bitmap filename from watermark parameters."""
    ratio_str = int(ratio * 100)
    return f"bitmap_v{vocab_size}_r{ratio_str}_k{key}.bin"


def initialize_watermark(watermark_metadata: dict, bitmap_dir: str, bitmap_device: str = "cpu") -> Watermark:
    """Initialize watermark from metadata."""
    ratio = watermark_metadata.get("ratio", 0.5)
    vocab_size = watermark_metadata.get("vocab_size", 126464)
    key = watermark_metadata.get("key", 42)

    bitmap_filename = generate_bitmap_filename(ratio, vocab_size, key)
    bitmap_path = os.path.join(bitmap_dir, bitmap_filename)

    if not os.path.exists(bitmap_path):
        raise FileNotFoundError(
            f"Bitmap file not found: {bitmap_path}\n"
            f"Expected bitmap filename: {bitmap_filename}\n"
            f"Parameters: ratio={ratio}, vocab_size={vocab_size}, key={key}"
        )

    config = WatermarkConfig(
        vocab_size=vocab_size,
        ratio=ratio,
        delta=watermark_metadata.get("delta", 2.0),
        key=key,
        prebias=False,
        strategy="normal",
        bitmap_path=bitmap_path
    )

    bitmap = PersistentBitmap(config.vocab_size, config.bitmap_path, device=bitmap_device)
    return Watermark(config, bitmap)


def process_json_files(
    input_files: List[str],
    output_file: str,
    bitmap_dir: str = ".",
    bitmap_device: str = "cpu",
    model_name: str = "GSAI-ML/LLaDA-8B-Instruct",
    max_length: int = 200,
    min_length: int = 1,
    manual_config: dict = None
) -> None:
    """Process JSON files and output length-based z-score analysis.

    Args:
        input_files: List of input JSON file paths
        output_file: Path to output JSON file
        bitmap_dir: Directory containing bitmap files
        bitmap_device: Device to store the bitmap on
        model_name: Model name for tokenizer
        max_length: Maximum generation length to analyze
        min_length: Minimum generation length to analyze
        manual_config: Manual watermark config if not in JSON
    """
    all_results = []
    watermark = None
    tokenizer = None

    # Load all JSON files
    print(f"Loading {len(input_files)} JSON file(s)...")
    for input_file in input_files:
        try:
            with open(input_file, 'r') as f:
                results = json.load(f)
                all_results.extend(results)
                print(f"  ✓ Loaded {len(results)} samples from {os.path.basename(input_file)}")
        except Exception as e:
            print(f"  ✗ Error loading {input_file}: {e}")
            continue

    if not all_results:
        print("Error: No valid results loaded from input files")
        return

    print(f"Total samples loaded: {len(all_results)}")

    # Get watermark configuration
    watermark_metadata = None
    if all_results[0].get("watermark_metadata"):
        watermark_metadata = all_results[0]["watermark_metadata"]
        print("Using watermark metadata from JSON file")
    elif manual_config:
        watermark_metadata = manual_config
        print("Using manual watermark configuration")
    else:
        print("Error: No watermark configuration found")
        return

    # Initialize watermark and tokenizer
    try:
        watermark = initialize_watermark(watermark_metadata, bitmap_dir, bitmap_device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        print(f"Error initializing: {e}")
        return

    # Perform analysis
    print(f"\nAnalyzing z-scores for generation lengths {min_length} to {max_length}...")
    length_stats = analyze_length_ranges(
        all_results, watermark, tokenizer, max_length, min_length
    )

    # Prepare output data
    output_data = {
        'metadata': {
            'num_samples': len(all_results),
            'min_length': min_length,
            'max_length': max_length,
            'watermark_config': watermark_metadata,
            'model': model_name,
            'input_files': input_files
        },
        'length_statistics': {}
    }

    # Convert length stats to serializable format
    for length, stats in length_stats.items():
        if stats['sample_count'] > 0:
            output_data['length_statistics'][str(length)] = {
                'sample_count': stats['sample_count'],
                'avg_z_score': float(stats.get('avg_z_score', 0)),
                'std_z_score': float(stats.get('std_z_score', 0)),
                'min_z_score': float(stats.get('min_z_score', 0)),
                'max_z_score': float(stats.get('max_z_score', 0)),
                'median_z_score': float(stats.get('median_z_score', 0)),
                'ci_lower': float(stats.get('ci_lower', 0)),
                'ci_upper': float(stats.get('ci_upper', 0)),
                'avg_detection_rate': float(stats.get('avg_detection_rate', 0)),
                'std_detection_rate': float(stats.get('std_detection_rate', 0))
            }

    # Add summary statistics
    valid_lengths = [int(l) for l in output_data['length_statistics'].keys()]
    if valid_lengths:
        z_scores_by_length = [output_data['length_statistics'][str(l)]['avg_z_score']
                              for l in sorted(valid_lengths)]
        output_data['summary'] = {
            'overall_avg_z_score': np.mean(z_scores_by_length),
            'min_length_with_data': min(valid_lengths),
            'max_length_with_data': max(valid_lengths),
            'z_score_at_50': output_data['length_statistics'].get('50', {}).get('avg_z_score'),
            'z_score_at_100': output_data['length_statistics'].get('100', {}).get('avg_z_score'),
            'z_score_at_150': output_data['length_statistics'].get('150', {}).get('avg_z_score'),
            'z_score_at_200': output_data['length_statistics'].get('200', {}).get('avg_z_score')
        }

    # Save output
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n✓ Analysis complete! Results saved to: {output_file}")

    # Print summary
    if valid_lengths:
        print("\nKey Statistics:")
        print(f"  • Samples analyzed: {len(all_results)}")
        print(f"  • Generation lengths: {min(valid_lengths)} to {max(valid_lengths)}")
        print(f"  • Overall average z-score: {output_data['summary']['overall_avg_z_score']:.4f}")

        for checkpoint in [50, 100, 150, 200]:
            if str(checkpoint) in output_data['length_statistics']:
                stats = output_data['length_statistics'][str(checkpoint)]
                print(f"  • At length {checkpoint}: z-score={stats['avg_z_score']:.4f}, "
                      f"detection_rate={stats['avg_detection_rate']:.2%} "
                      f"(n={stats['sample_count']})")


def main():
    parser = argparse.ArgumentParser(
        description="Calculate z-scores for various generation lengths to analyze watermark strength over time"
    )

    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="Input JSON file(s) containing generation results"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="zscore_length_analysis.json",
        help="Output JSON file for length-based z-score analysis (default: zscore_length_analysis.json)"
    )

    parser.add_argument(
        "--max_length",
        type=int,
        default=200,
        help="Maximum generation length to analyze (default: 200)"
    )

    parser.add_argument(
        "--min_length",
        type=int,
        default=1,
        help="Minimum generation length to analyze (default: 1)"
    )

    parser.add_argument(
        "--bitmap_dir",
        type=str,
        default=".",
        help="Directory containing bitmap files (default: current directory)"
    )

    parser.add_argument(
        "--bitmap_device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to store the bitmap on (default: cpu)"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="GSAI-ML/LLaDA-8B-Instruct",
        help="Model name for tokenizer"
    )

    # Watermark configuration arguments (for manual config)
    parser.add_argument("--vocab_size", type=int, default=126464)
    parser.add_argument("--ratio", type=float, default=0.5)
    parser.add_argument("--delta", type=float, default=2.0)
    parser.add_argument("--key", type=int, default=42)
    parser.add_argument("--use_manual_config", action="store_true",
                       help="Force use of manual config even if metadata exists in JSON")

    args = parser.parse_args()

    # Validate input files
    valid_input_files = []
    for input_file in args.input:
        if os.path.exists(input_file) and os.path.isfile(input_file):
            valid_input_files.append(input_file)
        else:
            print(f"Warning: Input file '{input_file}' not found or not a file")

    if not valid_input_files:
        print("Error: No valid input files found")
        return

    # Build manual config if needed
    manual_config = None
    if args.use_manual_config or any([
        args.vocab_size != 126464,
        args.ratio != 0.5,
        args.delta != 2.0,
        args.key != 42
    ]):
        manual_config = {
            "vocab_size": args.vocab_size,
            "ratio": args.ratio,
            "delta": args.delta,
            "key": args.key
        }
        print("Manual watermark configuration enabled")

    # Process files
    process_json_files(
        valid_input_files,
        args.output,
        args.bitmap_dir,
        args.bitmap_device,
        args.model,
        args.max_length,
        args.min_length,
        manual_config
    )


if __name__ == "__main__":
    main()