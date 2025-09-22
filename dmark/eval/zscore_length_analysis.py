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
    for result in tqdm(results, desc="Analyzing samples", leave=False):
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


def process_single_file(
    input_file: str,
    output_file: str,
    bitmap_dir: str = ".",
    bitmap_device: str = "cpu",
    model_name: str = "GSAI-ML/LLaDA-8B-Instruct",
    max_length: int = 200,
    min_length: int = 1,
    manual_config: dict = None,
    show_progress: bool = True
) -> Dict:
    """Process a single JSON file and output length-based z-score analysis.

    Args:
        input_file: Path to input JSON file
        output_file: Path to output JSON file
        bitmap_dir: Directory containing bitmap files
        bitmap_device: Device to store the bitmap on
        model_name: Model name for tokenizer
        max_length: Maximum generation length to analyze
        min_length: Minimum generation length to analyze
        manual_config: Manual watermark config if not in JSON
        show_progress: Whether to show progress bar

    Returns:
        Dictionary with processing status and statistics
    """
    file_basename = os.path.basename(input_file)

    # Load the JSON data
    try:
        with open(input_file, 'r') as f:
            results = json.load(f)
    except Exception as e:
        if show_progress:
            print(f"❌ Error reading {file_basename}: {e}")
        return {'status': 'error', 'message': str(e), 'file': file_basename}

    if not results:
        if show_progress:
            print(f"⏭️  Skipping {file_basename}: Empty file")
        return {'status': 'empty', 'message': 'Empty file', 'file': file_basename}

    # Get watermark configuration
    watermark_metadata = None
    is_watermarked = False

    if results[0].get("watermark_metadata"):
        watermark_metadata = results[0]["watermark_metadata"]
        is_watermarked = True
    elif manual_config:
        watermark_metadata = manual_config
    else:
        if show_progress:
            print(f"⏭️  Skipping {file_basename}: No watermark config")
        return {'status': 'no_config', 'message': 'No watermark config', 'file': file_basename}

    # Initialize watermark and tokenizer
    try:
        watermark = initialize_watermark(watermark_metadata, bitmap_dir, bitmap_device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except FileNotFoundError as e:
        if show_progress:
            print(f"❌ Bitmap error for {file_basename}: {str(e).split(':')[0]}")
        return {'status': 'bitmap_error', 'message': str(e), 'file': file_basename}
    except Exception as e:
        if show_progress:
            print(f"❌ Initialization error for {file_basename}: {e}")
        return {'status': 'init_error', 'message': str(e), 'file': file_basename}

    # Perform analysis
    if show_progress:
        print(f"📊 Analyzing {file_basename} ({len(results)} samples)...")

    length_stats = analyze_length_ranges(
        results, watermark, tokenizer, max_length, min_length
    )

    # Prepare output data
    output_data = {
        'metadata': {
            'input_file': file_basename,
            'num_samples': len(results),
            'min_length': min_length,
            'max_length': max_length,
            'watermark_config': watermark_metadata,
            'model': model_name,
            'is_watermarked': is_watermarked
        },
        'length_statistics': {}
    }

    # Convert length stats to serializable format
    valid_lengths = []
    for length, stats in length_stats.items():
        if stats['sample_count'] > 0:
            valid_lengths.append(length)
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
    if valid_lengths:
        z_scores_by_length = [output_data['length_statistics'][str(l)]['avg_z_score']
                              for l in sorted(valid_lengths)]
        output_data['summary'] = {
            'overall_avg_z_score': float(np.mean(z_scores_by_length)),
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

    if show_progress:
        print(f"✓ Saved analysis to: {os.path.basename(output_file)}")

    # Return statistics
    return {
        'status': 'success',
        'file': file_basename,
        'is_watermarked': is_watermarked,
        'num_samples': len(results),
        'overall_avg_z_score': output_data.get('summary', {}).get('overall_avg_z_score', 0),
        'output_file': output_file
    }


def find_json_files(input_dir: str, tag: str) -> List[str]:
    """Find JSON files to process, excluding already tagged files."""
    tag_suffix = f"_{tag}.json"
    return [f for f in os.listdir(input_dir)
            if f.endswith('.json')
            and not f.endswith(tag_suffix)
            and not f.startswith('_')]


def process_directory(
    input_dir: str,
    output_dir: str,
    tag: str,
    bitmap_dir: str,
    bitmap_device: str,
    model_name: str,
    max_length: int,
    min_length: int,
    manual_config: dict = None,
    consolidate: bool = False
) -> None:
    """Process all JSON files in a directory.

    Args:
        input_dir: Input directory path
        output_dir: Output directory path
        tag: Tag to append to output files
        bitmap_dir: Directory containing bitmap files
        bitmap_device: Device to store the bitmap on
        model_name: Model name for tokenizer
        max_length: Maximum generation length to analyze
        min_length: Minimum generation length to analyze
        manual_config: Manual watermark config if not in JSON
        consolidate: If True, also create a consolidated analysis file
    """
    # Find files to process
    json_files = find_json_files(input_dir, tag)

    if not json_files:
        print(f"No JSON files found in directory: {input_dir}")
        return

    print(f"\n{'='*60}")
    print(f"📁 Directory: {input_dir}")
    print(f"📊 Total JSON files found: {len(json_files)}")
    print(f"🔄 Files to process: {len(json_files)}")
    print(f"💾 Output directory: {output_dir}")
    print(f"📏 Length range: {min_length} to {max_length}")

    if manual_config:
        print(f"⚙️  Manual config: ratio={manual_config['ratio']}, key={manual_config['key']}")

    print(f"{'='*60}\n")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Process each file
    processed = 0
    failed = 0
    watermarked_files = 0
    non_watermarked_files = 0
    all_z_scores_overall = []

    # Main progress bar
    main_pbar = tqdm(json_files, desc="Overall Progress")

    for json_file in main_pbar:
        input_path = os.path.join(input_dir, json_file)
        output_name = json_file.replace(".json", f"_{tag}.json")
        output_path = os.path.join(output_dir, output_name)

        # Update progress bar description
        file_display = json_file[:30] + "..." if len(json_file) > 33 else json_file
        main_pbar.set_description(f"Processing: {file_display}")

        # Process file
        result = process_single_file(
            input_path, output_path, bitmap_dir, bitmap_device,
            model_name, max_length, min_length, manual_config,
            show_progress=False
        )

        # Track statistics
        if result['status'] == 'success':
            processed += 1
            if result['is_watermarked']:
                watermarked_files += 1
            else:
                non_watermarked_files += 1

            if result['overall_avg_z_score'] > 0:
                all_z_scores_overall.append(result['overall_avg_z_score'])

            tqdm.write(f"  ✓ {json_file} -> {output_name}")
        else:
            failed += 1
            tqdm.write(f"  ❌ {json_file}: {result.get('message', 'Unknown error')}")

        # Update progress bar postfix
        postfix = {'✅': processed, '❌': failed}
        if all_z_scores_overall:
            postfix['z̄'] = f"{np.mean(all_z_scores_overall):.2f}"
        main_pbar.set_postfix(postfix)

    main_pbar.close()

    # Print summary
    print(f"\n{'='*60}")
    print(f"📊 PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"✅ Successfully processed: {processed} files")
    if failed > 0:
        print(f"❌ Failed: {failed} files")

    if processed > 0:
        print(f"\n📈 File Types:")
        print(f"   • Watermarked: {watermarked_files}")
        print(f"   • Non-watermarked: {non_watermarked_files}")

    if all_z_scores_overall:
        print(f"\n📊 Overall Z-Score Statistics:")
        print(f"   • Average: {np.mean(all_z_scores_overall):.4f}")
        print(f"   • Min: {np.min(all_z_scores_overall):.4f}")
        print(f"   • Max: {np.max(all_z_scores_overall):.4f}")

    print(f"\n💾 Output directory: {output_dir}")
    print(f"{'='*60}")

    # Create consolidated analysis if requested
    if consolidate and processed > 0:
        print("\n📊 Creating consolidated analysis...")
        create_consolidated_analysis(output_dir, tag)


def create_consolidated_analysis(output_dir: str, tag: str) -> None:
    """Create a consolidated analysis from all processed files."""
    # Find all analysis files
    analysis_files = [f for f in os.listdir(output_dir)
                     if f.endswith(f"_{tag}.json")]

    if not analysis_files:
        print("No analysis files found for consolidation")
        return

    # Aggregate data
    all_data = []
    for file in analysis_files:
        with open(os.path.join(output_dir, file), 'r') as f:
            data = json.load(f)
            all_data.append(data)

    # Calculate consolidated statistics
    consolidated = {
        'metadata': {
            'num_files': len(analysis_files),
            'total_samples': sum(d['metadata']['num_samples'] for d in all_data),
            'files': [d['metadata']['input_file'] for d in all_data]
        },
        'consolidated_statistics': {},
        'summary': {
            'avg_overall_z_score': np.mean([d.get('summary', {}).get('overall_avg_z_score', 0)
                                           for d in all_data if d.get('summary')])
        }
    }

    # Aggregate by length
    length_data = {}
    for data in all_data:
        for length_str, stats in data.get('length_statistics', {}).items():
            if length_str not in length_data:
                length_data[length_str] = []
            length_data[length_str].append(stats['avg_z_score'])

    # Calculate consolidated stats for each length
    for length_str, z_scores in length_data.items():
        consolidated['consolidated_statistics'][length_str] = {
            'num_files': len(z_scores),
            'avg_z_score': float(np.mean(z_scores)),
            'std_z_score': float(np.std(z_scores)),
            'min_z_score': float(np.min(z_scores)),
            'max_z_score': float(np.max(z_scores))
        }

    # Save consolidated analysis
    consolidated_path = os.path.join(output_dir, f"_consolidated_{tag}.json")
    with open(consolidated_path, 'w') as f:
        json.dump(consolidated, f, indent=2)

    print(f"✓ Consolidated analysis saved to: {os.path.basename(consolidated_path)}")


def main():
    parser = argparse.ArgumentParser(
        description="Calculate z-scores for various generation lengths to analyze watermark strength over time"
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input JSON file(s) or directory containing JSON files"
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file or directory (default: auto-generated based on input)"
    )

    parser.add_argument(
        "--tag",
        type=str,
        default="length_analysis",
        help="Tag to append to output files when processing directory (default: length_analysis)"
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

    parser.add_argument(
        "--consolidate",
        action="store_true",
        help="Create a consolidated analysis file when processing directories"
    )

    args = parser.parse_args()

    # Validate input
    if not os.path.exists(args.input):
        print(f"Error: Input path '{args.input}' not found")
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

    # Process based on input type
    if os.path.isdir(args.input):
        # Directory processing
        if args.output:
            output_dir = args.output
        else:
            input_dirname = os.path.basename(os.path.normpath(args.input))
            parent_dir = os.path.dirname(os.path.normpath(args.input))
            output_dir = os.path.join(parent_dir, f"{input_dirname}_{args.tag}")

        process_directory(
            args.input, output_dir, args.tag, args.bitmap_dir,
            args.bitmap_device, args.model, args.max_length,
            args.min_length, manual_config, args.consolidate
        )

    elif os.path.isfile(args.input):
        # Single file processing
        if args.output:
            output_file = args.output
        else:
            output_file = args.input.replace(".json", f"_{args.tag}.json")

        print(f"\nProcessing: {os.path.basename(args.input)}")
        result = process_single_file(
            args.input, output_file, args.bitmap_dir, args.bitmap_device,
            args.model, args.max_length, args.min_length, manual_config
        )

        if result['status'] == 'success':
            print(f"\n✅ Analysis complete!")
            print(f"📊 Samples analyzed: {result['num_samples']}")
            print(f"📈 Overall average z-score: {result['overall_avg_z_score']:.4f}")
            print(f"💾 Output saved to: {output_file}")

            # Print key statistics
            with open(output_file, 'r') as f:
                data = json.load(f)

            if 'summary' in data:
                print(f"\n🔍 Key Statistics:")
                for checkpoint in [50, 100, 150, 200]:
                    if str(checkpoint) in data['length_statistics']:
                        stats = data['length_statistics'][str(checkpoint)]
                        print(f"  • At length {checkpoint}: z-score={stats['avg_z_score']:.4f}, "
                              f"detection_rate={stats['avg_detection_rate']:.2%}")
        else:
            print(f"❌ Error: {result.get('message', 'Unknown error')}")
    else:
        print(f"Error: '{args.input}' is neither a file nor a directory")


if __name__ == "__main__":
    main()