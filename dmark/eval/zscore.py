import argparse
import json
import math
import os
from typing import List, Dict
from tqdm import tqdm

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


def calculate_zscore(
    output_ids: List[int],
    watermark: Watermark,
    prompt_ids: List[int],
    max_tokens: int = None
) -> tuple[float, float, int, int]:
    """Calculate z-score for watermark detection.
    
    Args:
        output_ids: List of token IDs to check for watermark (generated tokens only)
        watermark: Watermark instance for efficient green list generation
        prompt_ids: List of prompt token IDs
        max_tokens: Maximum number of tokens to analyze (None = analyze all)
    
    Returns:
        Tuple of (detection_rate, z_score, detected_count, gen_len)
    """
    detected = 0
    gen_len = 0
    
    # Limit to max_tokens if specified
    tokens_to_check = min(len(output_ids), max_tokens) if max_tokens else len(output_ids)
    
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


def process_single_result(
    result: dict,
    watermark: Watermark,
    prompt_ids: List[int],
    max_tokens: int = None
) -> None:
    """Process a single result entry and add z-scores.
    
    Args:
        result: Single result dictionary to process
        watermark: Watermark instance for detection
        prompt_ids: List of prompt token IDs
        max_tokens: Maximum tokens to analyze
    """
    # Initialize watermark dict if not present
    if "watermark" not in result:
        result["watermark"] = {
            # Add essential fields for z-score calculation
            "ratio": watermark.watermark_config.ratio,
            "key": watermark.watermark_config.key,
            "vocab_size": watermark.watermark_config.vocab_size
        }
    
    # Process different ID types with their corresponding z-score fields
    id_mappings = [
        ("output_ids", "z_score_original", "detection_rate_original", "detected_original", "gen_len_original"),
        ("truncated_output_ids", "z_score_truncated", "detection_rate_truncated", "detected_truncated", "gen_len_truncated"),
        ("attacked_ids", "z_score_attacked", "detection_rate_attacked", "detected_attacked", "gen_len_attacked")
    ]
    
    for id_field, z_field, rate_field, detected_field, len_field in id_mappings:
        if id_field in result.get("data", {}):
            ids = result["data"][id_field]
            detection_rate, z_score, detected, gen_len = calculate_zscore(ids, watermark, prompt_ids, max_tokens)
            result["watermark"][z_field] = z_score
            result["watermark"][rate_field] = detection_rate
            result["watermark"][detected_field] = detected
            result["watermark"][len_field] = gen_len


def print_statistics(results: List[dict], output_file: str) -> None:
    """Print z-score statistics for processed results.
    
    Args:
        results: List of processed result dictionaries
        output_file: Path where results were saved
    """
    print(f"Processed {len(results)} results")
    print(f"Results saved to: {output_file}")
    print("\n--- Z-score Statistics ---")
    
    # Define score types to analyze
    score_types = [
        ("Original", "z_score_original"),
        ("Truncated", "z_score_truncated"),
        ("Attacked", "z_score_attacked")
    ]
    
    for label, field in score_types:
        # Extract scores for this type
        scores = [r["watermark"][field] for r in results 
                 if r.get("watermark") and field in r["watermark"]]
        
        if scores:
            avg_z = sum(scores) / len(scores)
            min_z = min(scores)
            max_z = max(scores)
            print(f"{label} output: avg={avg_z:.2f}, min={min_z:.2f}, max={max_z:.2f} (n={len(scores)})")


def generate_bitmap_filename(ratio: float, vocab_size: int, key: int) -> str:
    """Generate bitmap filename from watermark parameters.
    
    Args:
        ratio: Green list ratio
        vocab_size: Vocabulary size
        key: Random seed key
    
    Returns:
        Bitmap filename
    """
    # Format ratio as percentage (e.g., 0.5 -> 50, 0.25 -> 25)
    ratio_str = int(ratio * 100)
    return f"bitmap_v{vocab_size}_r{ratio_str}_k{key}.bin"


def initialize_watermark(watermark_metadata: dict, bitmap_dir: str, bitmap_device: str = "cpu") -> Watermark:
    """Initialize watermark from metadata.
    
    Args:
        watermark_metadata: Dictionary containing watermark configuration
        bitmap_dir: Directory containing bitmap files
        bitmap_device: Device to store the bitmap on ("cpu" or "cuda")
    
    Returns:
        Initialized Watermark instance
    """
    # Extract essential parameters for detection
    ratio = watermark_metadata.get("ratio", 0.5)
    vocab_size = watermark_metadata.get("vocab_size", 126464)
    key = watermark_metadata.get("key", 42)
    
    # Generate bitmap filename
    bitmap_filename = generate_bitmap_filename(ratio, vocab_size, key)
    bitmap_path = os.path.join(bitmap_dir, bitmap_filename)
    
    # Check if bitmap file exists - throw error if not
    if not os.path.exists(bitmap_path):
        raise FileNotFoundError(
            f"Bitmap file not found: {bitmap_path}\n"
            f"Expected bitmap filename: {bitmap_filename}\n"
            f"Parameters: ratio={ratio}, vocab_size={vocab_size}, key={key}"
        )
    
    # Use default values for fields not needed in detection
    config = WatermarkConfig(
        vocab_size=vocab_size,
        ratio=ratio,
        delta=watermark_metadata.get("delta", 2.0),  # Not used in detection, but keep if available
        key=key,
        prebias=False,  # Not needed for detection
        strategy="normal",  # Not needed for detection
        bitmap_path=bitmap_path
    )
    
    bitmap = PersistentBitmap(config.vocab_size, config.bitmap_path, device=bitmap_device)
    return Watermark(config, bitmap)


def process_json_file(
    input_file: str,
    output_file: str,
    bitmap_dir: str = ".",
    bitmap_device: str = "cpu",
    model_name: str = "facebook/llada-760m-split2",
    max_tokens: int = None,
    manual_config: dict = None,
    show_progress: bool = True,
    file_num: int = None,
    total_files: int = None
) -> Dict:
    """Process a JSON file containing generation results and add z-scores.
    
    Args:
        input_file: Path to input JSON file
        output_file: Path to output JSON file
        bitmap_dir: Directory containing bitmap files
        bitmap_device: Device to store the bitmap on ("cpu" or "cuda")
        model_name: Model name for tokenizer
        max_tokens: Maximum number of tokens to analyze (None = analyze all)
        manual_config: Manual watermark config to use if no metadata in JSON
    """
    # Load the JSON data
    try:
        with open(input_file, 'r') as f:
            results = json.load(f)
    except (IOError, OSError, json.JSONDecodeError) as e:
        if show_progress:
            print(f"Skipping {input_file}: cannot read file ({e})")
        return {'status': 'error', 'message': str(e)}
    
    if not results:
        if show_progress:
            print(f"Skipping {input_file}: empty file")
        return {'status': 'empty', 'message': 'Empty file'}
    
    # Try to get watermark config from JSON metadata or use manual config
    watermark_metadata = None
    is_watermarked = False
    if results[0].get("watermark_metadata"):
        watermark_metadata = results[0]["watermark_metadata"]
        is_watermarked = True
        if show_progress and not file_num:
            print(f"Using watermark metadata from JSON file")
    elif manual_config:
        watermark_metadata = manual_config
        if show_progress and not file_num:
            print(f"Using manual watermark configuration for non-watermarked content")
        # Note: We do NOT add watermark_metadata to results for non-watermarked content
        # It should remain None to indicate this is non-watermarked
    else:
        if show_progress:
            print(f"Skipping {input_file}: no watermark metadata found and no manual config provided")
        return {'status': 'no_config', 'message': 'No watermark config'}
    
    # Initialize components
    try:
        watermark = initialize_watermark(watermark_metadata, bitmap_dir, bitmap_device)
    except FileNotFoundError as e:
        if show_progress:
            print(f"Error: {e}")
        return {'status': 'bitmap_error', 'message': str(e)}
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Prepare progress bar description
    file_basename = os.path.basename(input_file)
    if file_num is not None and total_files is not None:
        desc = f"[{file_num}/{total_files}] {file_basename[:30]}..."
    else:
        desc = f"Processing {file_basename[:40]}..."
    
    # Statistics tracking
    stats = {
        'total': len(results),
        'processed': 0,
        'with_zscore': 0,
        'avg_zscore': 0.0,
        'detected_ratio': 0.0
    }
    
    # Process each result with progress tracking
    if show_progress:
        pbar = tqdm(results, desc=desc, leave=False, position=1)
    else:
        pbar = results
    
    z_scores_collected = []
    detection_rates = []
    
    for result in pbar:
        # Get prompt IDs
        prompt_text = result["data"].get("prompt", "")
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False) if prompt_text else []
        
        # Process this result
        process_single_result(result, watermark, prompt_ids, max_tokens)
        
        stats['processed'] += 1
        
        # Collect statistics
        if "watermark" in result:
            wm_data = result["watermark"]
            if "z_score_original" in wm_data:
                z_scores_collected.append(wm_data["z_score_original"])
                stats['with_zscore'] += 1
                if "detection_rate_original" in wm_data:
                    detection_rates.append(wm_data["detection_rate_original"])
            elif "z_score_truncated" in wm_data:
                z_scores_collected.append(wm_data["z_score_truncated"])
                stats['with_zscore'] += 1
                if "detection_rate_truncated" in wm_data:
                    detection_rates.append(wm_data["detection_rate_truncated"])
        
        # Update progress bar postfix with statistics
        if show_progress and isinstance(pbar, tqdm):
            if z_scores_collected:
                stats['avg_zscore'] = sum(z_scores_collected) / len(z_scores_collected)
            if detection_rates:
                stats['detected_ratio'] = sum(detection_rates) / len(detection_rates)
            
            postfix = {
                'z̄': f"{stats['avg_zscore']:.2f}",
                'det': f"{stats['detected_ratio']:.2%}",
                'n': stats['with_zscore']
            }
            pbar.set_postfix(postfix)
    
    # Save results to new file
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    # Prepare return statistics
    return_stats = {
        'status': 'success',
        'file': file_basename,
        'is_watermarked': is_watermarked,
        'total_samples': stats['total'],
        'processed': stats['processed'],
        'with_zscore': stats['with_zscore'],
        'avg_zscore': stats['avg_zscore'],
        'avg_detection_rate': stats['detected_ratio'],
        'output_file': output_file
    }
    
    # Print statistics only if not in batch mode
    if show_progress and not file_num:
        print_statistics(results, output_file)
    
    return return_stats


def find_json_files(input_dir: str, tag: str) -> List[str]:
    """Find JSON files to process, excluding already tagged and metadata files.
    
    Args:
        input_dir: Directory to search for JSON files
        tag: Tag to exclude from search
    
    Returns:
        List of JSON file names to process
    """
    tag_suffix = f"_{tag}.json"
    return [f for f in os.listdir(input_dir)
            if f.endswith('.json')
            and not f.endswith(tag_suffix)
            and not f.startswith('_')]


def process_directory(input_dir: str, output_dir: str, tag: str, bitmap_dir: str, bitmap_device: str, model: str, max_tokens: int, manual_config: dict = None, increment_mode: bool = False) -> None:
    """Process all JSON files in a directory.
    
    Args:
        input_dir: Input directory path
        output_dir: Output directory path
        tag: Tag to append to output files
        bitmap_dir: Directory containing bitmap files
        bitmap_device: Device to store the bitmap on ("cpu" or "cuda")
        model: Model name for tokenizer
        max_tokens: Maximum tokens to analyze
        manual_config: Manual watermark config to use if no metadata in JSON
        increment_mode: If True, skip files that already have output
    """
    # Find files to process
    json_files = find_json_files(input_dir, tag)
    
    if not json_files:
        print(f"No JSON files found in directory: {input_dir}")
        return
    
    # Filter out files that already have output in increment mode
    files_to_process = []
    files_skipped_existing = []
    
    for json_file in json_files:
        output_name = json_file.replace(".json", f"_{tag}.json")
        output_path = os.path.join(output_dir, output_name)
        if increment_mode and os.path.exists(output_path):
            files_skipped_existing.append(json_file)
        else:
            files_to_process.append(json_file)
    
    print(f"\n{'='*60}")
    print(f"📁 Directory: {input_dir}")
    print(f"📊 Total JSON files found: {len(json_files)}")
    if increment_mode and files_skipped_existing:
        print(f"⏭️  Skipping existing outputs: {len(files_skipped_existing)}")
    print(f"🔄 Files to process: {len(files_to_process)}")
    print(f"💾 Output directory: {output_dir}")
    
    if manual_config:
        print(f"⚙️  Manual config: ratio={manual_config['ratio']}, delta={manual_config['delta']}, key={manual_config['key']}")
    
    print(f"{'='*60}\n")
    
    if not files_to_process:
        print("No files to process (all outputs already exist)")
        return
    
    # Process each file with overall progress bar
    processed = 0
    failed = 0
    watermarked_files = 0
    non_watermarked_files = 0
    all_z_scores = []
    all_detection_rates = []
    
    # Main progress bar for overall progress
    main_pbar = tqdm(files_to_process, desc="Overall Progress", position=0)
    
    for idx, json_file in enumerate(main_pbar, 1):
        input_path = os.path.join(input_dir, json_file)
        output_name = json_file.replace(".json", f"_{tag}.json")
        output_path = os.path.join(output_dir, output_name)
        
        # Update main progress bar
        main_pbar.set_description(f"Overall [{idx}/{len(files_to_process)}]")
        
        # Process file with nested progress
        result = process_json_file(
            input_path, output_path, bitmap_dir, bitmap_device, model, max_tokens, manual_config,
            show_progress=True, file_num=idx, total_files=len(files_to_process)
        )
        
        # Track statistics
        if result['status'] == 'success':
            processed += 1
            if result['is_watermarked']:
                watermarked_files += 1
            else:
                non_watermarked_files += 1
            
            if result['avg_zscore'] > 0:
                all_z_scores.append(result['avg_zscore'])
            if result['avg_detection_rate'] > 0:
                all_detection_rates.append(result['avg_detection_rate'])
            
            # Update main progress bar postfix
            postfix = {
                '✅': processed,
                '❌': failed,
                'WM': watermarked_files,
                'Non-WM': non_watermarked_files
            }
            if all_z_scores:
                postfix['z̄'] = f"{sum(all_z_scores)/len(all_z_scores):.2f}"
            main_pbar.set_postfix(postfix)
        else:
            failed += 1
            # Update postfix with failure
            postfix = {
                '✅': processed,
                '❌': failed,
                'WM': watermarked_files,
                'Non-WM': non_watermarked_files
            }
            main_pbar.set_postfix(postfix)
    
    # Print final summary with statistics
    print(f"\n{'='*60}")
    print(f"📊 PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"✅ Successfully processed: {processed} files")
    if failed > 0:
        print(f"❌ Failed: {failed} files")
    if len(files_skipped_existing) > 0:
        print(f"⏭️  Skipped (existing): {len(files_skipped_existing)} files")
    
    print(f"\n📈 File Types:")
    print(f"   • Watermarked: {watermarked_files}")
    print(f"   • Non-watermarked: {non_watermarked_files}")
    
    if all_z_scores:
        print(f"\n📊 Z-Score Statistics:")
        print(f"   • Average: {sum(all_z_scores)/len(all_z_scores):.4f}")
        print(f"   • Min: {min(all_z_scores):.4f}")
        print(f"   • Max: {max(all_z_scores):.4f}")
    
    if all_detection_rates:
        print(f"\n🎯 Detection Rate Statistics:")
        print(f"   • Average: {sum(all_detection_rates)/len(all_detection_rates):.2%}")
        print(f"   • Min: {min(all_detection_rates):.2%}")
        print(f"   • Max: {max(all_detection_rates):.2%}")
    
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Calculate z-scores for watermark detection")
    parser.add_argument("--input", type=str, required=True,
                       help="Path to input directory or JSON file")
    parser.add_argument("--output", type=str, default=None,
                       help="Output directory path (default: auto-generated)")
    parser.add_argument("--tag", type=str, default="zscore",
                       help="Tag to append to output files (default: zscore)")
    parser.add_argument("--bitmap_dir", type=str, default=".",
                       help="Directory containing bitmap files (default: current directory)")
    parser.add_argument("--bitmap_device", type=str, default="cpu", choices=["cpu", "cuda"],
                       help="Device to store the bitmap on (default: cpu)")
    parser.add_argument("--model", type=str, default="GSAI-ML/LLaDA-8B-Instruct",
                       help="Model name for tokenizer")
    parser.add_argument("--max_tokens", type=int, default=200,
                       help="Maximum number of tokens to analyze")
    
    # Watermark configuration arguments (used when no metadata in JSON)
    parser.add_argument("--vocab_size", type=int, default=126464,
                       help="Vocabulary size for watermark (default: 126464)")
    parser.add_argument("--ratio", type=float, default=0.5,
                       help="Green list ratio for watermark (default: 0.5)")
    parser.add_argument("--delta", type=float, default=2.0,
                       help="Watermark strength parameter (default: 2.0)")
    parser.add_argument("--key", type=int, default=42,
                       help="Random seed for watermark hash function (default: 42)")
    parser.add_argument("--prebias", action="store_true",
                       help="Apply watermark bias before token selection")
    parser.add_argument("--strategy", type=str, default="normal",
                       choices=["normal", "predict", "reverse", "legacy-ahead", "legacy-both"],
                       help="Watermark strategy (default: normal)")
    parser.add_argument("--use_manual_config", action="store_true",
                       help="Force use of manual config even if metadata exists in JSON")
    parser.add_argument("--increment", action="store_true",
                       help="Increment mode: only process files that don't have output yet")
    
    args = parser.parse_args()
    
    # Validate input
    if not os.path.exists(args.input):
        print(f"Error: Input path '{args.input}' not found")
        return
    
    # Build manual config dictionary if needed
    manual_config = None
    if args.use_manual_config or any([
        args.vocab_size != 126464,
        args.ratio != 0.5,
        args.delta != 2.0,
        args.key != 42,
        args.prebias,
        args.strategy != "normal"
    ]):
        manual_config = {
            "vocab_size": args.vocab_size,
            "ratio": args.ratio,
            "delta": args.delta,
            "key": args.key,
            "prebias": args.prebias,
            "strategy": args.strategy
        }
        print(f"Manual watermark configuration enabled")
    
    # Process based on input type
    if os.path.isdir(args.input):
        # Determine output directory
        if args.output:
            output_dir = args.output
        else:
            input_dirname = os.path.basename(os.path.normpath(args.input))
            parent_dir = os.path.dirname(os.path.normpath(args.input))
            output_dir = os.path.join(parent_dir, f"{input_dirname}_{args.tag}")
        
        # Create output directory and process files
        os.makedirs(output_dir, exist_ok=True)
        process_directory(args.input, output_dir, args.tag, args.bitmap_dir, args.bitmap_device, args.model, args.max_tokens, manual_config, args.increment)
        
    elif os.path.isfile(args.input):
        # Determine output file path
        if args.output:
            os.makedirs(args.output, exist_ok=True)
            base_name = os.path.basename(args.input).replace(".json", f"_{args.tag}.json")
            output_file = os.path.join(args.output, base_name)
        else:
            output_file = args.input.replace(".json", f"_{args.tag}.json")
        
        # Check if output exists in increment mode
        if args.increment and os.path.exists(output_file):
            print(f"Skipping (output exists): {output_file}")
            return
        
        # Process single file
        print(f"\nProcessing: {os.path.basename(args.input)}")
        print(f"Output: {output_file}")
        result = process_json_file(args.input, output_file, args.bitmap_dir, args.bitmap_device, args.model, args.max_tokens, manual_config, show_progress=True)
        
        if result['status'] == 'success':
            print(f"\n✅ Successfully processed {result['total_samples']} samples")
            print(f"📊 Average z-score: {result['avg_zscore']:.4f}")
            print(f"🎯 Average detection rate: {result['avg_detection_rate']:.2%}")
    else:
        print(f"Error: '{args.input}' is neither a file nor a directory")
        return


if __name__ == "__main__":
    main()