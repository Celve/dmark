import argparse
import json
import math
import os
from typing import List
from tqdm import tqdm

from dmark.watermark.persistent_bitmap import PersistentBitmap
from transformers import AutoTokenizer

from dmark.watermark.config import WatermarkConfig
from dmark.watermark.watermark import Watermark


def calculate_zscore(
    output_ids: List[int],
    watermark: Watermark,
    prompt_ids: List[int]
) -> tuple[float, float]:
    """Calculate z-score for watermark detection.
    
    Args:
        output_ids: List of token IDs to check for watermark (generated tokens only)
        watermark: Watermark instance for efficient green list generation
        prompt_ids: List of prompt token IDs
    
    Returns:
        Tuple of (detection_rate, z_score)
    """
    detected = 0
    gen_len = 0
    
    for index in range(len(output_ids)):
        curr_token = output_ids[index]
        
        # Skip special tokens (EOS tokens)
        if curr_token == 126081 or curr_token == 126348:
            break
            
        # Always use i-1 token as previous token, regardless of strategy
        if index > 0:
            prev_token = output_ids[index - 1]
        else:
            # If index is 0, use last token of prompt
            if len(prompt_ids) > 0:
                prev_token = prompt_ids[-1]
            else:
                prev_token = 0
        
        # Generate green list using Watermark class (more efficient)
        green_list = watermark.gen_green_list(prev_token).bool()
        if green_list[curr_token]:
            detected += 1
        gen_len += 1
    
    if gen_len == 0:
        return 0.0, 0.0
    
    detection_rate = detected / gen_len
    z_score = 2 * (detected - gen_len * watermark.watermark_config.ratio) / math.sqrt(gen_len)
    
    return detection_rate, z_score


def process_json_file(
    input_file: str,
    output_file: str,
    bitmap_file: str = "bitmap.bin",
    model_name: str = "facebook/llada-760m-split2"
) -> None:
    """Process a JSON file containing generation results and add z-scores.
    
    Args:
        input_file: Path to input JSON file
        output_file: Path to output JSON file
        bitmap_file: Path to bitmap file for watermark detection
        model_name: Model name for tokenizer
    """
    # Load the JSON data
    with open(input_file, 'r') as f:
        results = json.load(f)
    
    # Initialize tokenizer
    if results[0].get("watermark_metadata") is None:
        return 
    
    # Extract watermark configuration
    wm_meta = results[0]["watermark_metadata"]
    watermark_config = WatermarkConfig(
        vocab_size=wm_meta.get("vocab_size", 126464),
        ratio=wm_meta.get("ratio", 0.5),
        delta=wm_meta.get("delta", 2.0),
        key=wm_meta.get("key", 42),
        prebias=wm_meta.get("prebias", False),
        strategy=wm_meta.get("strategy", "normal"),
        bitmap_path=bitmap_file
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    bitmap = PersistentBitmap(watermark_config.vocab_size, bitmap_file)
    watermark = Watermark(watermark_config, bitmap)
    
    # Process each result
    for result in tqdm(results, desc="Processing results"):
        # Get output IDs
        output_ids = result["data"]["output_ids"]
        
        # Tokenize prompt to get prompt_ids
        prompt_text = result["data"].get("prompt", "")
        if prompt_text:
            prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        else:
            prompt_ids = []
        
        # Calculate z-score
        detection_rate, z_score = calculate_zscore(
            output_ids,
            watermark,
            prompt_ids
        )
        
        # Add z-score and detection rate to result
        if "watermark" not in result:
            result["watermark"] = {}
        result["watermark"]["z_score"] = z_score
        result["watermark"]["detection_rate"] = detection_rate
    
    # Save results to new file
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"Processed {len(results)} results")
    print(f"Results saved to: {output_file}")
    
    # Print summary statistics
    z_scores = [r["watermark"]["z_score"] for r in results if r.get("watermark") is not None and r["watermark"].get("z_score") is not None]
    if z_scores:
        avg_z = sum(z_scores) / len(z_scores)
        max_z = max(z_scores)
        min_z = min(z_scores)
        print(f"Z-score statistics: avg={avg_z:.2f}, min={min_z:.2f}, max={max_z:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Calculate z-scores for watermark detection")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input directory or JSON file containing generation results"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Directory to save output files (default: same as input)"
    )
    parser.add_argument(
        "--bitmap",
        type=str,
        default="bitmap.bin",
        help="Path to bitmap file for watermark detection (default: bitmap.bin)"
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
    
    # Create output directory if specified
    if args.output:
        os.makedirs(args.output, exist_ok=True)
    
    # Determine if input is a file or directory
    if os.path.isfile(args.input):
        # Process single file
        if args.output:
            base_name = os.path.basename(args.input).replace(".json", "_zscore.json")
            output_file = os.path.join(args.output, base_name)
        else:
            output_file = args.input.replace(".json", "_zscore.json")
        process_json_file(args.input, output_file, args.bitmap, args.model)
    elif os.path.isdir(args.input):
        # Process all JSON files in directory
        json_files = [f for f in os.listdir(args.input) if f.endswith('.json') and not f.endswith('_zscore.json')]
        
        if not json_files:
            print(f"No JSON files found in directory: {args.input}")
            return
        
        print(f"Found {len(json_files)} JSON files to process")
        
        for json_file in json_files:
            input_path = os.path.join(args.input, json_file)
            
            if args.output:
                output_name = json_file.replace(".json", "_zscore.json")
                output_path = os.path.join(args.output, output_name)
            else:
                output_path = input_path.replace(".json", "_zscore.json")
            
            print(f"\nProcessing: {json_file}")
            process_json_file(input_path, output_path, args.bitmap, args.model)
    else:
        print(f"Error: '{args.input}' is neither a file nor a directory")
        return


if __name__ == "__main__":
    main()