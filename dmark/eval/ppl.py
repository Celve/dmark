import argparse
import json
import os
from typing import List
from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class PPLCalculator:
    """Perplexity calculator for text quality analysis."""

    def __init__(self, model, tokenizer, device='cuda') -> None:
        """
        Initialize the perplexity calculator.

        Parameters:
            model: The language model for perplexity calculation.
            tokenizer: The tokenizer for the language model.
            device (str): The device to use for the calculation.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.criterion = torch.nn.CrossEntropyLoss()

    def analyze_tokens(self, token_ids: List[int]) -> float:
        """Calculate the perplexity of the given token IDs."""
        if len(token_ids) < 2:
            return float('inf')
        
        with torch.no_grad():
            encoded_text = torch.tensor(token_ids, dtype=torch.long).to(self.device)
            logits = self.model(torch.unsqueeze(encoded_text, 0), return_dict=True).logits[0]
            loss = self.criterion(logits[:-1], encoded_text[1:])
            ppl = torch.exp(loss)
        return ppl.item()


def process_json_file(
    input_file: str,
    output_file: str,
    ppl_calculator: PPLCalculator
) -> bool:
    """Process a JSON file containing generation results and add perplexity scores.

    Args:
        input_file: Path to input JSON file
        output_file: Path to output JSON file
        ppl_calculator: PPLCalculator instance for perplexity calculation

    Returns:
        True if processing successful, False if file could not be read
    """
    # Try to load the JSON data
    try:
        with open(input_file, 'r') as f:
            results = json.load(f)
    except (IOError, OSError, json.JSONDecodeError) as e:
        print(f"⚠️  Skipping {os.path.basename(input_file)}: Cannot read file ({type(e).__name__}: {e})")
        return False
    
    # Process each result
    for result in tqdm(results, desc="Calculating perplexity"):
        if "data" not in result:
            continue
        prompt = result["data"]["prompt"]
        output = result["data"]["output"]
        full_text = prompt + output
        full_ids = ppl_calculator.tokenizer(full_text, add_special_tokens=False).input_ids
        perplexity = ppl_calculator.analyze_tokens(full_ids)
        
        # Add perplexity to result under text_quality
        if "text_quality" not in result:
            result["text_quality"] = {}
        result["text_quality"]["perplexity"] = perplexity
    
    # Save results to new file
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Processed {len(results)} results")
    print(f"Results saved to: {output_file}")

    # Print summary statistics
    perplexities = [r["text_quality"]["perplexity"] for r in results if r.get("text_quality") and r["text_quality"].get("perplexity") is not None]
    if perplexities:
        avg_ppl = sum(perplexities) / len(perplexities)
        max_ppl = max(perplexities)
        min_ppl = min(perplexities)
        print(f"Perplexity statistics: avg={avg_ppl:.2f}, min={min_ppl:.2f}, max={max_ppl:.2f}")

    return True


def main():
    parser = argparse.ArgumentParser(description="Calculate perplexity for generated text")
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
        help="Output directory path (default: {input_dir}_ppl for directories, same dir for files)"
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="ppl",
        help="Tag to append to output files/directory (default: ppl)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="Model name for perplexity calculation (default: meta-llama/Meta-Llama-3-8B-Instruct)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run model on (default: cuda)"
    )
    parser.add_argument(
        "--increment",
        action="store_true",
        help="Increment mode: only process files that don't have output yet"
    )

    args = parser.parse_args()
    
    # Check if input exists
    if not os.path.exists(args.input):
        print(f"Error: Input path '{args.input}' not found")
        return
    
    # Check if CUDA is available
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU instead")
        device = "cpu"
    
    # Initialize model and tokenizer once
    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    ).to(device)
    model.eval()
    
    # Initialize PPL calculator once
    ppl_calculator = PPLCalculator(model, tokenizer, device)
    
    # Determine output directory based on input type
    if os.path.isdir(args.input):
        # Input is directory
        if args.output:
            output_dir = args.output
        else:
            # Default: {input_dir}_{tag}
            input_dirname = os.path.basename(os.path.normpath(args.input))
            parent_dir = os.path.dirname(os.path.normpath(args.input))
            output_dir = os.path.join(parent_dir, f"{input_dirname}_{args.tag}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Process all JSON files in directory (excluding already tagged files and metadata files)
        tag_suffix = f"_{args.tag}.json"
        json_files = [f for f in os.listdir(args.input)
                     if f.endswith('.json') and not f.endswith(tag_suffix) and not f.startswith('_')]

        if not json_files:
            print(f"No JSON files found in directory: {args.input}")
            return

        # Filter files based on increment mode
        files_to_process = []
        files_skipped_existing = []

        for json_file in json_files:
            output_name = json_file.replace(".json", f"_{args.tag}.json")
            output_path = os.path.join(output_dir, output_name)

            if args.increment and os.path.exists(output_path):
                files_skipped_existing.append(json_file)
            else:
                files_to_process.append(json_file)

        print(f"📁 Directory: {args.input}")
        print(f"📊 Total JSON files found: {len(json_files)}")
        if args.increment and files_skipped_existing:
            print(f"⏭️  Already processed (will skip): {len(files_skipped_existing)}")
        print(f"🔄 Files to process: {len(files_to_process)}")
        print(f"📂 Output directory: {output_dir}")

        if not files_to_process:
            print("No new files to process")
            return

        successful = 0
        failed = 0
        skipped = len(files_skipped_existing)

        for json_file in files_to_process:
            input_path = os.path.join(args.input, json_file)
            output_name = json_file.replace(".json", f"_{args.tag}.json")
            output_path = os.path.join(output_dir, output_name)

            print(f"\nProcessing: {json_file}")
            if process_json_file(input_path, output_path, ppl_calculator):
                successful += 1
            else:
                failed += 1

        # Print summary
        print(f"\n{'='*50}")
        print(f"Processing complete:")
        print(f"  ✅ Successfully processed: {successful} files")
        if failed > 0:
            print(f"  ❌ Failed/Skipped: {failed} files")
        if args.increment and skipped > 0:
            print(f"  ⏭️  Already processed: {skipped} files")
            
    elif os.path.isfile(args.input):
        # Input is a file
        if args.output:
            # Save to specified output directory
            os.makedirs(args.output, exist_ok=True)
            base_name = os.path.basename(args.input).replace(".json", f"_{args.tag}.json")
            output_file = os.path.join(args.output, base_name)
        else:
            # Save alongside input file with tag
            output_file = args.input.replace(".json", f"_{args.tag}.json")

        # Check if output exists in increment mode
        if args.increment and os.path.exists(output_file):
            print(f"⏭️  Skipping (output exists): {output_file}")
            return

        if not process_json_file(args.input, output_file, ppl_calculator):
            print(f"❌ Failed to process file")
    else:
        print(f"Error: '{args.input}' is neither a file nor a directory")
        return


if __name__ == "__main__":
    main()