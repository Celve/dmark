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
    model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
    device: str = "cuda"
) -> None:
    """Process a JSON file containing generation results and add perplexity scores.
    
    Args:
        input_file: Path to input JSON file
        output_file: Path to output JSON file
        model_name: Model name for perplexity calculation
        device: Device to run model on
    """
    # Check if CUDA is available
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU instead")
        device = "cpu"
    
    # Load the JSON data
    with open(input_file, 'r') as f:
        results = json.load(f)
    
    # Initialize model and tokenizer
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    ).to(device)
    model.eval()
    
    # Initialize PPL calculator
    ppl_calculator = PPLCalculator(model, tokenizer, device)
    
    # Process each result
    for result in tqdm(results, desc="Calculating perplexity"):
        prompt = result["data"]["prompt"]
        output = result["data"]["output"]
        full_text = prompt + output
        full_ids = tokenizer(full_text, add_special_tokens=False).input_ids
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
        help="Directory to save output files (default: same as input)"
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
            base_name = os.path.basename(args.input).replace(".json", "_ppl.json")
            output_file = os.path.join(args.output, base_name)
        else:
            output_file = args.input.replace(".json", "_ppl.json")
        process_json_file(args.input, output_file, args.model, args.device)
    elif os.path.isdir(args.input):
        # Process all JSON files in directory
        json_files = [f for f in os.listdir(args.input) if f.endswith('.json') and not f.endswith('_ppl.json')]
        
        if not json_files:
            print(f"No JSON files found in directory: {args.input}")
            return
        
        print(f"Found {len(json_files)} JSON files to process")
        
        for json_file in json_files:
            input_path = os.path.join(args.input, json_file)
            
            if args.output:
                output_name = json_file.replace(".json", "_ppl.json")
                output_path = os.path.join(args.output, output_name)
            else:
                output_path = input_path.replace(".json", "_ppl.json")
            
            print(f"\nProcessing: {json_file}")
            process_json_file(input_path, output_path, args.model, args.device)
    else:
        print(f"Error: '{args.input}' is neither a file nor a directory")
        return


if __name__ == "__main__":
    main()