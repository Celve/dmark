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
    model_name: str = "GSAI-ML/LLaDA-8B-Instruct",
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
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None
    ).to(device)
    model.eval()
    
    # Initialize PPL calculator
    ppl_calculator = PPLCalculator(model, tokenizer, device)
    
    # Process each result
    for result in tqdm(results, desc="Calculating perplexity"):
        # Get output IDs
        output_ids = result["data"]["output_ids"]
        
        # Skip special tokens (EOS tokens) in output
        filtered_output = []
        for token in output_ids:
            if token == 126081 or token == 126348:
                break
            filtered_output.append(token)
        
        # Calculate perplexity
        if filtered_output and len(filtered_output) >= 2:
            perplexity = ppl_calculator.analyze_tokens(filtered_output)
        else:
            perplexity = None
        
        # Add perplexity to result
        result["perplexity"] = perplexity
    
    # Save results to new file
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"Processed {len(results)} results")
    print(f"Results saved to: {output_file}")
    
    # Print summary statistics
    perplexities = [r["perplexity"] for r in results if r.get("perplexity") is not None]
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
        help="Path to input JSON file containing generation results"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to output JSON file with perplexity scores added (default: input_ppl.json)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="GSAI-ML/LLaDA-8B-Instruct",
        help="Model name for perplexity calculation (default: GSAI-ML/LLaDA-8B-Instruct)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run model on (default: cuda)"
    )
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found")
        return
    
    # Set default output file if not provided
    if args.output is None:
        args.output = args.input.replace(".json", "_ppl.json")
    
    # Process the file
    process_json_file(args.input, args.output, args.model, args.device)


if __name__ == "__main__":
    main()