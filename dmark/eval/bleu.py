import argparse
import json
import os
from typing import List
from tqdm import tqdm
import sacrebleu


class BLEUCalculator:
    """BLEU calculator for text quality analysis."""

    def __init__(self) -> None:
        pass

    def analyze(self, text: str, reference: str) -> float:
        """Calculate the BLEU score of the given text with the reference."""
        b = sacrebleu.corpus_bleu([text], [[reference]]).score
        return b


def process_json_file(
    input_file: str,
    output_file: str,
    reference_field: str = "original_output"
) -> None:
    """Process a JSON file containing generation results and add BLEU scores.
    
    Args:
        input_file: Path to input JSON file
        output_file: Path to output JSON file
        reference_field: Field name containing reference text (default: "original_output")
    """
    # Load the JSON data
    with open(input_file, 'r') as f:
        results = json.load(f)
    
    # Initialize BLEU calculator
    bleu_calculator = BLEUCalculator()
    
    # Process each result
    for result in tqdm(results, desc="Calculating BLEU scores"):
        # Get output text and reference text
        output_text = result["data"].get("output", "")
        
        # Try to find reference text in different locations
        reference_text = None
        
        # First check if reference_field exists in data
        if reference_field in result["data"]:
            reference_text = result["data"][reference_field]
        # For attacked files, original_output might be stored
        elif "original_output" in result["data"]:
            reference_text = result["data"]["original_output"]
        # For non-attacked files, we might compare with prompt continuation
        elif "reference" in result["data"]:
            reference_text = result["data"]["reference"]
        
        # Calculate BLEU score if both texts exist
        if output_text and reference_text:
            bleu_score = bleu_calculator.analyze(output_text, reference_text)
        else:
            bleu_score = None
        
        # Add BLEU score to result under text_quality
        if "text_quality" not in result:
            result["text_quality"] = {}
        result["text_quality"]["bleu"] = bleu_score
        
        # Store which reference was used
        if bleu_score is not None:
            result["text_quality"]["bleu_reference_field"] = reference_field if reference_field in result["data"] else "original_output"
    
    # Save results to new file
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"Processed {len(results)} results")
    print(f"Results saved to: {output_file}")
    
    # Print summary statistics
    bleu_scores = [r["text_quality"]["bleu"] for r in results if r.get("text_quality") and r["text_quality"].get("bleu") is not None]
    if bleu_scores:
        import numpy as np
        avg_bleu = np.mean(bleu_scores)
        max_bleu = np.max(bleu_scores)
        min_bleu = np.min(bleu_scores)
        print(f"BLEU statistics: avg={avg_bleu:.2f}, min={min_bleu:.2f}, max={max_bleu:.2f}")
        print(f"Evaluated {len(bleu_scores)} samples with references")
    else:
        print("No reference texts found for BLEU calculation")


def main():
    parser = argparse.ArgumentParser(description="Calculate BLEU scores for generated text")
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
        help="Output directory path (default: {input_dir}_bleu for directories, same dir for files)"
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="bleu",
        help="Tag to append to output files/directory (default: bleu)"
    )
    parser.add_argument(
        "--reference_field",
        type=str,
        default="original_output",
        help="Field name containing reference text (default: original_output)"
    )
    
    args = parser.parse_args()
    
    # Check if input exists
    if not os.path.exists(args.input):
        print(f"Error: Input path '{args.input}' not found")
        return
    
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
        
        print(f"Found {len(json_files)} JSON files to process")
        print(f"Output directory: {output_dir}")
        
        for json_file in json_files:
            input_path = os.path.join(args.input, json_file)
            output_name = json_file.replace(".json", f"_{args.tag}.json")
            output_path = os.path.join(output_dir, output_name)
            
            print(f"\nProcessing: {json_file}")
            process_json_file(input_path, output_path, args.reference_field)
            
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
        
        process_json_file(args.input, output_file, args.reference_field)
    else:
        print(f"Error: '{args.input}' is neither a file nor a directory")
        return


if __name__ == "__main__":
    main()