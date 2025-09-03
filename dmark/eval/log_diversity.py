import argparse
import json
import math
import os
from typing import Dict, List, Set, Tuple
from tqdm import tqdm


class LogDiversityAnalyzer:
    """Log diversity analyzer for text quality analysis."""
    
    def __init__(self) -> None:
        pass

    def _eval_text(self, text: str, ngram: int) -> Tuple[int, int]:
        """Evaluate text to compute the number of unique and total n-grams."""
        tokens = text.split()
        ngram_set = set()
        total_ngrams = 0

        for i in range(len(tokens) - ngram + 1):
            ngram_set.add(" ".join(tokens[i:i + ngram]))
            total_ngrams += 1

        return len(ngram_set), total_ngrams

    def _eval_one_instance(self, text: str, ngram_list: List[int]) -> Tuple[Dict, Set]:
        """Evaluate a single text instance for multiple n-gram lengths."""
        results = {}
        for n in ngram_list:
            unique, total = self._eval_text(text, n)
            results[n] = {"unique": unique, "total": total}
        unique_tokens = set(text.split())
        return results, unique_tokens

    def analyze(self, text: str) -> float:
        """Analyze text to compute log diversity based on n-gram uniqueness."""
        ngram_list = [2, 3, 4]
        prediction_results = {n: {"unique": 0, "total": 0} for n in ngram_list}
        unique_token_set = set()

        stripped_text = text.strip()
        
        # Handle empty text
        if not stripped_text:
            return 0.0
        
        ngram_results, unique_tokens = self._eval_one_instance(stripped_text, ngram_list)

        unique_token_set.update(unique_tokens)

        for n in ngram_list:
            prediction_results[n]["unique"] += ngram_results[n]["unique"]
            prediction_results[n]["total"] += ngram_results[n]["total"]

        # Compute diversity scores for each n-gram length
        diversity_scores = []
        for n in ngram_list:
            if prediction_results[n]["total"] > 0:
                diversity_score = 1 - (prediction_results[n]["unique"] / prediction_results[n]["total"])
            else:
                diversity_score = 0
            diversity_scores.append(diversity_score)

        # Overall diversity is the product of individual n-gram diversities
        overall_diversity = (1 - diversity_scores[0] / 100) * (1 - diversity_scores[1] / 100) * (1 - diversity_scores[2] / 100)
        log_diversity = -math.log(max(1 - overall_diversity, math.exp(-20)))

        return log_diversity


def process_json_file(
    input_file: str,
    output_file: str
) -> None:
    """Process a JSON file containing generation results and add log diversity scores.
    
    Args:
        input_file: Path to input JSON file
        output_file: Path to output JSON file
    """
    # Load the JSON data
    with open(input_file, 'r') as f:
        results = json.load(f)
    
    # Initialize log diversity analyzer
    analyzer = LogDiversityAnalyzer()
    
    # Process each result
    for result in tqdm(results, desc="Calculating log diversity"):
        # Get output text
        output_text = result["data"].get("output", "")
        
        # Calculate log diversity
        if output_text:
            log_diversity = analyzer.analyze(output_text)
        else:
            log_diversity = None
        
        # Add log diversity to result under text_quality
        if "text_quality" not in result:
            result["text_quality"] = {}
        result["text_quality"]["log_diversity"] = log_diversity
    
    # Save results to new file
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"Processed {len(results)} results")
    print(f"Results saved to: {output_file}")
    
    # Print summary statistics
    diversities = [r["text_quality"]["log_diversity"] for r in results if r.get("text_quality") and r["text_quality"].get("log_diversity") is not None]
    if diversities:
        import numpy as np
        avg_diversity = np.mean(diversities)
        max_diversity = np.max(diversities)
        min_diversity = np.min(diversities)
        print(f"Log diversity statistics: avg={avg_diversity:.4f}, min={min_diversity:.4f}, max={max_diversity:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Calculate log diversity for generated text")
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
            base_name = os.path.basename(args.input).replace(".json", "_log_diversity.json")
            output_file = os.path.join(args.output, base_name)
        else:
            output_file = args.input.replace(".json", "_log_diversity.json")
        process_json_file(args.input, output_file)
    elif os.path.isdir(args.input):
        # Process all JSON files in directory
        json_files = [f for f in os.listdir(args.input) if f.endswith('.json') and not f.endswith('_log_diversity.json')]
        
        if not json_files:
            print(f"No JSON files found in directory: {args.input}")
            return
        
        print(f"Found {len(json_files)} JSON files to process")
        
        for json_file in json_files:
            input_path = os.path.join(args.input, json_file)
            
            if args.output:
                output_name = json_file.replace(".json", "_log_diversity.json")
                output_path = os.path.join(args.output, output_name)
            else:
                output_path = input_path.replace(".json", "_log_diversity.json")
            
            print(f"\nProcessing: {json_file}")
            process_json_file(input_path, output_path)
    else:
        print(f"Error: '{args.input}' is neither a file nor a directory")
        return


if __name__ == "__main__":
    main()