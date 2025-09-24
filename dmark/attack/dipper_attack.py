import argparse
import json
import os
import time
from typing import Dict, Any, List, Tuple, Optional
from tqdm import tqdm
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoTokenizer
import nltk
from nltk.tokenize import sent_tokenize
from datetime import datetime
import numpy as np

try:
    nltk.download('punkt', quiet=True)
except:
    pass

class DipperParaphraser(object):
    def __init__(self, model="kalpeshk2011/dipper-paraphraser-xxl", verbose=True):
        time1 = time.time()
        self.tokenizer = T5Tokenizer.from_pretrained('google/t5-v1_1-xxl')
        self.model = T5ForConditionalGeneration.from_pretrained(model)
        if verbose:
            print(f"{model} model loaded in {time.time() - time1:.2f} seconds")

        if torch.cuda.is_available():
            self.model.cuda()
        self.model.eval()

    def paraphrase(self, input_text, lex_diversity, order_diversity, prefix="", sent_interval=3, **kwargs):
        """Paraphrase a text using the DIPPER model.

        Args:
            input_text (str): The text to paraphrase.
            lex_diversity (int): The lexical diversity of the output (0, 20, 40, 60, 80, 100).
            order_diversity (int): The order diversity of the output (0, 20, 40, 60, 80, 100).
            prefix (str): Optional prefix context for the paraphrasing.
            sent_interval (int): Number of sentences to paraphrase at once.
            **kwargs: Additional keyword arguments like top_p, top_k, max_length.
        """
        assert lex_diversity in [0, 20, 40, 60, 80, 100], "Lexical diversity must be one of 0, 20, 40, 60, 80, 100."
        assert order_diversity in [0, 20, 40, 60, 80, 100], "Order diversity must be one of 0, 20, 40, 60, 80, 100."

        lex_code = int(100 - lex_diversity)
        order_code = int(100 - order_diversity)

        input_text = " ".join(input_text.split())
        sentences = sent_tokenize(input_text)
        prefix = " ".join(prefix.replace("\n", " ").split())
        output_text = ""

        for sent_idx in range(0, len(sentences), sent_interval):
            curr_sent_window = " ".join(sentences[sent_idx:sent_idx + sent_interval])
            final_input_text = f"lexical = {lex_code}, order = {order_code}"
            if prefix:
                final_input_text += f" {prefix}"
            final_input_text += f" <sent> {curr_sent_window} </sent>"

            final_input = self.tokenizer([final_input_text], return_tensors="pt")
            if torch.cuda.is_available():
                final_input = {k: v.cuda() for k, v in final_input.items()}

            with torch.inference_mode():
                outputs = self.model.generate(**final_input, **kwargs)
            outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            prefix += " " + outputs[0]
            output_text += " " + outputs[0]

        return output_text.strip()


class DipperAttackProcessor:
    """Processor for applying DIPPER paraphrase attacks to JSON files."""

    def __init__(self,
                 lex_diversity: int = 60,
                 order_diversity: int = 60,
                 sent_interval: int = 3,
                 min_output_length: int = 200,
                 model_name: str = "GSAI-ML/LLaDA-8B-Instruct",
                 dipper_model: str = "kalpeshk2011/dipper-paraphraser-xxl",
                 max_length: int = 512,
                 top_p: float = 0.9,
                 top_k: int = 50,
                 do_sample: bool = True):
        """Initialize the DIPPER attack processor.

        Args:
            lex_diversity: Lexical diversity for paraphrasing (0-100 in steps of 20)
            order_diversity: Order diversity for paraphrasing (0-100 in steps of 20)
            sent_interval: Number of sentences to paraphrase at once
            min_output_length: Minimum output length to truncate to before attack
            model_name: Name of the model for tokenizer
            dipper_model: Name of the DIPPER model to use
            max_length: Maximum length for generation
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            do_sample: Whether to use sampling
        """
        self.lex_diversity = lex_diversity
        self.order_diversity = order_diversity
        self.sent_interval = sent_interval
        self.min_output_length = min_output_length
        self.model_name = model_name
        self.dipper_model = dipper_model
        self.max_length = max_length
        self.top_p = top_p
        self.top_k = top_k
        self.do_sample = do_sample

        self.paraphraser = None
        self.tokenizer = None

    def initialize(self):
        """Initialize the paraphraser and tokenizer."""
        print("Loading DIPPER model...")
        self.paraphraser = DipperParaphraser(model=self.dipper_model, verbose=True)

        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)

    def truncate_tokens(self, token_ids: List[int]) -> Tuple[List[int], bool, int]:
        """Truncate tokens to min_output_length if needed."""
        original_length = len(token_ids)
        if original_length > self.min_output_length:
            return token_ids[:self.min_output_length], True, original_length
        return token_ids, False, original_length

    def get_metadata(self, original_length: int, attacked_length: int, paraphrase_params: dict) -> Dict[str, Any]:
        """Generate metadata for the attack."""
        return {
            'type': 'dipper_paraphrase',
            'original_length': original_length,
            'attacked_length': attacked_length,
            'length_change': attacked_length - original_length,
            'length_change_ratio': (attacked_length - original_length) / original_length if original_length > 0 else 0,
            'paraphrase_params': paraphrase_params
        }

    def process_sample(self, result: dict, idx: int) -> dict:
        """Process a single sample with DIPPER paraphrasing."""
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

        # Decode to text
        original_text = self.tokenizer.decode(output_ids, skip_special_tokens=True)

        if not original_text.strip():
            return attacked_result

        try:
            # Apply DIPPER paraphrasing
            paraphrased_text = self.paraphraser.paraphrase(
                original_text,
                lex_diversity=self.lex_diversity,
                order_diversity=self.order_diversity,
                sent_interval=self.sent_interval,
                max_length=self.max_length,
                top_p=self.top_p,
                top_k=self.top_k,
                do_sample=self.do_sample
            )

            # Encode paraphrased text back to token IDs
            attacked_ids = self.tokenizer.encode(paraphrased_text, add_special_tokens=False)

            # Add attacked data to result
            attacked_result['data']['attacked_text'] = paraphrased_text
            attacked_result['data']['attacked_ids'] = attacked_ids

            # Create attack metadata
            paraphrase_params = {
                'lex_diversity': self.lex_diversity,
                'order_diversity': self.order_diversity,
                'sent_interval': self.sent_interval,
                'max_length': self.max_length,
                'top_p': self.top_p,
                'top_k': self.top_k,
                'do_sample': self.do_sample,
                'dipper_model': self.dipper_model
            }

            attack_metadata = self.get_metadata(len(output_ids), len(attacked_ids), paraphrase_params)
            attack_metadata['source_field'] = source_field
            attack_metadata['pre_attack_truncation'] = {
                'applied': truncation_applied,
                'original_full_length': original_full_length,
                'truncated_to': len(output_ids) if truncation_applied else original_full_length,
                'min_output_length': self.min_output_length
            }

            attacked_result['attack_metadata'] = attack_metadata

        except Exception as e:
            print(f"  Warning: Failed to paraphrase sample {idx}: {e}")
            attacked_result['attack_metadata'] = {
                'type': 'dipper_paraphrase',
                'error': str(e),
                'original_length': len(output_ids)
            }

        return attacked_result

    def process_file(self, input_file: str, output_file: str) -> Dict[str, Any]:
        """Process a JSON file and apply DIPPER attacks."""
        with open(input_file, 'r') as f:
            results = json.load(f)

        stats = {
            'total_samples': len(results),
            'successful_attacks': 0,
            'failed_attacks': 0,
            'original_lengths': [],
            'attacked_lengths': [],
            'length_changes': []
        }

        attacked_results = []

        try:
            for idx, result in enumerate(tqdm(results, desc="Applying DIPPER attacks", leave=False)):
                attacked_result = self.process_sample(result, idx)

                # Update statistics
                if 'attack_metadata' in attacked_result and 'error' not in attacked_result['attack_metadata']:
                    metadata = attacked_result['attack_metadata']
                    stats['successful_attacks'] += 1
                    stats['original_lengths'].append(metadata['original_length'])
                    stats['attacked_lengths'].append(metadata['attacked_length'])
                    stats['length_changes'].append(metadata['length_change'])
                else:
                    stats['failed_attacks'] += 1

                attacked_results.append(attacked_result)

        except KeyboardInterrupt:
            print(f"\n\nInterrupted! Saving {len(attacked_results)} processed samples to partial file...")
            # Save partial results
            if output_file.endswith('.json'):
                partial_file = output_file[:-5] + '.partial.json'
            else:
                partial_file = output_file + '.partial'

            with open(partial_file, 'w') as f:
                json.dump(attacked_results, f, indent=4)

            print(f"Partial results saved to: {partial_file}")
            print(f"Processed {len(attacked_results)}/{len(results)} samples")
            print(f"Successful: {stats['successful_attacks']}, Failed: {stats['failed_attacks']}")
            return stats

        # Save results
        with open(output_file, 'w') as f:
            json.dump(attacked_results, f, indent=4)

        return stats

    def process_directory(self, input_dir: str, output_dir: str = None) -> None:
        """Process all JSON files in a directory."""
        # Auto-generate output directory name if not provided
        if output_dir is None:
            base_dir = os.path.dirname(input_dir.rstrip('/'))
            dir_name = os.path.basename(input_dir.rstrip('/'))
            output_dir = os.path.join(base_dir, f"{dir_name}_dipper")

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Find all JSON files
        json_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.json')])

        if not json_files:
            print(f"No JSON files found in {input_dir}")
            return

        self.print_configuration(input_dir, output_dir, len(json_files))

        # Initialize once for all files
        self.initialize()

        # Track overall statistics
        overall_stats = {
            'files_processed': 0,
            'total_samples': 0,
            'successful_attacks': 0,
            'failed_attacks': 0,
            'all_original_lengths': [],
            'all_attacked_lengths': [],
            'all_length_changes': []
        }

        # Process each file
        for json_file in tqdm(json_files, desc="Processing files"):
            input_path = os.path.join(input_dir, json_file)

            # Add suffix to filename
            base_name = json_file[:-5] if json_file.endswith('.json') else json_file
            output_filename = f"{base_name}_dipper.json"
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
                overall_stats['all_length_changes'].extend(stats['length_changes'])

                print(f"✓ {json_file}: {stats['successful_attacks']}/{stats['total_samples']} samples paraphrased")

            except Exception as e:
                print(f"✗ {json_file}: Error - {e}")

        # Save and print summary
        self.save_summary(output_dir, input_dir, overall_stats)
        self.print_summary(overall_stats, len(json_files), output_dir)

    def print_configuration(self, input_dir: str, output_dir: str, num_files: int):
        """Print attack configuration."""
        print(f"\n{'='*70}")
        print(f"DIPPER Attack Configuration")
        print(f"{'='*70}")
        print(f"Input directory: {input_dir}")
        print(f"Output directory: {output_dir}")
        print(f"Lexical diversity: {self.lex_diversity}")
        print(f"Order diversity: {self.order_diversity}")
        print(f"Sentence interval: {self.sent_interval}")
        print(f"Min output length (truncate before attack): {self.min_output_length}")
        print(f"Files to process: {num_files}")
        print(f"Model: {self.model_name}")
        print(f"DIPPER model: {self.dipper_model}")
        print(f"{'='*70}\n")

    def save_summary(self, output_dir: str, input_dir: str, stats: Dict[str, Any]):
        """Save attack summary to JSON file."""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'input_directory': input_dir,
            'output_directory': output_dir,
            'attack_configuration': {
                'type': 'dipper_paraphrase',
                'lex_diversity': self.lex_diversity,
                'order_diversity': self.order_diversity,
                'sent_interval': self.sent_interval,
                'min_output_length': self.min_output_length,
                'model': self.model_name,
                'dipper_model': self.dipper_model,
                'max_length': self.max_length,
                'top_p': self.top_p,
                'top_k': self.top_k,
                'do_sample': self.do_sample
            },
            'statistics': {
                'files_processed': stats['files_processed'],
                'total_samples': stats['total_samples'],
                'successful_attacks': stats['successful_attacks'],
                'failed_attacks': stats['failed_attacks'],
                'success_rate': stats['successful_attacks'] / stats['total_samples'] if stats['total_samples'] > 0 else 0
            }
        }

        if stats['all_original_lengths']:
            summary['statistics']['length_statistics'] = {
                'avg_original_length': float(np.mean(stats['all_original_lengths'])),
                'avg_attacked_length': float(np.mean(stats['all_attacked_lengths'])),
                'avg_length_change': float(np.mean(stats['all_length_changes'])),
                'std_original_length': float(np.std(stats['all_original_lengths'])),
                'std_attacked_length': float(np.std(stats['all_attacked_lengths'])),
                'std_length_change': float(np.std(stats['all_length_changes'])),
                'avg_length_change_ratio': float(np.mean([
                    c / o if o > 0 else 0
                    for c, o in zip(stats['all_length_changes'], stats['all_original_lengths'])
                ]))
            }

        summary_path = os.path.join(output_dir, '_dipper_attack_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

    def print_summary(self, stats: Dict[str, Any], num_files: int, output_dir: str):
        """Print final summary."""
        print(f"\n{'='*70}")
        print(f"DIPPER Attack Summary")
        print(f"{'='*70}")
        print(f"Files processed: {stats['files_processed']}/{num_files}")
        print(f"Total samples: {stats['total_samples']}")
        print(f"Successful paraphrases: {stats['successful_attacks']}")
        print(f"Failed paraphrases: {stats['failed_attacks']}")

        if stats['all_original_lengths']:
            print(f"\nLength Statistics:")
            print(f"  Average original length: {np.mean(stats['all_original_lengths']):.1f} ± {np.std(stats['all_original_lengths']):.1f}")
            print(f"  Average attacked length: {np.mean(stats['all_attacked_lengths']):.1f} ± {np.std(stats['all_attacked_lengths']):.1f}")
            print(f"  Average length change: {np.mean(stats['all_length_changes']):.1f} ± {np.std(stats['all_length_changes']):.1f}")

            avg_ratio = np.mean([
                c / o if o > 0 else 0
                for c, o in zip(stats['all_length_changes'], stats['all_original_lengths'])
            ])
            print(f"  Average length change ratio: {avg_ratio:.1%}")

        print(f"\nOutput directory: {output_dir}")
        print(f"Attack summary saved to: {os.path.join(output_dir, '_dipper_attack_summary.json')}")
        print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(
        description="Apply DIPPER paraphrase attacks to watermarked text in JSON files"
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
        help="Output directory (auto-generated with '_dipper' suffix if not specified)"
    )

    parser.add_argument(
        "--lex-diversity",
        type=int,
        default=60,
        choices=[0, 20, 40, 60, 80, 100],
        help="Lexical diversity for paraphrasing (default: 60)"
    )

    parser.add_argument(
        "--order-diversity",
        type=int,
        default=60,
        choices=[0, 20, 40, 60, 80, 100],
        help="Order diversity for paraphrasing (default: 60)"
    )

    parser.add_argument(
        "--sent-interval",
        type=int,
        default=3,
        help="Number of sentences to paraphrase at once (default: 3)"
    )

    parser.add_argument(
        "--min-output-length",
        type=int,
        default=200,
        help="Minimum output length to truncate to before applying attack (default: 200)"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="GSAI-ML/LLaDA-8B-Instruct",
        help="Model name for tokenizer (default: GSAI-ML/LLaDA-8B-Instruct)"
    )

    parser.add_argument(
        "--dipper-model",
        type=str,
        default="kalpeshk2011/dipper-paraphraser-xxl",
        help="DIPPER model to use (default: kalpeshk2011/dipper-paraphraser-xxl)"
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum length for generation (default: 512)"
    )

    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p sampling parameter (default: 0.9)"
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k sampling parameter (default: 50)"
    )

    parser.add_argument(
        "--no-sampling",
        action="store_true",
        help="Disable sampling (use greedy decoding)"
    )

    args = parser.parse_args()

    # Check if input exists
    if not os.path.exists(args.input):
        print(f"Error: Input path '{args.input}' not found")
        return

    # Create processor
    processor = DipperAttackProcessor(
        lex_diversity=args.lex_diversity,
        order_diversity=args.order_diversity,
        sent_interval=args.sent_interval,
        min_output_length=args.min_output_length,
        model_name=args.model,
        dipper_model=args.dipper_model,
        max_length=args.max_length,
        top_p=args.top_p,
        top_k=args.top_k,
        do_sample=not args.no_sampling
    )

    # Process based on input type
    if os.path.isfile(args.input):
        # Single file processing
        if args.output is None:
            dir_path = os.path.dirname(args.input)
            base_name = os.path.splitext(os.path.basename(args.input))[0]
            output_file = os.path.join(dir_path, f"{base_name}_dipper.json")
        else:
            output_file = args.output

        # Ensure output directory exists
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        print(f"Processing single file: {args.input}")
        print(f"Output: {output_file}")

        # Process file
        processor.initialize()
        stats = processor.process_file(args.input, output_file)

        print(f"\nDIPPER attack complete!")
        print(f"Samples processed: {stats['total_samples']}")
        print(f"Successful paraphrases: {stats['successful_attacks']}")
        print(f"Output saved to: {output_file}")

    elif os.path.isdir(args.input):
        # Directory processing
        processor.process_directory(args.input, args.output)
    else:
        print(f"Error: '{args.input}' is neither a file nor a directory")


if __name__ == "__main__":
    main()
