"""Perplexity annotator built atop the generic JSON processor."""

import argparse
from pathlib import Path
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from dmark.eval.process import process_dir, process_file

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

def _build_ppl_calculator(model_name: str, device: str) -> PPLCalculator:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    ).to(device)
    model.eval()
    return PPLCalculator(model, tokenizer, device)


def _build_transform(ppl_calc: PPLCalculator):
    def transform(instance: dict) -> dict:
        data = instance.get("data", {})
        prompt = data.get("prompt", "") or ""
        output = data.get("output", "") or ""
        text = prompt + output
        token_ids = ppl_calc.tokenizer(text, add_special_tokens=False).input_ids
        ppl = ppl_calc.analyze_tokens(token_ids)
        return {"perplexity": ppl}

    return transform


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Attach perplexity scores to generation outputs.")
    parser.add_argument("--input", type=Path, required=True, help="Input JSON file or directory.")
    parser.add_argument(
        "--output",
        type=Path,
        help="Output directory (for dir input) or output file path (for single file).",
    )
    parser.add_argument("--tag", type=str, default="ppl", help="Suffix for output file names.")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-4B-Instruct-2507",
        help="HF model to compute perplexity.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Computation device.",
    )
    inc_group = parser.add_mutually_exclusive_group()
    inc_group.add_argument(
        "--increment",
        action="store_true",
        default=True,
        help="Skip files whose tagged output already exists (default: on).",
    )
    inc_group.add_argument(
        "--no-increment",
        action="store_false",
        dest="increment",
        help="Process files even if tagged outputs already exist.",
    )
    parser.add_argument(
        "--insert-key",
        type=str,
        default="text_quality",
        help="Top-level key to insert perplexity dict into.",
    )
    return parser.parse_args()


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def main():
    args = parse_args()
    device = _resolve_device(args.device)
    ppl_calc = _build_ppl_calculator(args.model, device)
    transform = _build_transform(ppl_calc)

    if args.input.is_dir():
        output_dir = args.output or args.input.with_name(f"{args.input.name}_{args.tag}")
        process_dir(
            args.input,
            transform,
            insert_key=args.insert_key,
            output_dir=output_dir,
            lazy=args.increment,
        )
    else:
        input_file = args.input
        if args.output:
            output_dir = args.output
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{input_file.stem}_{args.tag}.json"
        else:
            output_path = input_file.with_name(f"{input_file.stem}_{args.tag}.json")

        process_file(
            input_file,
            transform,
            insert_key=args.insert_key,
            output_path=output_path,
            lazy=args.increment,
        )


if __name__ == "__main__":
    main()
