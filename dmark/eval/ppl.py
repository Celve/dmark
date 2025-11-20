"""Perplexity annotator built atop the generic JSON processor."""

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from dmark.eval.process import process_file
from dmark.eval.ppl_legacy import PPLCalculator


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
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="HF model to compute perplexity.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Computation device.",
    )
    parser.add_argument(
        "--increment",
        action="store_true",
        help="Skip files whose tagged output already exists.",
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


def _process_dir(
    input_dir: Path,
    output_dir: Path,
    transform,
    insert_key: str,
    tag: str,
    increment: bool,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in input_dir.iterdir():
        if path.is_dir() or path.suffix != ".json" or path.name.startswith("_"):
            continue
        out_path = output_dir / f"{path.stem}_{tag}.json"
        if increment and out_path.exists():
            print(f"⏭️  Skipping {path.name} (exists).")
            continue
        process_file(path, transform, insert_key=insert_key, output_path=out_path)


def main():
    args = parse_args()
    device = _resolve_device(args.device)
    ppl_calc = _build_ppl_calculator(args.model, device)
    transform = _build_transform(ppl_calc)

    if args.input.is_dir():
        output_dir = args.output or args.input.with_name(f"{args.input.name}_{args.tag}")
        _process_dir(
            args.input,
            output_dir,
            transform,
            insert_key=args.insert_key,
            tag=args.tag,
            increment=args.increment,
        )
    else:
        input_file = args.input
        if args.output:
            output_dir = args.output
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{input_file.stem}_{args.tag}.json"
        else:
            output_path = input_file.with_name(f"{input_file.stem}_{args.tag}.json")

        if args.increment and Path(output_path).exists():
            print(f"⏭️  Skipping {output_path} (exists).")
            return

        process_file(
            input_file,
            transform,
            insert_key=args.insert_key,
            output_path=output_path,
        )


if __name__ == "__main__":
    main()
