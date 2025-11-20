"""CLI utility to attach watermark detection outputs to result JSON files.

Uses the detect() method from watermark implementations and the batch helpers
in dmark/eval/process.py to process either a single file or all JSON files in
an input directory.
"""

import argparse
from pathlib import Path
from typing import Callable, Any

import torch
from transformers import AutoTokenizer

from dmark.eval.process import process_file, process_dir
from dmark.gen.utils import build_watermark


def _build_watermark(args: argparse.Namespace):
    bitmap_path = None
    if args.strategy != "pattern-mark":
        ratio_int = int(args.ratio * 100)
        fname = f"bitmap_v{args.vocab_size}_r{ratio_int}_k{args.key}.bin"
        bitmap_path = Path(args.bitmap_dir) / fname
        if not bitmap_path.exists():
            raise FileNotFoundError(
                f"Bitmap not found at {bitmap_path}. "
                "Provide --bitmap-dir that contains generated bitmaps or adjust parameters."
            )

    cfg: dict[str, object] = {
        "vocab_size": args.vocab_size,
        "ratio": args.ratio,
        "delta": args.delta,
        "key": args.key,
        "strategy": args.strategy,
        "bitmap_path": str(bitmap_path) if bitmap_path is not None else "",
        "pattern_length": args.pattern_length,
    }
    return build_watermark(cfg, bitmap_device=args.bitmap_device, mask_id=args.mask_id)


def _make_transform(tokenizer, watermark, token_field: str, insert_key: str) -> Callable[[dict], dict]:
    def transform(instance: dict) -> dict:
        data = instance.get("data", {})
        tokens = data.get(token_field)
        if tokens is None:
            return {}

        prompt_text = data.get("prompt", "")
        prompt_ids = (
            tokenizer.encode(prompt_text, add_special_tokens=False) if prompt_text else []
        )

        flat_tokens = prompt_ids + tokens
        tensor = torch.tensor(flat_tokens, dtype=torch.long)
        detection = watermark.detect(tensor, len(prompt_ids))
        return detection

    return transform


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Attach watermark detect() results to JSON files.")
    parser.add_argument("--input", type=Path, required=True, help="Input JSON file or directory.")
    parser.add_argument("--output", type=Path, help="Output file when processing a single input file.")
    parser.add_argument("--output-dir", type=Path, help="Output directory when processing a directory.")
    parser.add_argument("--insert-key", type=str, default="watermark_detection", help="Field to insert results into.")
    parser.add_argument("--token-field", type=str, default="output_ids", help="Which data field holds generated ids.")
    parser.add_argument("--model", type=str, default="GSAI-ML/LLaDA-8B-Instruct", help="Tokenizer model name.")
    parser.add_argument("--mask-id", type=int, default=126336, help="Mask token id (passed to build_watermark).")
    parser.add_argument("--bitmap-device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--bitmap-dir", type=Path, default=Path("."), help="Directory containing bitmap files.")
    parser.add_argument("--vocab_size", type=int, default=126464)
    parser.add_argument("--ratio", type=float, default=0.5)
    parser.add_argument("--delta", type=float, default=2.0)
    parser.add_argument("--key", type=int, default=42)
    parser.add_argument(
        "--strategy",
        type=str,
        default="normal",
        choices=["normal", "predict", "bidirectional", "predict-bidirectional", "pattern-mark"],
    )
    parser.add_argument("--pattern_length", type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    watermark = _build_watermark(args)

    transform = _make_transform(tokenizer, watermark, args.token_field, args.insert_key)

    if args.input.is_file():
        process_file(
            args.input,
            transform,
            insert_key=args.insert_key,
            output_path=args.output,
        )
    else:
        if args.output_dir is None:
            raise ValueError("--output-dir is required when --input is a directory")
        process_dir(
            args.input,
            transform,
            insert_key=args.insert_key,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
