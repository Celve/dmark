import argparse
import json

from pydantic import BaseModel
from typing import Any, Optional

import torch
import tqdm
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset

from dmark.llada.gen import generate
from dmark.llada.gen_legacy import generate_LLaDA
from dmark.watermark.config import WatermarkConfig
from dmark.watermark.persistent_bitmap import PersistentBitmap
from dmark.watermark.watermark import Watermark

device = "cuda"

class GenConfig(BaseModel):
    model: str
    dataset: str
    steps: int = 256
    gen_length: int = 256
    block_length: int = 32
    temperature: float = 0.0
    cfg_scale: float = 0.0
    remasking: str = "low_confidence"

class ExprConfig(BaseModel):
    num_samples: int
    output_dir: Optional[str]

def parse_args(): 
    parser = argparse.ArgumentParser()

    # we starts with dataset and model
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument(
        "--model",
        type=str,
        default="GSAI-ML/LLaDA-8B-Instruct",
        help="Model name or path",
    )

    # then we add number of samples and output directory
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default=None)

    # then we add generation arguments
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--gen_length", type=int, default=256)
    parser.add_argument("--block_length", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--cfg_scale", type=float, default=0.0)
    parser.add_argument("--remasking", type=str, default="low_confidence", choices=["low_confidence", "random"])

    # now it's time to add watermark arguments 
    # even though many of them have default values, the watermark will only be enabled if the strategy is not None
    parser.add_argument("--strategy", type=str, default=None, choices=["normal", "predict", "reverse", "legacy-ahead", "legacy-both"])
    parser.add_argument("--bitmap", type=str, default="bitmap.bin")
    parser.add_argument("--vocab_size", type=int, default=126464)
    parser.add_argument("--ratio", type=float, default=0.5)
    parser.add_argument("--delta", type=float, default=2.0)
    parser.add_argument("--key", type=int, default=42)
    parser.add_argument("--prebias", type=bool, default=False)


    args = parser.parse_args()

    gen_config = GenConfig(
        model=args.model,
        dataset=args.dataset,
        steps=args.steps,
        gen_length=args.gen_length,
        block_length=args.block_length,
        temperature=args.temperature,
        cfg_scale=args.cfg_scale,
        remasking=args.remasking,
    )

    watermark_config = WatermarkConfig(
        vocab_size=args.vocab_size,
        ratio=args.ratio,
        delta=args.delta,
        key=args.key,
        prebias=args.prebias,
        strategy=args.strategy,
    )

    expr_config = ExprConfig(
        num_samples=args.num_samples,
        output_dir=args.output_dir,
    )

    return gen_config, watermark_config, expr_config


def run_generation(
    gen_config: GenConfig,
    watermark_config: WatermarkConfig,
    expr_config: ExprConfig,
) -> list[dict[str, Any]]:
    """Run the generation process with optional watermarking."""
    # Load dataset
    dataset = load_dataset(gen_config.dataset, split="train")

    # Set up watermarking if config is provided
    if watermark_config.strategy is not None:
        bitmap = PersistentBitmap(watermark_config.vocab_size, gen_config.bitmap)
        watermark = Watermark(watermark_config, bitmap)
    else: 
        watermark = None

    # Load model
    model = (
        AutoModel.from_pretrained(
            gen_config.model,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        .to(device)
        .eval()
    )
    tokenizer = AutoTokenizer.from_pretrained(gen_config.model, trust_remote_code=True)

    results = []

    for i in tqdm(range(expr_config.num_samples), desc="Processing dataset"):
        prompt = dataset[i]["question"]
        gt = dataset[i]["answer"]
        m = [
            {"role": "user", "content": prompt},
        ]
        prompt = tokenizer.apply_chat_template(
            m, add_generation_prompt=True, tokenize=False
        )

        input_ids = tokenizer(prompt)["input_ids"]
        input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

        if (
            watermark_config.strategy == "legacy-ahead"
            or watermark_config.strategy == "legacy-both"
        ):
            out = generate_LLaDA(
                model,
                input_ids,
                steps=gen_config.steps,
                gen_length=gen_config.gen_length,
                block_length=gen_config.block_length,
                temperature=gen_config.temperature,
                cfg_scale=gen_config.cfg_scale,
                remasking=gen_config.remasking,
                watermark_config=watermark.watermark_config if watermark else None,
            )
        else:
            out = generate(
                model,
                input_ids,
                steps=gen_config.steps,
                gen_length=gen_config.gen_length,
                block_length=gen_config.block_length,
                temperature=gen_config.temperature,
                cfg_scale=gen_config.cfg_scale,
                remasking=gen_config.remasking,
                watermark=watermark,
            )

        output = tokenizer.batch_decode(
            out[:, input_ids.shape[1] :], skip_special_tokens=True
        )[0]
        
        results.append(
            {
                "data": 
                {
                    "prompt": prompt,
                    "ground_truth": gt,
                    "output": output,
                    "output_ids": out[:, input_ids.shape[1] :][0].tolist(),
                },
                "generation_metadata": gen_config.model_dump(),
                "watermark_metadata": watermark_config.model_dump() if watermark_config.strategy is not None else None,
            }
        )

    return results

if __name__ == "__main__":
    gen_config, watermark_config, expr_config = parse_args()
    results = run_generation(gen_config, watermark_config, expr_config)
    if expr_config.output_dir is not None:
        with open(expr_config.output_dir, "w") as f:
            json.dump(results, f, indent=4)
    else:
        print(results)