import argparse
import json
import os

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
    minimum_output_token: Optional[int]

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
    parser.add_argument("--minimum_output_token", type=int, default=None)
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
        minimum_output_token=args.minimum_output_token,
    )

    return gen_config, watermark_config, expr_config


def generate_result_filename(
    gen_config: GenConfig,
    watermark_config: WatermarkConfig,
    expr_config: ExprConfig,
) -> str:
    """Generate a comprehensive result filename based on all configuration parameters."""
    import datetime
    
    # Extract model name (last part after /)
    model_name = gen_config.model.split('/')[-1]
    
    # Start with base components
    components = [
        "results",
        gen_config.dataset.replace('/', '_'),
        model_name,
    ]
    
    # Add generation parameters
    components.extend([
        f"steps{gen_config.steps}",
        f"len{gen_config.gen_length}",
        f"blk{gen_config.block_length}",
        f"temp{gen_config.temperature}",
        f"cfg{gen_config.cfg_scale}",
        f"mask_{gen_config.remasking}",
    ])
    
    # Add watermark parameters if enabled
    if watermark_config.strategy is not None:
        components.append("wm")
        components.extend([
            f"r{watermark_config.ratio}",
            f"d{watermark_config.delta}",
            f"k{watermark_config.key}",
            watermark_config.strategy,
        ])
        
        # Add prebias if enabled
        if watermark_config.prebias:
            components.append("prebias")
        
        # Add vocab size if not default
        if watermark_config.vocab_size != 126464:
            components.append(f"v{watermark_config.vocab_size}")
    else:
        components.append("nowm")
    
    # Add number of samples
    components.append(f"n{expr_config.num_samples}")
    
    # Add minimum output token if specified
    if expr_config.minimum_output_token is not None:
        components.append(f"min{expr_config.minimum_output_token}")
    
    # Add timestamp for uniqueness
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    components.append(timestamp)
    
    # Join with underscores and add extension
    filename = "-".join(components) + ".json"
    
    return filename


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
        bitmap = PersistentBitmap(watermark_config.vocab_size, watermark_config.bitmap if hasattr(watermark_config, 'bitmap') else "bitmap.bin")
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
    dataset_idx = 0
    pbar = tqdm.tqdm(total=expr_config.num_samples, desc="Collecting valid samples")
    
    try:
        while len(results) < expr_config.num_samples and dataset_idx < len(dataset):
            prompt = dataset[dataset_idx]["question"]
            gt = dataset[dataset_idx]["answer"]
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

            output_ids = out[:, input_ids.shape[1] :][0]
            
            # Trim output_ids at first occurrence of special tokens (126081 or 126348)
            trimmed_length = len(output_ids)
            for i, curr_token in enumerate(output_ids):
                if curr_token == 126081 or curr_token == 126348:
                    trimmed_length = i
                    break
            output_ids = output_ids[:trimmed_length]
            
            # Decode the trimmed output_ids
            if trimmed_length > 0:
                output = tokenizer.decode(output_ids, skip_special_tokens=True)
            else:
                output = ""
            
            # Check if output meets minimum token requirement
            num_output_tokens = output_ids.shape[0]
            if expr_config.minimum_output_token is None or num_output_tokens >= expr_config.minimum_output_token:
                results.append(
                    {
                        "data": 
                        {
                            "prompt": prompt,
                            "ground_truth": gt,
                            "output": output,
                            "output_ids": output_ids.tolist(),
                            "num_output_tokens": num_output_tokens,
                        },
                        "generation_metadata": gen_config.model_dump(),
                        "watermark_metadata": watermark_config.model_dump() if watermark_config.strategy is not None else None,
                    }
                )
                pbar.update(1)
            else:
                pbar.set_postfix({"skipped": f"{num_output_tokens} < {expr_config.minimum_output_token}"})
            
            dataset_idx += 1
    
    except KeyboardInterrupt:
        print(f"\n\nInterrupted! Collected {len(results)} samples so far.")
        pbar.close()
        return results
    
    pbar.close()
    
    if len(results) < expr_config.num_samples:
        print(f"Warning: Only collected {len(results)} valid samples out of {expr_config.num_samples} requested.")
        print(f"Dataset exhausted at index {dataset_idx}.")

    return results

if __name__ == "__main__":
    gen_config, watermark_config, expr_config = parse_args()
    results = run_generation(gen_config, watermark_config, expr_config)
    
    # Save results if any were collected
    if results:
        if expr_config.output_dir is not None:
            os.makedirs(expr_config.output_dir, exist_ok=True)
            output_filename = generate_result_filename(gen_config, watermark_config, expr_config)
            output_filename = os.path.join(expr_config.output_dir, output_filename)
            # Save results to file
            with open(output_filename, "w") as f:
                json.dump(results, f, indent=4)
            print(f"Results saved to: {output_filename}")
            print(f"Total samples collected: {len(results)}")
        else: 
            print(results)
    else:
        print("No results were collected.")