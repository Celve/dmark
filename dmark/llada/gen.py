import argparse
import json
import os
from typing import Any, Optional

import numpy as np
import torch 
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from dmark.llada.gen_legacy import generate_LLaDA
from dmark.watermark.legacy import detect
from dmark.watermark.config import WatermarkConfig
from dmark.watermark.detect import Detector
from dmark.watermark.persistent_bitmap import PersistentBitmap
from dmark.watermark.ppl import PerplexityEval
from dmark.watermark.watermark import Watermark


def add_gumbel_noise(logits, temperature):
    """
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    """
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    """
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    """
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = (
        torch.zeros(
            mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64
        )
        + base
    )

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, : remainder[i]] += 1

    return num_transfer_tokens


@torch.no_grad()
def generate(
    model,
    prompt,
    steps=128,
    gen_length=128,
    block_length=128,
    temperature=0.0,
    cfg_scale=0.0,
    remasking="low_confidence",
    mask_id=126336,
    watermark: Optional[Watermark] = None,
):
    """
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    """
    if watermark is not None:
        watermark.init()

    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(
        model.device
    )
    x[:, : prompt.shape[1]] = prompt.clone()

    prompt_index = x != mask_id

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    for num_block in range(num_blocks):
        block_mask_index = (
            x[
                :,
                prompt.shape[1]
                + num_block * block_length : prompt.shape[1]
                + (num_block + 1) * block_length :,
            ]
            == mask_id
        )
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        for i in range(steps):
            mask_index = x == mask_id
            if cfg_scale > 0.0:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x).logits

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            if watermark is not None and watermark.watermark_config.prebias:
                logits_todo = watermark.apply_all(
                    x,
                    mask_id,
                    logits_with_noise,
                    prompt.shape[1] + num_block * block_length,
                    prompt.shape[1] + (num_block + 1) * block_length,
                )
            else:
                logits_todo = logits_with_noise

            x0 = torch.argmax(logits_todo, dim=-1)  # b, l

            if remasking == "low_confidence":
                p = F.softmax(logits_todo, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                )  # b, l
            elif remasking == "random":
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length :] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])

                if watermark is not None:
                    select_indices = select_index.tolist()
                    select_indices.sort()
                    for index in select_indices:
                        if x[j, index - 1] == mask_id:
                            prev_logits = logits_todo[j, index - 1]
                            prev_token = None
                        else:
                            prev_logits = None
                            prev_token = x[j, index - 1]
                        x0[j, index] = watermark.apply_once(
                            logits_with_noise[j, index],
                            index - prompt.shape[1],
                            prev_logits,
                            prev_token,
                            x0[j, index + 1] if index + 1 < x0.shape[1] else None,
                        )

                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]

    return x


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="LLaDA text generation with optional watermarking"
    )

    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        default="GSAI-ML/LLaDA-8B-Instruct",
        help="Model name or path",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")

    # Dataset arguments
    parser.add_argument(
        "--dataset", type=str, default="sentence-transformers/eli5", help="Dataset path"
    )
    parser.add_argument(
        "--num_samples", type=int, default=10, help="Number of samples to process"
    )

    # Generation arguments
    parser.add_argument(
        "--steps", type=int, default=128, help="Number of generation steps"
    )
    parser.add_argument(
        "--gen_length", type=int, default=128, help="Length of generated text"
    )
    parser.add_argument(
        "--block_length", type=int, default=32, help="Block length for generation"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0, help="Temperature for sampling"
    )
    parser.add_argument("--cfg_scale", type=float, default=0.0, help="CFG scale")
    parser.add_argument(
        "--remasking",
        type=str,
        default="low_confidence",
        choices=["low_confidence", "random"],
        help="Remasking strategy",
    )

    # Watermark arguments
    parser.add_argument(
        "--strategy",
        type=str,
        default=None,
        choices=["normal", "predict", "reverse", "legacy-ahead", "legacy-both"],
        help="Watermark strategy (if provided, watermarking is enabled)",
    )
    parser.add_argument(
        "--bitmap", type=str, default="bitmapt.bin", help="Path to bitmap file"
    )
    parser.add_argument(
        "--vocab_size", type=int, default=126464, help="Vocabulary size"
    )
    parser.add_argument("--ratio", type=float, default=0.5, help="Watermark ratio")
    parser.add_argument("--delta", type=float, default=2.0, help="Watermark delta")
    parser.add_argument("--key", type=int, default=42, help="Watermark key")
    parser.add_argument("--prebias", action="store_true", help="Enable prebias")

    # Output arguments
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Output directory for results"
    )

    args = parser.parse_args()
    
    watermark_config = WatermarkConfig(
        vocab_size=args.vocab_size,
        ratio=args.ratio,
        delta=args.delta,
        key=args.key,
        prebias=args.prebias,
        strategy=args.strategy,
    )
    
    args.watermark_config = watermark_config
    return args


def run_generation(
    model_path: str,
    device: str,
    dataset_name: str,
    num_samples: int,
    steps: int,
    gen_length: int,
    block_length: int,
    temperature: float,
    cfg_scale: float,
    remasking: str,
    watermark_config: WatermarkConfig,
    bitmap_path: str,
    output_dir: Optional[str],
) -> list[dict[str, Any]]:
    """Run the generation process with optional watermarking."""
    # Load dataset
    dataset = load_dataset(dataset_name, split="train")

    # Set up watermarking if config is provided
    if watermark_config.strategy is not None:
        bitmap = PersistentBitmap(watermark_config.vocab_size, bitmap_path)
        watermark = Watermark(watermark_config, bitmap)
        enable_watermark = True

    # Load model
    model = (
        AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        .to(device)
        .eval()
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    perplexity_eval = PerplexityEval()

    results = []

    for i in tqdm(range(num_samples), desc="Processing dataset"):
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
                steps=steps,
                gen_length=gen_length,
                block_length=block_length,
                temperature=temperature,
                cfg_scale=cfg_scale,
                remasking=remasking,
                watermark_config=watermark.watermark_config if watermark else None,
            )
        else:
            out = generate(
                model,
                input_ids,
                steps=steps,
                gen_length=gen_length,
                block_length=block_length,
                temperature=temperature,
                cfg_scale=cfg_scale,
                remasking=remasking,
                watermark=watermark,
            )

        output = tokenizer.batch_decode(
            out[:, input_ids.shape[1] :], skip_special_tokens=True
        )[0]
        all = tokenizer.batch_decode(out, skip_special_tokens=True)[0]
        
        detector_config = watermark_config
        if (
            detector_config.strategy == "legacy-ahead"
            or detector_config.strategy == "legacy-both"
        ):
            detect_rate, z_score = detect(
                out[0][input_ids.shape[1] - 1 :], detector_config
            )
        else:
            detect_rate, z_score = Detector(detector_config).detect(
                out[0], input_ids.shape[1]
            )
        ppl = perplexity_eval.evaluate(all)
        results.append(
            {
                "prompt": prompt,
                "ground_truth": gt,
                "output": output,
                "detect_rate": detect_rate,
                "z_score": z_score,
                "ppl": ppl,
                "generation_metadata": {
                    "model": model_path,
                    "dataset": dataset_name,
                    "steps": steps,
                    "gen_length": gen_length,
                    "block_length": block_length,
                    "temperature": temperature,
                    "cfg_scale": cfg_scale,
                    "remasking": remasking,
                    "device": device,
                },
                "watermark_metadata": {
                    "strategy": watermark_config.strategy,
                    "vocab_size": watermark_config.vocab_size,
                    "ratio": watermark_config.ratio,
                    "delta": watermark_config.delta,
                    "key": watermark_config.key,
                    "prebias": watermark_config.prebias,
                } if enable_watermark else None,
            }
        )

    # Generate result filename based on arguments
    dataset_short_name = dataset_name.split("/")[-1]
    model_short_name = model_path.split("/")[-1]

    if output_dir is not None:
        if enable_watermark:
            result_name = f"results_{dataset_short_name}_{model_short_name}_wm_r{watermark_config.ratio}_d{watermark_config.delta}_k{watermark_config.key}"
            if watermark_config.prebias:
                result_name += "_prebias"
            result_name += f"_{watermark_config.strategy}_r{remasking}"
        else:
            result_name = f"results_{dataset_short_name}_{model_short_name}_no_wm"

        result_name += f"_s{steps}_l{gen_length}_b{block_length}_t{temperature}_n{num_samples}.json"

        result_path = os.path.join(output_dir, result_name)

        with open(result_path, "w") as f:
            json.dump(results, f, indent=4)

        print(f"Results saved to: {result_path}")
    
    return results


def main():
    """Main entry point for the script."""
    args = parse_args()
    
    run_generation(
        model_path=args.model,
        device=args.device,
        dataset_name=args.dataset,
        num_samples=args.num_samples,
        steps=args.steps,
        gen_length=args.gen_length,
        block_length=args.block_length,
        temperature=args.temperature,
        cfg_scale=args.cfg_scale,
        remasking=args.remasking,
        watermark_config=args.watermark_config,
        bitmap_path=args.bitmap,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
