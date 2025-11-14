import json
import os
from collections import Counter
from typing import Any

import torch
import tqdm
from transformers import AutoModel, AutoTokenizer

from dmark.dataset.c4 import C4Dataset
from dmark.dataset.eli5 import ELI5Dataset
from dmark.dataset.gsm8k import GSM8KDataset
from dmark.gen.utils import ExprConfig, GenConfig, generate_result_filename, parse_args
from dmark.llada.gen_llada import generate
from dmark.watermark.config import WatermarkConfig
from dmark.watermark.persistent_bitmap import PersistentBitmap
from dmark.watermark.watermark import Watermark

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_dataset(gen_config: GenConfig, tokenizer) -> Any:
    dataset_name = gen_config.dataset.lower()
    if dataset_name == "allenai/c4":
        return C4Dataset(
            tokenizer,
            dataset_name="allenai/c4",
            config_name="en",
            split="train",
            device=DEVICE,
            prompt_tokens=30,
            continuation_tokens=gen_config.gen_length,
            streaming=True,
        )
    if dataset_name in {"sentence-transformers/eli5", "eli5"}:
        return ELI5Dataset(
            tokenizer,
            dataset_name=gen_config.dataset,
            split="train",
            device=DEVICE,
        )
    if dataset_name in {"openai/gsm8k", "gsm8k"}:
        return GSM8KDataset(
            tokenizer,
            dataset_name=gen_config.dataset,
            split="train",
            device=DEVICE,
        )
    raise ValueError(
        f"Unsupported dataset '{gen_config.dataset}'. "
        "Add a matching Dataset implementation under dmark/dataset."
    )


def run_generation(gen_config: GenConfig, watermark_config: WatermarkConfig, expr_config: ExprConfig) -> list[dict[str, Any]]:
    tokenizer = AutoTokenizer.from_pretrained(gen_config.model, trust_remote_code=True)
    dataset = _build_dataset(gen_config, tokenizer)

    watermark: Watermark | None = None
    if watermark_config.strategy is not None:
        bitmap = PersistentBitmap(
            watermark_config.vocab_size,
            watermark_config.bitmap_path,
            device=expr_config.bitmap_device,
        )
        watermark = Watermark(watermark_config, bitmap)

    model = (
        AutoModel.from_pretrained(
            gen_config.model,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        .to(DEVICE)
        .eval()
    )

    results: list[dict[str, Any]] = []
    dataset_index = 0
    pbar = tqdm.tqdm(total=expr_config.num_samples, desc="Collecting valid samples")

    try:
        while len(results) < expr_config.num_samples:
            try:
                sample = dataset.sample(dataset_index)
            except IndexError:
                print(f"Dataset exhausted at index {dataset_index}.")
                break

            dataset_index += 1
            prompt = sample["prompt"]
            ground_truth = sample.get("ground_truth")
            input_ids = sample["input_ids"].to(DEVICE)

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

            output_ids = out[:, input_ids.shape[1]:][0].detach().cpu().tolist()

            trimmed_length = len(output_ids)
            for idx, curr_token in enumerate(output_ids):
                if curr_token == 126081 or curr_token == 126348:
                    trimmed_length = idx
                    break
            output_ids = output_ids[:trimmed_length]

            num_output_tokens = len(output_ids)
            if (
                expr_config.minimum_output_token is not None
                and num_output_tokens < expr_config.minimum_output_token
            ):
                pbar.set_postfix(
                    {
                        "skipped": f"min_tokens: {num_output_tokens} < {expr_config.minimum_output_token}"
                    }
                )
                continue

            if num_output_tokens > 0:
                token_counts = Counter(output_ids)
                max_count = max(token_counts.values())
                max_ratio = max_count / num_output_tokens
                if max_ratio > expr_config.repeat_ratio:
                    most_repeated_token = max(token_counts, key=token_counts.get)
                    pbar.set_postfix(
                        {
                            "skipped": f"repetition: token {most_repeated_token} appears {max_count}/{num_output_tokens} ({max_ratio:.1%})"
                        }
                    )
                    continue

            output_text = (
                tokenizer.decode(output_ids, skip_special_tokens=True)
                if output_ids
                else ""
            )

            results.append(
                {
                    "data": {
                        "prompt": prompt,
                        "ground_truth": ground_truth,
                        "output": output_text,
                        "output_ids": output_ids,
                        "num_output_tokens": num_output_tokens,
                    },
                    "generation_metadata": gen_config.model_dump(),
                    "watermark_metadata": watermark_config.model_dump()
                    if watermark_config.strategy is not None
                    else None,
                    "expr_metadata": expr_config.model_dump(),
                }
            )
            pbar.update(1)

    except KeyboardInterrupt:
        print(f"\n\nInterrupted! Collected {len(results)} samples so far.")
    finally:
        pbar.close()

    if len(results) < expr_config.num_samples:
        print(
            f"Warning: Only collected {len(results)} valid samples out of {expr_config.num_samples} requested."
        )

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
