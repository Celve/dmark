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
from dmark.gen.dream import DreamGenerationMixin
from dmark.gen.utils import (
    DreamExprConfig,
    DreamGenConfig,
    build_watermark,
    generate_dream_result_filename,
    parse_dream_args,
)
from dmark.watermark.config import WatermarkConfig
from dmark.watermark.watermark.base import BaseWatermark

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_dataset(gen_config: DreamGenConfig, tokenizer) -> Any:
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


def _prepare_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.padding_side != "left":
        tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = (
            tokenizer.eos_token or tokenizer.unk_token or tokenizer.bos_token
        )
    if tokenizer.pad_token is None:
        raise ValueError("Tokenizer must provide a pad token for batch inference.")
    if tokenizer.pad_token_id == 126336:
        raise ValueError("Tokenizer pad token matches mask id 126336.")
    return tokenizer


def _tensor_to_1d_cpu(tensor: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(tensor):
        tensor = torch.tensor(tensor, dtype=torch.long)
    else:
        tensor = tensor.detach()
    if tensor.dim() == 2 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    if tensor.dim() != 1:
        raise ValueError(
            f"Expected dataset input_ids to be 1D or (1, L), got shape {tuple(tensor.shape)}"
        )
    return tensor.to(torch.long).cpu().contiguous()


def _resolve_mask_token_id(model) -> int:
    mask_id = getattr(getattr(model, "generation_config", None), "mask_token_id", None)
    if mask_id is None:
        mask_id = getattr(getattr(model, "config", None), "mask_token_id", None)
    if mask_id is None:
        raise ValueError("Model does not define mask_token_id; cannot enable watermarking.")
    return int(mask_id)



def run_generation(
    gen_config: DreamGenConfig,
    watermark_config: WatermarkConfig,
    expr_config: DreamExprConfig,
) -> list[dict[str, Any]]:
    tokenizer = _prepare_tokenizer(gen_config.model)
    dataset = _build_dataset(gen_config, tokenizer)
    stop_token_ids: set[int] = set()
    if tokenizer.eos_token_id is not None:
        stop_token_ids.add(tokenizer.eos_token_id)

    model = (
        AutoModel.from_pretrained(
            gen_config.model,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        .to(DEVICE)
        .eval()
    )

    watermark: BaseWatermark | None = None
    if watermark_config.strategy is not None:
        mask_id = _resolve_mask_token_id(model)
        watermark = build_watermark(
            watermark_config,
            bitmap_device=expr_config.bitmap_device,
            mask_id=mask_id,
        )

    if watermark is not None:
        model.diffusion_generate = DreamGenerationMixin.diffusion_generate.__get__(
            model, model.__class__
        )
        model._sample = DreamGenerationMixin._sample.__get__(model, model.__class__)
        model._prepare_generation_config = (
            DreamGenerationMixin._prepare_generation_config.__get__(
                model, model.__class__
            )
        )
        model._prepare_special_tokens = (
            DreamGenerationMixin._prepare_special_tokens.__get__(
                model, model.__class__
            )
        )
        model._prepare_generated_length = (
            DreamGenerationMixin._prepare_generated_length.__get__(
                model, model.__class__
            )
        )
        model._validate_generated_length = (
            DreamGenerationMixin._validate_generated_length.__get__(
                model, model.__class__
            )
        )
        model._expand_inputs_for_generation = (
            DreamGenerationMixin._expand_inputs_for_generation.__get__(
                model, model.__class__
            )
        )

    batch_size = max(1, expr_config.batch_size)
    results: list[dict[str, Any]] = []
    dataset_index = 0
    dataset_exhausted = False
    skipped_min_tokens = 0
    skipped_repetition = 0
    pbar = tqdm.tqdm(total=expr_config.num_samples, desc="Collecting valid samples")

    def fetch_next_sample():
        nonlocal dataset_index
        try:
            raw = dataset.sample(dataset_index)
        except IndexError:
            return None
        dataset_index += 1
        prompt = raw["prompt"]
        ground_truth = raw.get("ground_truth")
        input_ids = _tensor_to_1d_cpu(raw["input_ids"])
        return {
            "prompt": prompt,
            "ground_truth": ground_truth,
            "input_ids": input_ids,
        }

    def collate_batch(batch_samples):
        max_len = max(sample["input_ids"].shape[0] for sample in batch_samples)
        batch_input_ids = torch.full(
            (len(batch_samples), max_len),
            tokenizer.pad_token_id,
            dtype=torch.long,
            device=DEVICE,
        )
        attention_mask = torch.zeros(
            (len(batch_samples), max_len), dtype=torch.long, device=DEVICE
        )
        for idx, sample in enumerate(batch_samples):
            seq = sample["input_ids"].to(DEVICE)
            seq_len = seq.shape[0]
            batch_input_ids[idx, max_len - seq_len :] = seq
            attention_mask[idx, max_len - seq_len :] = 1
        return batch_input_ids, attention_mask

    def trim_special_tokens(output_ids: torch.Tensor) -> torch.Tensor:
        trimmed_length = output_ids.shape[0]
        for i, curr_token in enumerate(output_ids):
            if curr_token.item() in stop_token_ids:
                trimmed_length = i
                break
        return output_ids[:trimmed_length]

    def process_sample(sample, output_ids_tensor):
        nonlocal skipped_min_tokens, skipped_repetition
        output_ids = trim_special_tokens(output_ids_tensor.detach().cpu())
        num_output_tokens = int(output_ids.shape[0])

        if (
            expr_config.minimum_output_token
            and num_output_tokens < expr_config.minimum_output_token
        ):
            skipped_min_tokens += 1
            pbar.set_postfix(
                {
                    "skipped": f"min_tokens: {num_output_tokens} < {expr_config.minimum_output_token}"
                }
            )
            return

        if num_output_tokens > 0:
            token_counts = Counter(output_ids.tolist())
            max_count = max(token_counts.values())
            max_ratio = max_count / num_output_tokens
            if max_ratio > expr_config.repeat_ratio:
                skipped_repetition += 1
                most_repeated_token = max(token_counts, key=token_counts.get)
                pbar.set_postfix(
                    {
                        "skipped": (
                            f"repetition: token {most_repeated_token} appears "
                            f"{max_count}/{num_output_tokens} ({max_ratio:.1%})"
                        )
                    }
                )
                return

        output_list = output_ids.tolist()
        output_text = (
            tokenizer.decode(output_list, skip_special_tokens=True) if output_list else ""
        )
        results.append(
            {
                "data": {
                    "prompt": sample["prompt"],
                    "ground_truth": sample["ground_truth"],
                    "output": output_text,
                    "output_ids": output_list,
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

    generation_kwargs = {
        "max_new_tokens": gen_config.gen_length,
        "output_history": gen_config.output_history,
        "return_dict_in_generate": gen_config.return_dict_in_generate,
        "steps": gen_config.steps,
        "temperature": gen_config.temperature,
        "top_p": gen_config.top_p,
        "alg": gen_config.alg,
        "alg_temp": gen_config.alg_temp,
        "eps": gen_config.eps,
    }
    if gen_config.top_k is not None:
        generation_kwargs["top_k"] = gen_config.top_k
    if watermark is not None:
        generation_kwargs["watermark"] = watermark
    if expr_config.ignore_eos:
        generation_kwargs["logits_eos_inf"] = True

    try:
        while len(results) < expr_config.num_samples:
            remaining = expr_config.num_samples - len(results)
            target_batch = min(batch_size, remaining)
            batch_samples = []
            while len(batch_samples) < target_batch:
                sample = fetch_next_sample()
                if sample is None:
                    dataset_exhausted = True
                    break
                batch_samples.append(sample)

            if not batch_samples:
                break

            batch_input_ids, batch_attention_mask = collate_batch(batch_samples)
            output = model.diffusion_generate(
                batch_input_ids,
                attention_mask=batch_attention_mask,
                **generation_kwargs,
            )
            generated_segment = (
                output.sequences[:, batch_input_ids.shape[1] :]
                if gen_config.return_dict_in_generate
                else output[:, batch_input_ids.shape[1] :]
            )
            for idx, sample in enumerate(batch_samples):
                process_sample(sample, generated_segment[idx])

            if dataset_exhausted:
                break

    except KeyboardInterrupt:
        print(f"\n\nInterrupted! Collected {len(results)} samples so far.")
        pbar.close()
        return results

    pbar.close()

    if len(results) < expr_config.num_samples:
        print(
            f"Warning: Only collected {len(results)} valid samples out of {expr_config.num_samples} requested."
        )
        if dataset_exhausted:
            print(f"Dataset exhausted at index {dataset_index}.")

    print(f"Skipped due to repetition: {skipped_repetition}")
    if expr_config.minimum_output_token is not None:
        print(f"Skipped due to min tokens: {skipped_min_tokens}")

    return results


if __name__ == "__main__":
    gen_config, watermark_config, expr_config = parse_dream_args()
    results = run_generation(gen_config, watermark_config, expr_config)

    if results:
        if expr_config.output_dir is not None:
            os.makedirs(expr_config.output_dir, exist_ok=True)
            output_filename = generate_dream_result_filename(
                gen_config, watermark_config, expr_config
            )
            output_filename = os.path.join(expr_config.output_dir, output_filename)
            with open(output_filename, "w") as f:
                json.dump(results, f, indent=4)
            print(f"Results saved to: {output_filename}")
            print(f"Total samples collected: {len(results)}")
        else:
            print(results)
    else:
        print("No results were collected.")
