import json
from typing import Optional

import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset

from watermark.config import WatermarkConfig
from watermark.detect import Detector
from watermark.persistent_bitmap import PersistentBitmap
from watermark.watermark import Watermark


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
        watermark.init(gen_length)

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
                        )

                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]

    return x


def main():
    device = "cuda"
    dataset_path = "sentence-transformers/eli5"
    enable_watermark = False

    dataset = load_dataset(dataset_path, split='train')

    watermark_config = WatermarkConfig(vocab_size=126464, ratio=0.5, delta=3.0, key=42, prebias=True, enable_reverse=True)
    bitmap = PersistentBitmap(watermark_config.vocab_size, "../bitmapt.bin")
    watermark = Watermark(watermark_config, bitmap)

    model = (
        AutoModel.from_pretrained(
            "GSAI-ML/LLaDA-8B-Instruct",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        .to(device)
        .eval()
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True
    )

    results = []

    for i in tqdm(range(20), desc="Processing dataset"):
        prompt = dataset[i]['question']
        gt = dataset[i]['answer']
        m = [
            {"role": "user", "content": prompt},
        ]
        prompt = tokenizer.apply_chat_template(
            m, add_generation_prompt=True, tokenize=False
        )

        input_ids = tokenizer(prompt)["input_ids"]
        input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

        out = generate(
            model,
            input_ids,
            steps=128,
            gen_length=256,
            block_length=32,
            temperature=0.0,
            cfg_scale=0.0,
            remasking="low_confidence",
            watermark=watermark if enable_watermark else None,
        )

        output = tokenizer.batch_decode(out[:, input_ids.shape[1] :], skip_special_tokens=True)[0]
        z_score = Detector(watermark_config).detect(out[0], input_ids.shape[1])
        results.append({
            "prompt": prompt,
            "ground_truth": gt,
            "output": output,
            "z_score": z_score,
        })
    
    result_name = f"results_{dataset_path.split('/')[-1]}_ratio{watermark_config.ratio}_delta{watermark_config.delta}_key{watermark_config.key}_prebias{watermark_config.prebias}_enable_reverse{watermark_config.enable_reverse}.json"

    with open(result_name, "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()
