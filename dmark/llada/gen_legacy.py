import json
import logging
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from omegaconf import DictConfig, ListConfig, OmegaConf
from transformers import AutoModel, AutoTokenizer


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
    In the bidirectional process, the interval [0, 1] is uniformly discretized into steps intervals.
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
def generate_LLaDA(
    model,
    prompt,
    steps=128,
    gen_length=128,
    block_length=128,
    temperature=0.0,
    cfg_scale=0.0,
    remasking="low_confidence",
    mask_id=126336,
    watermark_config=None,
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
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(
        model.device
    )
    x[:, : prompt.shape[1]] = prompt.clone()

    prompt_index = x != mask_id

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    total_watermark_time = 0

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
            # logits_with_noise.shape: [batch_size, sequence_length, vocab_size]
            assert (
                logits_with_noise.shape[-1] == watermark_config.vocab_size
            ), "Logits shape mismatch with model vocab size."

            start_time = time.time()
            if watermark_config is not None:
                # Apply watermarking logic here if needed
                assert (
                    watermark_config.strategy == "legacy-ahead"
                    or watermark_config.strategy == "legacy-both"
                ), "Watermark method must be 'legacy-ahead' or 'legacy-both'."
                for j in range(logits_with_noise.shape[0]):
                    for k in range(logits_with_noise.shape[1]):
                        if watermark_config.strategy == "legacy-ahead":
                            if (
                                mask_index[j, k] == True
                                and k != 0
                                and mask_index[j, k - 1] == False
                            ):
                                rng = random.Random(
                                    x[j, k - 1].item() * watermark_config.key
                                )
                                watermark_number = int(
                                    watermark_config.ratio * watermark_config.vocab_size
                                )
                                # 从[0, watermark_config.vocab_size)中随机采样
                                watermark_samples = rng.sample(
                                    range(watermark_config.vocab_size), watermark_number
                                )
                                # Apply watermarking logic for 'ahead' method
                                logits_with_noise[
                                    j, k, watermark_samples
                                ] += watermark_config.delta
                        elif watermark_config.strategy == "legacy-both":
                            if mask_index[j, k] == True:
                                flag_ahead = k != 0 and mask_index[j, k - 1] == False
                                flag_behind = (
                                    k != logits_with_noise.shape[1] - 1
                                    and mask_index[j, k + 1] == False
                                )
                                if flag_ahead or flag_behind:
                                    rng = random.Random(
                                        x[j, k - 1 if flag_ahead else k + 1].item()
                                        * watermark_config.key
                                    )
                                    watermark_number = int(
                                        watermark_config.ratio
                                        * watermark_config.vocab_size
                                    )
                                    # 从[0, watermark_config.vocab_size)中随机采样
                                    watermark_samples = rng.sample(
                                        range(watermark_config.vocab_size),
                                        watermark_number,
                                    )
                                    # Apply watermarking logic for 'ahead' method
                                    logits_with_noise[
                                        j, k, watermark_samples
                                    ] += watermark_config.delta
            total_watermark_time += time.time() - start_time

            x0 = torch.argmax(logits_with_noise, dim=-1)  # b, l

            if remasking == "low_confidence":
                p = F.softmax(logits, dim=-1)
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
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]

    print(f"Total watermarking time: {total_watermark_time:.2f} seconds")
    return x
