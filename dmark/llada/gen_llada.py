import argparse
import json
import os
from typing import Any, Optional

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
        prompt: A tensor of shape (B, L) where B is batch size.
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence', 'random', 'right_to_left', or 'left_to_right'.
        mask_id: The toke id of [MASK] is 126336.
    """
    if watermark is not None:
        watermark.init()

    # Support batch inference
    batch_size = prompt.size(0)
    prompt_len = prompt.size(1)
    total_len = prompt_len + gen_length
    
    x = torch.full((batch_size, total_len), mask_id, dtype=torch.long, device=model.device)
    x[:, :prompt_len] = prompt

    prompt_index = x != mask_id

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    for num_block in range(num_blocks):
        block_mask_index = (
            x[
                :,
                prompt_len
                + num_block * block_length : prompt_len
                + (num_block + 1) * block_length,
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
                    prompt_len + num_block * block_length,
                    prompt_len + (num_block + 1) * block_length,
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
            elif remasking in ["right_to_left", "left_to_right"]:
                # Vectorized position-based confidence scores
                x0_p = torch.zeros((batch_size, total_len), device=x0.device)
                start_idx = prompt_len + num_block * block_length
                end_idx = prompt_len + (num_block + 1) * block_length
                
                # Create position weights for the block
                pos = torch.arange(block_length, device=x0.device)
                if remasking == "right_to_left":
                    # Higher confidence for rightmost tokens
                    weights = block_length - pos
                else:  # left_to_right
                    # Higher confidence for leftmost tokens
                    weights = pos + 1
                
                # Broadcast weights to all batch samples
                x0_p[:, start_idx:end_idx] = weights.unsqueeze(0)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, prompt_len + (num_block + 1) * block_length :] = -float('inf')

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -float('inf'))

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for b in range(batch_size):
                k = int(num_transfer_tokens[b, i].item())
                if k == 0:
                    continue  # Skip if no tokens to transfer
                    
                _, select_index = torch.topk(confidence[b], k=k)

                if watermark is not None:
                    # Sort indices to process left-to-right for proper prev/next dependencies
                    select_indices = torch.sort(select_index).values
                    for idx in select_indices:
                        idx = idx.item()
                        if x[b, idx - 1] == mask_id:
                            prev_logits = logits_todo[b, idx - 1]
                            prev_token = None
                        else:
                            prev_logits = None
                            prev_token = x[b, idx - 1]
                        x0[b, idx] = watermark.apply_once(
                            logits_with_noise[b, idx],
                            idx - prompt_len,
                            prev_logits,
                            prev_token,
                            x0[b, idx + 1] if idx + 1 < x0.shape[1] else None,
                        )

                transfer_index[b, select_index] = True
            x[transfer_index] = x0[transfer_index]

    return x
