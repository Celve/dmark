from typing import Callable, Optional

import torch

from watermark.config import WatermarkConfig
from watermark.persistent_bitmap import PersistentBitmap


def watermark(
    next_token: Optional[torch.Tensor],
    supposed_token: Optional[torch.Tensor],
    logits: torch.Tensor,
    prev_token: torch.Tensor,
    watermark_config: WatermarkConfig,
    bitmap: PersistentBitmap,
) -> torch.Tensor:
    """
    Apply watermarking bias to logits for token generation.

    Args:
        next_token: The next token to generate (if available)
        supposed_token: The expected token according to the model
        logits: Raw logits from the language model
        prev_token: The previous token in the sequence
        watermark_config: Configuration containing watermark parameters
        bitmap: Persistent bitmap storing precomputed green lists

    Returns:
        The selected token index after applying watermark biases
    """
    prev_bias = watermark_config.gen_bias(prev_token)
    next_bias = torch.zeros_like(logits)
    if next_token is not None:
        sampled = torch.argmax(
            logits
        )  # TODO: we have to support different sampling methods
        if sampled != supposed_token:
            ndarray = bitmap.get_row(next_token.item())
            next_bias = (
                torch.from_numpy(ndarray).to(torch.bool) * watermark_config.delta
            )

    biased_logits = logits + prev_bias + next_bias
    return torch.argmax(biased_logits)  # TODO: sampling also matters here
