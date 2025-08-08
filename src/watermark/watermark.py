from typing import Callable, Optional

import torch

from watermark.config import WatermarkConfig
from watermark.persistent_bitmap import PersistentBitmap


class Watermark:
    def __init__(self, watermark_config: WatermarkConfig, bitmap: PersistentBitmap):
        self.watermark_config = watermark_config
        self.bitmap = bitmap
        self.gen_len = None

    def init(self, prompt: torch.Tensor, gen_len: int):
        self.gen_len = gen_len
        self.assumed = torch.ones(gen_len, dtype=torch.int32) * -1

    def apply(
        self,
        curr_logits: torch.Tensor,
        pos: int,
        prev_logits: Optional[torch.Tensor],
        prev_token: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if prev_token is not None:
            prev_bias = self.watermark_config.gen_bias(prev_token)
        else:
            prev_token = prev_logits.argmax(dim=-1)
            prev_bias = self.watermark_config.gen_bias(prev_token)
            self.assumed[pos - 1] = prev_token

        next_bias = torch.zeros_like(curr_logits)
        sampled = torch.argmax(
            curr_logits
        )  # TODO: we have to support different sampling methods
        if self.assumed[pos] != -1 and sampled != self.assumed[pos]:
            ndarray = self.bitmap.get_row(sampled.item())
            next_bias = (
                torch.from_numpy(ndarray).to(torch.bool) * self.watermark_config.delta
            )

        biased_logits = curr_logits + prev_bias + next_bias
        return torch.argmax(biased_logits)  # TODO: sampling also matters here
