from typing import Callable, Optional

import torch

from watermark.config import WatermarkConfig
from watermark.persistent_bitmap import PersistentBitmap


class Watermark:
    def __init__(self, watermark_config: WatermarkConfig, bitmap: PersistentBitmap):
        self.watermark_config = watermark_config
        self.bitmap = bitmap
        self.gen_len = None

    def init(self, gen_len: int):
        self.gen_len = gen_len
        self.assumed = torch.ones(gen_len, dtype=torch.int32) * -1
        self.double = 0
        self.green = 0
    
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
            ).to("cuda")
            self.double += 1

        biased_logits = curr_logits + prev_bias + next_bias
        result = torch.argmax(biased_logits)  # TODO: sampling also matters here
        green_list = self.watermark_config.gen_green_list(prev_token).bool()
        if green_list[result.item()].item():
            self.green += 1
        else: 
            print("curr_logits topk(10):")
            values, indices = curr_logits.topk(10)
            for i in range(10):
                print(f"  index: {indices[i].item()}, value: {values[i].item()}")
            print("biased_logits topk(10):")
            b_values, b_indices = biased_logits.topk(10)
            for i in range(10):
                print(f"  index: {b_indices[i].item()}, value: {b_values[i].item()}")
        return result
