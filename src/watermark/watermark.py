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

    def apply_once(
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
            row_tensor = self.bitmap.get_col(sampled.item())
            next_bias = row_tensor.float() * self.watermark_config.delta
            next_bias = next_bias.to(curr_logits.device)
            self.double += 1

        biased_logits = curr_logits + prev_bias
        if self.watermark_config.enable_reverse:
            biased_logits = biased_logits + next_bias
        result = torch.argmax(biased_logits)  # TODO: sampling also matters here
        green_list = self.watermark_config.gen_green_list(prev_token).bool()
        if green_list[result.item()].item():
            self.green += 1
        return result

    def apply_all(
        self, logits: torch.Tensor, start_index: int, end_index: int
    ) -> torch.Tensor:
        biased_logits = logits.clone()
        predicted = logits[:, start_index:end_index].argmax(dim=-1)
        for i in range(logits.shape[0]):
            bias = (
                self.bitmap.get_rows(
                    predicted[i]
                ).float()
                * self.watermark_config.delta
            )
            biased_logits[i, start_index:end_index] += bias
        return biased_logits
