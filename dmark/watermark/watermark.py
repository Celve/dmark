from typing import Callable, Optional

import torch

from dmark.watermark.config import WatermarkConfig
from dmark.watermark.persistent_bitmap import PersistentBitmap


class Watermark:
    def __init__(self, watermark_config: WatermarkConfig, bitmap: PersistentBitmap):
        self.watermark_config = watermark_config
        self.bitmap = bitmap

    def init(self):
        self.double = 0
        self.green = 0
    
    def gen_green_list(self, prev_token: int) -> torch.Tensor:
        return self.bitmap.get_row(prev_token)
    
    def is_token_in_green_list(self, token: int, prev_token: int) -> bool:
        green_list = self.bitmap.get_row(prev_token)
        return bool(green_list[token].item())

    def apply_once(
        self,
        curr_logits: torch.Tensor,
        pos: int,
        prev_logits: Optional[torch.Tensor],
        prev_token: Optional[torch.Tensor],
        next_token: Optional[torch.Tensor],
    ) -> torch.Tensor:
        prev_bias = torch.zeros_like(curr_logits)
        next_bias = torch.zeros_like(curr_logits)

        if self.watermark_config.strategy == "normal":
            if prev_token is not None:
                prev_bias = (
                    self.bitmap.get_row(prev_token.item()).float()
                    * self.watermark_config.delta
                )
        else:
            assert (
                self.watermark_config.strategy == "reverse"
                or self.watermark_config.strategy == "predict"
            )

            if prev_token is None:
                prev_token = prev_logits.argmax(dim=-1)
            prev_bias = (
                self.bitmap.get_row(prev_token.item()).float()
                * self.watermark_config.delta
            )

            if next_token is not None and self.watermark_config.strategy == "reverse":
                col_tensor = self.bitmap.get_col(next_token.item())
                next_bias = col_tensor.float() * self.watermark_config.delta
                next_bias = next_bias.to(curr_logits.device)
                self.double += 1

        biased_logits = curr_logits + prev_bias + next_bias
        result = torch.argmax(biased_logits)  # TODO: sampling also matters here
        return result

    def apply_all(
        self,
        x: torch.Tensor,
        mask_id: int,
        logits: torch.Tensor,
        start_index: int,
        end_index: int,
    ) -> torch.Tensor:
        biased_logits = logits.clone()
        predicted = logits[:, start_index - 1 : end_index - 1].argmax(dim=-1)
        for i in range(logits.shape[0]):
            bias = (
                (x[i, start_index - 1 : end_index - 1] != mask_id).unsqueeze(1).float()
                * self.bitmap.get_rows(predicted[i]).float()
            ) * self.watermark_config.delta
            biased_logits[i, start_index:end_index] += bias
        return biased_logits
