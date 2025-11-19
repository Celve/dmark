from typing import Optional

import torch

from dmark.watermark.config import WatermarkConfig
from dmark.watermark.persistent_bitmap import PersistentBitmap
from dmark.watermark.watermark.base import BaseWatermark


class BidirectionalWatermark(BaseWatermark):
    def __init__(self, watermark_config: WatermarkConfig, bitmap: PersistentBitmap, mask_id: int):
        super().__init__(watermark_config, mask_id)
        self.bitmap = bitmap

    def _token_to_index(self, token: Optional[torch.Tensor | int]) -> Optional[int]:
        if token is None:
            return None
        if isinstance(token, torch.Tensor):
            return int(token.detach().item())
        return int(token)

    def _row_bias(self, token_index: int, template: torch.Tensor) -> torch.Tensor:
        row = self.bitmap.get_row(token_index)
        non_blocking = row.device.type == "cpu" and template.device.type == "cuda"
        row = row.to(device=template.device, dtype=template.dtype, non_blocking=non_blocking)
        return row * self.watermark_config.delta

    def _col_bias(self, token_index: int, template: torch.Tensor) -> torch.Tensor:
        col = self.bitmap.get_col(token_index)
        non_blocking = col.device.type == "cpu" and template.device.type == "cuda"
        col = col.to(device=template.device, dtype=template.dtype, non_blocking=non_blocking)
        return col * self.watermark_config.delta

    def apply_single(
        self,
        curr_logits: torch.Tensor,
        prev_logits: Optional[torch.Tensor],
        prev_token: Optional[torch.Tensor],
        next_logits: Optional[torch.Tensor],
        next_token: Optional[torch.Tensor],
        index: int,
    ) -> torch.Tensor:
        biased_logits = curr_logits

        prev_idx = self._token_to_index(prev_token)
        if prev_idx is not None:
            biased_logits = biased_logits + self._row_bias(prev_idx, curr_logits)

        next_idx = self._token_to_index(next_token)
        if next_idx is not None:
            biased_logits = biased_logits + self._col_bias(next_idx, curr_logits)

        return torch.argmax(biased_logits)

    def apply_range(
        self,
        x: torch.Tensor,
        logits: torch.Tensor,
        start_index: int,
        end_index: int,
    ) -> torch.Tensor:
        biased_logits = logits.clone()
        device = logits.device

        prev_slice = x[:, start_index - 1 : end_index - 1]
        if end_index + 1 <= x.shape[1]:
            next_slice = x[:, start_index + 1 : end_index + 1]
        else:
            pad_width = end_index + 1 - x.shape[1]
            padded = torch.full(
                (x.shape[0], pad_width),
                self.mask_id,
                dtype=x.dtype,
                device=x.device,
            )
            next_slice = torch.cat(
                [
                    x[:, start_index + 1 : x.shape[1]],
                    padded,
                ],
                dim=1,
            )

        prev_valid = prev_slice != self.mask_id
        next_valid = next_slice != self.mask_id

        prev_safe = prev_slice.clone()
        prev_safe[~prev_valid] = 0

        next_safe = next_slice.clone()
        next_safe[~next_valid] = 0

        vocab_size = self.bitmap.vocab_size

        row_rows = self.bitmap.get_rows(prev_safe.reshape(-1)).reshape(
            *prev_safe.shape,
            vocab_size,
        )
        col_rows = self.bitmap.get_cols(next_safe.reshape(-1)).permute(1, 0).reshape(
            *next_safe.shape,
            vocab_size,
        )

        if row_rows.device.type != device.type:
            row_rows = row_rows.to(device=device, dtype=logits.dtype)
        if col_rows.device.type != device.type:
            col_rows = col_rows.to(device=device, dtype=logits.dtype)

        row_bias = row_rows * prev_valid.unsqueeze(-1).to(device=device, dtype=logits.dtype)
        col_bias = col_rows * next_valid.unsqueeze(-1).to(device=device, dtype=logits.dtype)

        bias = (row_bias + col_bias) * self.watermark_config.delta
        biased_logits[:, start_index:end_index] += bias
        return biased_logits
