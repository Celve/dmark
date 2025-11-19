from typing import Optional

import torch

from dmark.watermark.config import WatermarkConfig
from dmark.watermark.persistent_bitmap import PersistentBitmap
from dmark.watermark.watermark.base import BaseWatermark


class KGWWatermark(BaseWatermark):
    """Classic KGW watermarking equivalent to the legacy "normal" strategy."""

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

    def apply_single(
        self,
        curr_logits: torch.Tensor,
        prev_logits: Optional[torch.Tensor],
        prev_token: Optional[torch.Tensor],
        next_logits: Optional[torch.Tensor],
        next_token: Optional[torch.Tensor],
        index: int,
    ) -> torch.Tensor:
        prev_idx = self._token_to_index(prev_token)
        if prev_idx is None:
            biased_logits = curr_logits
        else:
            bias = self._row_bias(prev_idx, curr_logits)
            biased_logits = curr_logits + bias
        return torch.argmax(biased_logits)

    def apply_range(
        self,
        x: torch.Tensor,
        logits: torch.Tensor,
        start_index: int,
        end_index: int,
    ) -> torch.Tensor:
        biased_logits = logits.clone()
        prev_slice = x[:, start_index - 1 : end_index - 1]
        valid_mask = prev_slice != self.mask_id
        if not torch.any(valid_mask):
            return biased_logits

        safe_indices = prev_slice.clone()
        safe_indices[~valid_mask] = 0

        rows = self.bitmap.get_rows(safe_indices.reshape(-1)).reshape(
            *safe_indices.shape,
            self.bitmap.vocab_size,
        )

        non_blocking = rows.device.type == "cpu" and logits.device.type == "cuda"
        rows = rows.to(device=logits.device, dtype=logits.dtype, non_blocking=non_blocking)
        mask = valid_mask.unsqueeze(-1).to(device=logits.device, dtype=logits.dtype)
        bias = rows * mask * self.watermark_config.delta
        biased_logits[:, start_index:end_index] += bias

        return biased_logits
