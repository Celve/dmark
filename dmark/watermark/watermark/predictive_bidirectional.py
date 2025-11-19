from typing import Optional

import torch

from dmark.watermark.config import WatermarkConfig
from dmark.watermark.persistent_bitmap import PersistentBitmap
from dmark.watermark.watermark.base import BaseWatermark


class PredictiveBidirectionalWatermark(BaseWatermark):
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

    def _resolve_prev_token(
        self,
        prev_token: Optional[torch.Tensor],
        prev_logits: Optional[torch.Tensor],
    ) -> Optional[int]:
        if prev_token is not None:
            return self._token_to_index(prev_token)
        if prev_logits is None:
            return None
        return int(prev_logits.argmax(dim=-1).detach().item())

    def _resolve_next_token(
        self,
        next_token: Optional[torch.Tensor],
        next_logits: Optional[torch.Tensor],
    ) -> Optional[int]:
        if next_token is not None:
            return self._token_to_index(next_token)
        if next_logits is None:
            return None
        return int(next_logits.argmax(dim=-1).detach().item())

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
        next_idx = self._token_to_index(next_token)

        if prev_idx is None and next_idx is None:
            prev_idx = self._resolve_prev_token(prev_token, prev_logits)
            next_idx = None

        biased_logits = curr_logits
        if prev_idx is not None:
            biased_logits = biased_logits + self._row_bias(prev_idx, curr_logits)
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
        batch_size = x.shape[0]
        vocab_size = self.bitmap.vocab_size
        span = end_index - start_index
        device = logits.device

        # Previous tokens
        prev_slice = x[:, start_index - 1 : end_index - 1]
        prev_valid = prev_slice != self.mask_id

        safe_prev = prev_slice.clone()
        safe_prev[~prev_valid] = 0
        prev_rows = self.bitmap.get_rows(safe_prev.reshape(-1)).reshape(
            batch_size,
            span,
            vocab_size,
        )
        prev_rows = prev_rows.to(device=device, dtype=logits.dtype)
        prev_mask = prev_valid.unsqueeze(-1).to(device=device, dtype=logits.dtype)

        # Next tokens
        next_slice = x[:, start_index + 1 : end_index + 1]
        if next_slice.shape[1] < span:
            pad = torch.full(
                (batch_size, span - next_slice.shape[1]),
                self.mask_id,
                dtype=x.dtype,
                device=x.device,
            )
            next_slice = torch.cat([next_slice, pad], dim=1)

        next_valid = next_slice != self.mask_id
        safe_next = next_slice.clone()
        safe_next[~next_valid] = 0
        next_cols = self.bitmap.get_cols(safe_next.reshape(-1))
        next_cols = next_cols.permute(1, 0).reshape(batch_size, span, vocab_size)
        next_cols = next_cols.to(device=device, dtype=logits.dtype)
        next_mask = next_valid.unsqueeze(-1).to(device=device, dtype=logits.dtype)

        # Identify positions where neither prev nor next provides real context
        predict_mask = (~prev_valid) & (~next_valid)
        if torch.any(predict_mask):
            filler = logits[:, start_index - 1 : end_index - 1].argmax(dim=-1)
            filler_rows = self.bitmap.get_rows(filler.reshape(-1)).reshape(
                batch_size,
                span,
                vocab_size,
            )
            filler_rows = filler_rows.to(device=device, dtype=logits.dtype)
            prev_rows[predict_mask] = filler_rows[predict_mask]
            prev_mask[predict_mask] = 1.0

        total_bias = (prev_rows * prev_mask) + (next_cols * next_mask)

        total_bias *= self.watermark_config.delta
        biased_logits[:, start_index:end_index] += total_bias
        return biased_logits
