import math
from typing import Any, Dict, Optional

import torch

from dmark.watermark.base import BaseWatermark, validate_config_dict
from dmark.watermark.persistent_bitmap import PersistentBitmap


class KGWWatermark(BaseWatermark):
    """Classic KGW watermarking equivalent to the legacy "normal" strategy."""

    def __init__(self, watermark_config: Dict[str, Any], bitmap: PersistentBitmap, mask_id: int):
        cfg = validate_config_dict(
            watermark_config,
            required={
                "vocab_size": (int,),
                "ratio": (float, int),
                "delta": (float, int),
                "key": (int,),
            },
            context="KGWWatermark",
        )

        super().__init__(cfg, mask_id)
        self.bitmap = bitmap
        self.vocab_size = int(cfg["vocab_size"])
        self.ratio = float(cfg["ratio"])
        self.delta = float(cfg["delta"])
        self.key = int(cfg["key"])

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
        return row * self.delta

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
        bias = rows * mask * self.delta
        biased_logits[:, start_index:end_index] += bias

        return biased_logits

    def detect(self, tokens: torch.Tensor, prompt_len: int) -> Dict[str, Any]:
        detected = 0
        gen_len = 0

        for index in range(prompt_len, tokens.shape[0]):
            prev_token = tokens[index - 1]
            curr_token = tokens[index]
            if curr_token == 126081 or curr_token == 126348:
                break
            if self.bitmap.get_row(prev_token)[curr_token.item()]:
                detected += 1
            gen_len += 1
        if gen_len == 0:
            return {
                "detection_rate": 0.0,
                "z_score": 0.0,
            }

        detection_rate = detected / gen_len
        expected = gen_len * self.ratio
        variance = gen_len * self.ratio * (1.0 - self.ratio)
        if variance <= 0.0:
            z_score = 0.0
        else:
            z_score = (detected - expected) / math.sqrt(variance)
        return {
            "detection_rate": detection_rate,
            "z_score": z_score,
        }
