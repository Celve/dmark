from abc import ABC
from typing import Optional

import torch

from dmark.watermark.config import WatermarkConfig


class BaseWatermark(ABC): 
    def __init__(self, watermark_config: WatermarkConfig, mask_id: int):
        self.watermark_config = watermark_config
        self.mask_id = mask_id

    def apply_single(
        self, 
        curr_logits: torch.Tensor,
        prev_logits: Optional[torch.Tensor],
        prev_token: Optional[torch.Tensor],
        next_logits: Optional[torch.Tensor],
        next_token: Optional[torch.Tensor],
        index: int,
    ) -> torch.Tensor:
        raise NotImplementedError

    def apply_range(
        self, 
        x: torch.Tensor,
        logits: torch.Tensor,
        start_index: int,
        end_index: int,
    ) -> torch.Tensor:
        raise NotImplementedError