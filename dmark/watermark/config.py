import json
from typing import Optional

import torch

from pydantic import BaseModel


class WatermarkConfig(BaseModel):
    vocab_size: int
    ratio: float
    delta: float
    key: int
    prebias: bool = False
    strategy: Optional[str] = None
    bitmap_path: Optional[str] = None

    def gen_green_list(self, prev_token: torch.Tensor) -> torch.Tensor:
        torch.manual_seed(prev_token.item() * self.key)
        return torch.bernoulli(torch.full((self.vocab_size,), self.ratio, device="cuda"))

    def gen_bias(self, prev_token: torch.Tensor) -> torch.Tensor:
        return self.gen_green_list(prev_token) * self.delta
