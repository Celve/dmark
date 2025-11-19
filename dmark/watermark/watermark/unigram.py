from typing import Optional, Dict, List, Any
import torch
import numpy as np
from scipy.stats import hypergeom

from dmark.watermark.config import WatermarkConfig
from dmark.watermark.watermark.base import BaseWatermark

class UnigramWatermark(BaseWatermark):
    def __init__(self, watermark_config: WatermarkConfig, mask_id: int):
        super().__init__(watermark_config, mask_id)
        self.seed = watermark_config.key
        self.delta = watermark_config.delta
        self.gamma = watermark_config.ratio
        self.vocab_size = watermark_config.vocab_size
        
        # We assume device is cuda if available, else cpu, consistent with other parts or config
        # However, BaseWatermark doesn't store device. We'll init greenlist on CPU first or 
        # rely on the device of the tensors passed in apply methods.
        # But the user code init_greenlist uses a specific device. 
        # Let's use 'cuda' if available as default or check config.
        # The config doesn't explicitly have device. 
        # We will initialize greenlist lazily or on a default device.
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._init_greenlist()

    def _init_greenlist(self):
        partition_size = int(self.vocab_size * self.gamma)

        # Fork RNG to ensure reproducibility regardless of global state
        with torch.random.fork_rng(devices=[self.device] if self.device == "cuda" else None):
            torch.manual_seed(self.seed)
            # We need to make sure we are on the right device
            vocab_permutation = torch.randperm(self.vocab_size, device=self.device)
            self.greenlist = vocab_permutation[:partition_size]

    def get_key_params(self):
        return {
            "seed": self.seed,
            "delta": self.delta,
            "gamma": self.gamma,
        }
    
    def apply_single(
        self,
        curr_logits: torch.Tensor,
        prev_logits: Optional[torch.Tensor],
        prev_token: Optional[torch.Tensor],
        next_logits: Optional[torch.Tensor],
        next_token: Optional[torch.Tensor],
        index: int,
    ) -> torch.Tensor:
        # curr_logits: [batch_size, vocab_size]
        
        # Ensure greenlist is on the same device as logits
        if self.greenlist.device != curr_logits.device:
            self.greenlist = self.greenlist.to(curr_logits.device)

        biased_logits = curr_logits.clone()
        biased_logits[:, self.greenlist] += self.delta
        
        return torch.argmax(biased_logits, dim=-1)

    def apply_range(
        self,
        x: torch.Tensor,
        logits: torch.Tensor,
        start_index: int,
        end_index: int,
    ) -> torch.Tensor:
        # logits: [batch_size, seq_len, vocab_size]
        
        if self.greenlist.device != logits.device:
            self.greenlist = self.greenlist.to(logits.device)

        biased_logits = logits.clone()
        # Apply to the specified range
        # logits[:, start_index:end_index, greenlist] += delta
        # We need to be careful with indexing. 
        # logits is [batch, seq, vocab]
        # We want to add delta to all time steps in range, for greenlist tokens.
        
        biased_logits[:, start_index:end_index, self.greenlist] += self.delta
        
        return biased_logits
