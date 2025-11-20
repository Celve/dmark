import math
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from scipy.stats import hypergeom

from dmark.watermark.base import BaseWatermark, EOS_TOKENS, validate_config_dict


class UnigramWatermark(BaseWatermark):
    def __init__(self, watermark_config: Dict[str, object], mask_id: int):
        cfg = validate_config_dict(
            watermark_config,
            required={
                "vocab_size": (int,),
                "delta": (float, int),
                "gamma": (float, int),
                "key": (int,),
            },
            context="UnigramWatermark",
        )

        super().__init__(cfg, mask_id)
        self.seed = int(cfg["key"])
        self.delta = float(cfg["delta"])
        self.gamma = float(cfg["gamma"])
        self.vocab_size = int(cfg["vocab_size"])
        self.strategy = cfg.get("strategy", "unigram")
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
            # Cache as Python set for fast membership checks during detection
            self._green_set = set(int(t) for t in self.greenlist.tolist())

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

    def detect(self, tokens: torch.Tensor, prompt_len: int) -> Dict[str, Any]:
        """Detect unigram watermark by counting hits in the global greenlist."""
        if tokens.ndim != 1:
            raise ValueError(
                f"detect expects a 1D tensor of token ids, got shape {tuple(tokens.shape)}"
            )

        detected = 0
        gen_len = 0

        for index in range(prompt_len, tokens.shape[0]):
            curr_token = int(tokens[index].item())
            if curr_token in EOS_TOKENS:
                break
            if curr_token in self._green_set:
                detected += 1
            gen_len += 1

        if gen_len == 0:
            return {
                "detection_rate": 0.0,
                "z_score": 0.0,
                "detected": 0,
                "gen_len": 0,
                "strategy": self.strategy,
            }

        gamma = self.gamma
        detection_rate = detected / gen_len
        expected = gen_len * gamma
        variance = gen_len * gamma * (1.0 - gamma)
        if variance <= 0.0:
            z_score = 0.0
        else:
            z_score = (detected - expected) / math.sqrt(variance)

        return {
            "detection_rate": detection_rate,
            "z_score": z_score,
            "detected": detected,
            "gen_len": gen_len,
            "strategy": self.strategy,
        }
