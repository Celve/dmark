from __future__ import annotations

import hashlib
import hmac
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
from scipy.stats import norm

from dmark.watermark.base import BaseWatermark, EOS_TOKENS, validate_config_dict

Token = Union[int, str]


@dataclass
class DetectionResult:
    is_watermarked: bool          # whether the sequence is flagged as watermarked
    p_value: float                # false-positive probability
    count: int                    # number of alternating windows observed
    count_support: int            # total number of length-m windows = n - m + 1
    threshold_fpr: float
    details: Dict[str, object]


class PatternMark(BaseWatermark):
    """
    Simplified PATTERN-MARK (two-bucket, alternating only).
    """

    def __init__(self, watermark_config: Dict[str, object], mask_id: int):
        cfg = validate_config_dict(
            watermark_config,
            required={
                "vocab_size": (int,),
                "key": (int,),
                "pattern_length": (int,),
                "delta": (float, int),
            },
            context="PatternMark",
        )

        super().__init__(cfg, mask_id)
        self.m = int(cfg["pattern_length"])
        self.key = int(cfg["key"])
        self.secret_key = str(self.key).encode("utf-8")
        self.vocab_size = int(cfg["vocab_size"])
        self.delta = float(cfg["delta"])
        self.strategy = cfg.get("strategy", "pattern-mark")
        
        # Deterministic PRF-based split into 2 buckets using HMAC-SHA256 % 2
        # We precompute this for all tokens and store as a tensor for efficient access on GPU
        self.buckets = torch.zeros(self.vocab_size, dtype=torch.long)
        for t in range(self.vocab_size):
            msg = str(t).encode("utf-8", errors="ignore")
            h = hmac.new(self.secret_key, msg, hashlib.sha256).digest()
            self.buckets[t] = (h[0] & 1)
            
        # Default RNG seed for first_key generation (if needed)
        seed = int.from_bytes(hashlib.sha256(self.secret_key).digest()[:8], "big")
        self._rng = random.Random(seed)
        
        # Fixed first key for apply_single compatibility
        # In a real scenario we might want this to be random per sequence,
        # but apply_single doesn't give us sequence ID.
        # We use a deterministic first key based on the config key.
        self.first_key = 0 if (self.key % 2) == 0 else 1

    def _get_buckets(self, device):
        if self.buckets.device != device:
            self.buckets = self.buckets.to(device)
        return self.buckets

    def apply_single(
        self,
        curr_logits: torch.Tensor,
        prev_logits: Optional[torch.Tensor],
        prev_token: Optional[torch.Tensor],
        next_logits: Optional[torch.Tensor],
        next_token: Optional[torch.Tensor],
        index: int,
    ) -> torch.Tensor:
        # Determine key for this position: k_i = k_0 XOR (i % 2)
        key = self.first_key ^ (index & 1)
        
        buckets = self._get_buckets(curr_logits.device)
        
        # Add delta to logits where bucket matches key
        # buckets == key gives a boolean mask
        mask = (buckets == key)

        biased_logits = curr_logits.clone()
        biased_logits[mask] += self.delta

        return torch.argmax(biased_logits)

    def apply_range(
        self,
        x: torch.Tensor,
        logits: torch.Tensor,
        start_index: int,
        end_index: int,
    ) -> torch.Tensor:
        biased_logits = logits.clone()
        buckets = self._get_buckets(logits.device)
        
        # Generate keys for the range
        indices = torch.arange(start_index, end_index, device=logits.device)
        keys = self.first_key ^ (indices & 1) # shape: (span,)
        
        # keys needs to be broadcastable to (batch, span, vocab)
        # buckets shape: (vocab,)
        # we want mask shape: (batch, span, vocab)
        
        # Expand keys to (1, span, 1)
        keys_expanded = keys.unsqueeze(0).unsqueeze(-1)
        
        # Expand buckets to (1, 1, vocab)
        buckets_expanded = buckets.unsqueeze(0).unsqueeze(0)
        
        # Create mask
        mask = (buckets_expanded == keys_expanded)

        # Apply bias
        bias = mask.float() * self.delta
        biased_logits[:, start_index:end_index] += bias
        
        return biased_logits

    # ---------- detector (alternating only, sequence-level) ----------
    def detect_sequence(self, sequence: Sequence[Token], fpr_threshold: float = 1e-3) -> DetectionResult:
        """
        Counts length-m alternating windows in the recovered key stream and
        computes an exact p-value under the null (uniform, independent bits).
        """
        n = len(sequence)
        m = self.m
        if n < m:
            return DetectionResult(False, 1.0, 0, 0, fpr_threshold, {"n": n, "m": m})

        # Recover keys via the secret split
        # buckets is on CPU by default or whatever device it was last used on.
        # Ensure we use CPU for detection logic which is likely CPU-bound/scalar.
        buckets_cpu = self.buckets.cpu().numpy()
        keys = [buckets_cpu[int(t)] for t in sequence]

        # Count alternating windows using alternating-tail length
        alt_tail, count = 1, 0
        for i in range(1, n):
            alt_tail = (alt_tail + 1) if (keys[i] != keys[i - 1]) else 1
            if i >= m - 1 and alt_tail >= m:
                count += 1
        support = n - m + 1

        # Null distribution for alternating windows (l=2, alternating T): fast DP
        PTn = self._ptn_two_key_alternating(n, m)
        s = sum(PTn)
        if s <= 0:
            raise RuntimeError("Internal error: P_{T,n} sums to zero.")
        PTn = [p / s for p in PTn]

        p_value = sum(PTn[count:])  # upper tail
        return DetectionResult(
            is_watermarked=(p_value <= fpr_threshold),
            p_value=p_value,
            count=count,
            count_support=support,
            threshold_fpr=fpr_threshold,
            details={"n": n, "m": m, "keys_recovered": keys, "PTn": PTn},
        )

    def detect(self, tokens: torch.Tensor, prompt_len: int) -> Dict[str, object]:
        """Detect PATTERN-MARK watermark and return (rate, z-score).

        We first run the exact alternating-window detector to obtain the
        upper-tail probability (p-value) for observing at least the given
        number of alternating windows. The z-score is then defined as the
        standard-normal quantile corresponding to this upper-tail probability,
        i.e. z such that P(Z >= z) = p_value, Z ~ N(0, 1).
        """
        if tokens.ndim != 1:
            raise ValueError(
                f"detect expects a 1D tensor of token ids, got shape {tuple(tokens.shape)}"
            )

        # Extract generated portion and stop at EOS
        seq: List[int] = []
        for index in range(prompt_len, tokens.shape[0]):
            curr_token = int(tokens[index].item())
            if curr_token in EOS_TOKENS:
                break
            seq.append(curr_token)

        if not seq:
            # No usable tokens – behave like the null with p=1
            return {
                "p_value": 1.0,
                "count": 0,
                "count_support": 0,
                "threshold_fpr": 1e-3,
                "is_watermarked": False,
                "strategy": self.strategy,
            }

        result = self.detect_sequence(seq)
        support = result.count_support
        if support <= 0:
            return {
                "p_value": float(result.p_value),
                "count": result.count,
                "count_support": support,
                "threshold_fpr": result.threshold_fpr,
                "is_watermarked": result.is_watermarked,
                "strategy": self.strategy,
            }

        # Use exact upper-tail probability from the discrete null
        p_value = float(result.p_value)

        return {
            "p_value": p_value,
            "count": result.count,
            "count_support": support,
            "threshold_fpr": result.threshold_fpr,
            "is_watermarked": result.is_watermarked,
            "strategy": self.strategy,
        }

    # ---------- optimized DP for two-key alternating target (only path kept) ----------
    @staticmethod
    def _ptn_two_key_alternating(n: int, m: int) -> List[float]:
        """
        Distribution of C = number of length-m alternating windows in a random
        length-n binary sequence with fair i.i.d. bits.

        State P_i[c][L]: probability that after i tokens, we have c windows and
        current alternating-tail length L in {1..m-1}. Only L up to m-1 matters.
        """
        support = max(0, n - m + 1)
        if support == 0:
            return [1.0]

        # allocate helper
        def zeros():
            return [[0.0 for _ in range(m)] for _ in range(support + 1)]  # ignore index 0 for L

        P_prev = zeros()
        # i=1 initialization: tail length is 1 with prob 1
        P_prev[0][1] = 1.0

        # build up to i = m-1
        for i in range(2, m):
            P_curr = zeros()
            # When next bit equals previous => tail resets to 1 (prob 1/2)
            eq_mass = sum(P_prev[0][L] for L in range(1, i)) * 0.5
            P_curr[0][1] = eq_mass
            # When next bit flips => tail increases by 1 (prob 1/2)
            for L in range(2, i + 1):
                P_curr[0][L] = 0.5 * P_prev[0][L - 1]
            P_prev = P_curr

        # i from m..n
        for i in range(m, n + 1):
            P_curr = zeros()
            max_c_prev = min(i - m, support)
            # next equals previous (prob 1/2): tail -> 1
            for c in range(0, max_c_prev + 1):
                P_curr[c][1] = 0.5 * sum(P_prev[c][L] for L in range(1, m))
            # next flips (prob 1/2): tail increases
            for L in range(2, m):  # L' = L-1 -> L
                for c in range(0, max_c_prev + 1):
                    P_curr[c][L] += 0.5 * P_prev[c][L - 1]
            # if tail hits m (i.e., L becomes m-1 and flips), we add one occurrence
            max_c_curr = min(i - m + 1, support)
            for c in range(1, max_c_curr + 1):
                P_curr[c][m - 1] += 0.5 * P_prev[c - 1][m - 1]
            P_prev = P_curr

        # marginalize over tail length
        PTn = [sum(P_prev[c][L] for L in range(1, m)) for c in range(support + 1)]
        return PTn
