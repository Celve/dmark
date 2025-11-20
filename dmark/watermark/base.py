from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import torch

EOS_TOKENS = {126081, 126348}
# Token IDs treated as EOS by Meta Llama models; kept here for shared masking logic.


def validate_config_dict(
    config: Dict[str, Any],
    required: Dict[str, Tuple[type, ...]],
    context: str,
) -> Dict[str, Any]:
    """Ensure ``config`` contains required keys with expected types.

    Args:
        config: User-provided configuration dictionary.
        required: Mapping from key name to acceptable types.
        context: Human-readable name of the caller for error messages.

    Returns:
        The original ``config`` dict if validation succeeds.

    Raises:
        TypeError: If ``config`` is not a dict or a value has the wrong type.
        ValueError: If any required keys are missing.
    """
    if not isinstance(config, dict):
        raise TypeError(f"{context} expected config as dict, got {type(config).__name__}")

    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(
            f"{context} missing required config keys: {', '.join(missing)}"
        )

    type_errors = []
    for key, expected_types in required.items():
        value = config[key]
        if not isinstance(expected_types, tuple):
            expected_types = (expected_types,)
        if not isinstance(value, expected_types):
            expected_names = ", ".join(t.__name__ for t in expected_types)
            type_errors.append(
                f"{key} (expected {expected_names}, got {type(value).__name__})"
            )

    if type_errors:
        joined = "; ".join(type_errors)
        raise TypeError(f"{context} has invalid config types: {joined}")

    return config


class BaseWatermark(ABC):
    """Abstract interface for watermarkers used across generation and detection.

    Generation in this project can be order-agnostic (spans sampled out of order),
    so implementations should rely on explicit indices rather than implicit causal
    state when mutating logits.
    """

    def __init__(self, watermark_config: Dict[str, Any], mask_id: int):
        """Persist common configuration and the model-specific mask token."""
        self.watermark_config = watermark_config
        self.mask_id = mask_id

    @abstractmethod
    def apply_single(
        self,
        curr_logits: torch.Tensor,
        prev_logits: Optional[torch.Tensor],
        prev_token: Optional[torch.Tensor],
        next_logits: Optional[torch.Tensor],
        next_token: Optional[torch.Tensor],
        index: int,
    ) -> torch.Tensor:
        """Adjust logits for a single token position during generation.

        The provided ``index`` is the absolute position; do not assume sequential
        traversal because decoding may visit positions out of order.
        """
        raise NotImplementedError

    @abstractmethod
    def apply_range(
        self,
        x: torch.Tensor,
        logits: torch.Tensor,
        start_index: int,
        end_index: int,
    ) -> torch.Tensor:
        """Apply the watermark to a slice of logits, inclusive of ``start_index``.

        Called by order-agnostic generation loops; implementations should key off
        ``start_index``/``end_index`` instead of assuming consecutive decoding.
        """
        raise NotImplementedError

    @abstractmethod
    def detect(self, tokens: torch.Tensor, prompt_len: int) -> Dict[str, Any]:
        """Return detection metrics (e.g., z-score) for a completed sequence."""
        raise NotImplementedError
