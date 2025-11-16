from typing import Optional

import torch

from dmark.watermark.config import WatermarkConfig
from dmark.watermark.persistent_bitmap import PersistentBitmap


class Watermark:
    def __init__(self, watermark_config: WatermarkConfig, bitmap: PersistentBitmap):
        self.watermark_config = watermark_config
        self.bitmap = bitmap

    def init(self):
        self.double = 0
        self.green = 0
    
    def gen_green_list(self, prev_token: int) -> torch.Tensor:
        return self.bitmap.get_row(prev_token)
    
    def is_token_in_green_list(self, token: int, prev_token: int) -> bool:
        green_list = self.bitmap.get_row(prev_token)
        return bool(green_list[token].item())

    def _token_to_index(self, token: Optional[torch.Tensor | int]) -> Optional[int]:
        if token is None:
            return None
        if isinstance(token, torch.Tensor):
            return int(token.detach().item())
        return int(token)

    def _resolve_token_index(
        self,
        token: Optional[torch.Tensor | int],
        logits: Optional[torch.Tensor],
        *,
        require_logits: bool,
    ) -> Optional[int]:
        if token is None:
            if logits is None:
                if require_logits:
                    raise ValueError("Logits required to predict token when token is None")
                return None
            token = logits.argmax(dim=-1)
        return self._token_to_index(token)

    def _scaled_bitmap_bias(
        self,
        vector: torch.Tensor,
        template: torch.Tensor,
    ) -> torch.Tensor:
        non_blocking = vector.device.type == "cpu" and template.device.type == "cuda"
        bias = vector.to(device=template.device, dtype=template.dtype, non_blocking=non_blocking)
        return bias * self.watermark_config.delta

    def _row_bias(self, token_index: int, template: torch.Tensor) -> torch.Tensor:
        return self._scaled_bitmap_bias(self.bitmap.get_row(token_index), template)

    def _col_bias(self, token_index: int, template: torch.Tensor) -> torch.Tensor:
        return self._scaled_bitmap_bias(self.bitmap.get_col(token_index), template)

    def apply_once(
        self,
        curr_logits: torch.Tensor,
        prev_logits: Optional[torch.Tensor],
        prev_token: Optional[torch.Tensor],
        next_logits: Optional[torch.Tensor],
        next_token: Optional[torch.Tensor],
    ) -> torch.Tensor:
        strategy = self.watermark_config.strategy
        biases: list[torch.Tensor] = []

        if strategy == "normal":
            prev_idx = self._token_to_index(prev_token)
            if prev_idx is not None:
                biases.append(self._row_bias(prev_idx, curr_logits))
        elif strategy == "predict":
            prev_idx = self._resolve_token_index(prev_token, prev_logits, require_logits=True)
            biases.append(self._row_bias(prev_idx, curr_logits))
        elif strategy == "bidirectional":
            prev_idx = self._token_to_index(prev_token)
            if prev_idx is not None:
                biases.append(self._row_bias(prev_idx, curr_logits))

            next_idx = self._token_to_index(next_token)
            if next_idx is not None:
                biases.append(self._col_bias(next_idx, curr_logits))
        elif strategy == "predict-bidirectional":
            prev_idx = self._resolve_token_index(prev_token, prev_logits, require_logits=True)
            biases.append(self._row_bias(prev_idx, curr_logits))

            next_idx = self._resolve_token_index(next_token, next_logits, require_logits=False)
            if next_idx is not None:
                biases.append(self._col_bias(next_idx, curr_logits))
        else:
            raise ValueError(f"Invalid strategy: {strategy}")

        biased_logits = curr_logits
        for bias in biases:
            biased_logits = biased_logits + bias
        result = torch.argmax(biased_logits)  # TODO: sampling also matters here
        return result

    def apply_all(
        self,
        x: torch.Tensor,
        mask_id: int,
        logits: torch.Tensor,
        start_index: int,
        end_index: int,
    ) -> torch.Tensor:
        biased_logits = logits.clone()
        prev_slice = x[:, start_index - 1 : end_index - 1]
        model_pred = logits[:, start_index - 1 : end_index - 1].argmax(dim=-1)
        predicted = torch.where(
            prev_slice == mask_id,
            model_pred,
            prev_slice.to(model_pred.device),
        )
        for i in range(logits.shape[0]):
            bias = (
                (x[i, start_index - 1 : end_index - 1] != mask_id).unsqueeze(1).float()
                * self.bitmap.get_rows(predicted[i]).float().to(logits.device)
            ) * self.watermark_config.delta
            biased_logits[i, start_index:end_index] += bias
        return biased_logits
