from __future__ import annotations

from typing import Any

import torch
from datasets import load_dataset
from transformers import PreTrainedTokenizerBase

from dmark.dataset.base import Dataset


class C4Dataset(Dataset):
    """Wrapper around the C4 dataset replicating only_gen text prompting logic."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        *,
        dataset_name: str = "allenai/c4",
        config_name: str = "en",
        split: str = "train",
        device: str | torch.device = "cuda",
        prompt_tokens: int = 30,
        continuation_tokens: int = 256,
        text_field: str = "text",
        streaming: bool = True,
    ) -> None:
        if prompt_tokens <= 0:
            raise ValueError("prompt_tokens must be positive")
        if continuation_tokens <= 0:
            raise ValueError("continuation_tokens must be positive")

        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.dataset_name = dataset_name
        self.config_name = config_name
        self.split = split
        self.prompt_tokens = prompt_tokens
        self.continuation_tokens = continuation_tokens
        self.text_field = text_field
        self.min_required_tokens = prompt_tokens + continuation_tokens
        self.streaming = streaming

        if streaming:
            self._dataset_iter = iter(
                load_dataset(
                    dataset_name,
                    config_name,
                    split=split,
                    streaming=True,
                )
            )
            self._dataset = None
            self._dataset_length = None
        else:
            self._dataset = load_dataset(dataset_name, config_name, split=split)
            self._dataset_iter = None
            self._dataset_length = len(self._dataset)

        self._dataset_position = 0
        self._cache: list[dict[str, Any]] = []

    def __len__(self) -> int:
        if self._dataset_length is None:
            raise TypeError(
                "Streaming C4 dataset does not have a predefined length. Disable "
                "streaming to access __len__."
            )
        return self._dataset_length

    def sample(self, index: int) -> dict[str, Any]:
        return self.get_sample(index)

    def get_sample(self, index: int) -> dict[str, Any]:
        if index < 0:
            raise IndexError("Index must be non-negative")

        self._ensure_cached(index)
        return self._cache[index]

    def _ensure_cached(self, target_index: int) -> None:
        while len(self._cache) <= target_index:
            try:
                raw = self._next_row()
            except StopIteration:
                raise IndexError(
                    f"Dataset exhausted before reaching sample {target_index}"
                ) from None

            sample = self._build_sample(raw)
            if sample is None:
                continue
            self._cache.append(sample)

    def _next_row(self) -> dict[str, Any]:
        if self.streaming:
            if self._dataset_iter is None:
                raise RuntimeError("Streaming iterator not initialized")
            return next(self._dataset_iter)

        assert self._dataset is not None
        if self._dataset_position >= len(self._dataset):
            raise StopIteration
        row = self._dataset[self._dataset_position]
        self._dataset_position += 1
        return row

    def _build_sample(self, raw: dict[str, Any]) -> dict[str, Any] | None:
        text = raw.get(self.text_field)
        if not isinstance(text, str) or not text.strip():
            return None

        tokenized = self.tokenizer(text, return_tensors="pt")
        text_tokens = tokenized["input_ids"][0]

        if len(text_tokens) < self.min_required_tokens:
            return None

        prompt_ids = text_tokens[: self.prompt_tokens]
        gt_ids = text_tokens[
            self.prompt_tokens : self.prompt_tokens + self.continuation_tokens
        ]

        prompt = self.tokenizer.decode(prompt_ids, skip_special_tokens=True)
        ground_truth = self.tokenizer.decode(gt_ids, skip_special_tokens=True)
        input_ids = prompt_ids.unsqueeze(0).to(self.device)

        return {
            "prompt": prompt,
            "ground_truth": ground_truth,
            "input_ids": input_ids,
        }
