from __future__ import annotations

from typing import Any

import torch
from datasets import load_dataset
from transformers import PreTrainedTokenizerBase

from dmark.dataset.base import Dataset


class ELI5Dataset(Dataset):
    """Match the QA prompting path in dmark/llada/only_gen.py."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        *,
        dataset_name: str = "sentence-transformers/eli5",
        split: str = "train",
        device: str | torch.device = "cuda",
        question_field: str = "question",
        answer_field: str = "answer",
    ) -> None:
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.dataset_name = dataset_name
        self.split = split
        self.question_field = question_field
        self.answer_field = answer_field
        self.dataset = load_dataset(dataset_name, split=split)

    def __len__(self) -> int:
        return len(self.dataset)

    def sample(self, index: int) -> dict[str, Any]:
        return self.get_sample(index)

    def get_sample(self, index: int) -> dict[str, Any]:
        if index < 0 or index >= len(self.dataset):
            raise IndexError(f"Index {index} out of range for split {self.split}")

        raw = self.dataset[index]
        question = self._get_field(raw, self.question_field)
        answer = self._get_field(raw, self.answer_field)

        prompt = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            add_generation_prompt=True,
            tokenize=False,
        )
        tokenized = self.tokenizer(prompt, return_tensors="pt")
        input_ids = tokenized["input_ids"].to(self.device)

        return {
            "prompt": prompt,
            "ground_truth": answer,
            "question": question,
            "input_ids": input_ids,
        }

    def _get_field(self, raw: dict[str, Any], key: str) -> str:
        if key not in raw or raw[key] in (None, ""):
            raise KeyError(f"Field '{key}' missing for dataset {self.dataset_name}")

        value = raw[key]
        if isinstance(value, (list, tuple)):
            if not value:
                raise KeyError(
                    f"Field '{key}' is empty for dataset {self.dataset_name}"
                )
            value = value[0]

        return str(value).strip()
