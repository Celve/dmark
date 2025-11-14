from __future__ import annotations

from typing import Any

import torch
from datasets import load_dataset
from transformers import PreTrainedTokenizerBase

from dmark.dataset.base import Dataset


# Mirrors the instruction and formatting logic inside dmark/llada/only_gen.py so
# that dataset sampling and standalone generation stay consistent.
MATH_INSTRUCTION = (
    "You are a math expert. You will be given a question to solve. "
    "Solve it step by step. Wrap the final answer in a \\boxed{{}}.\n"
    "Respond in the following format:\n"
    "<reasoning>\n"
    "Your reasoning here\n"
    "</reasoning>\n"
    "<answer>\n"
    "\\boxed{...}\n"
    "</answer>"
)


class GSM8KDataset(Dataset):
    """GSM8K dataset wrapper using the same prompt shaping as only_gen.py."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        *,
        dataset_name: str = "openai/gsm8k",
        split: str = "train",
        subset: str | None = "main",
        device: str | torch.device = "cuda",
    ) -> None:
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        allowed = {"gsm8k", "openai/gsm8k"}
        if dataset_name not in allowed:
            raise ValueError(
                f"{self.__class__.__name__} only supports {allowed}, got {dataset_name}"
            )
        self.dataset_name = dataset_name
        self.split = split
        self.subset = subset or "main"

        self.dataset = load_dataset("gsm8k", self.subset, split=split)

    def __len__(self) -> int:
        return len(self.dataset)

    def sample(self, index: int) -> dict[str, Any]:
        return self.get_sample(index)

    def get_sample(self, index: int) -> dict[str, Any]:
        if index < 0 or index >= len(self.dataset):
            raise IndexError(f"Index {index} out of range for split {self.split}")

        question = self.dataset[index]["question"]
        answer = self.dataset[index]["answer"]

        prompt_body = f"{MATH_INSTRUCTION}\n\n{question}"
        chat_prompt = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_body}],
            add_generation_prompt=True,
            tokenize=False,
        )
        tokenized = self.tokenizer(chat_prompt, return_tensors="pt")
        input_ids = tokenized["input_ids"].to(self.device)

        return {
            "prompt": chat_prompt,
            "ground_truth": answer,
            "question": question,
            "input_ids": input_ids,
        }
