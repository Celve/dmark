import json
import torch


class WatermarkConfig:
    def __init__(self, vocab_size: int, ratio: float, delta: float, key: int):
        self.vocab_size = vocab_size
        self.ratio = ratio
        self.delta = delta
        self.key = key

    @classmethod
    def from_json(cls, json_file: str) -> "WatermarkConfig":
        with open(json_file, 'r') as f:
            data = json.load(f)
        return cls(
            vocab_size=data['vocab_size'],
            ratio=data['ratio'],
            delta=data['delta'],
            key=data['key']
        )

    def gen_green_list(self, prev_token: torch.Tensor) -> torch.Tensor:
        torch.manual_seed(prev_token.item() * self.key)
        return torch.bernoulli(torch.full((self.vocab_size,), self.ratio))

    def gen_bias(self, prev_token: torch.Tensor) -> torch.Tensor:
        return self.gen_green_list(prev_token) * self.delta