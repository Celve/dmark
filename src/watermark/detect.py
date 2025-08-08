import math

import torch

from watermark.config import WatermarkConfig


class Detector:
    def __init__(self, watermark_config: WatermarkConfig):
        self.config = watermark_config

    def detect(self, tokens: torch.Tensor, prompt_len: int) -> float:
        detected = 0
        gen_len = 0
        for index in range(prompt_len, tokens.shape[0]):
            prev_token = tokens[index - 1]
            curr_token = tokens[index]
            if curr_token == 126081: 
                break
            green_list = self.config.gen_green_list(prev_token).bool()
            if green_list[curr_token.item()]:
                detected += 1
            gen_len += 1
        return detected / gen_len
