import random
import numpy as np

from dmark.watermark.config import WatermarkConfig


def detect(input_ids, watermark_config: WatermarkConfig) -> tuple[float, float]:
    is_watermark = [False] * len(input_ids)
    for j in range(1, len(input_ids)):
        if watermark_config.strategy == "legacy-ahead":
            rng = random.Random(input_ids[j - 1] * watermark_config.key)
            watermark_number = int(watermark_config.ratio * watermark_config.vocab_size)
            watermark_samples = rng.sample(
                range(watermark_config.vocab_size), watermark_number
            )
            if input_ids[j] in watermark_samples:
                is_watermark[j] = True
        elif watermark_config.strategy == "legacy-both":
            rng = random.Random(input_ids[j - 1] * watermark_config.key)
            watermark_number = int(watermark_config.ratio * watermark_config.vocab_size)
            watermark_samples = rng.sample(
                range(watermark_config.vocab_size), watermark_number
            )
            if input_ids[j] in watermark_samples:
                is_watermark[j] = True

            if j != len(input_ids) - 1:
                rng = random.Random(input_ids[j + 1] * watermark_config.key)
                watermark_number = int(
                    watermark_config.ratio * watermark_config.vocab_size
                )
                watermark_samples = rng.sample(
                    range(watermark_config.vocab_size), watermark_number
                )
                if input_ids[j] in watermark_samples:
                    is_watermark[j] = True
    detected = np.sum(is_watermark)
    gen_len = len(is_watermark)
    return detected / gen_len, 2 * (detected - gen_len / 2.0) / np.sqrt(gen_len)
