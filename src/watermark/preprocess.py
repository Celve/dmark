import argparse
import time

from watermark.persistent_bitmap import PersistentBitmap
from watermark.config import WatermarkConfig


def preprocess(watermark_config: WatermarkConfig, file_path: str):
    start_time = time.time()
    print(f"Starting preprocessing for vocab_size={watermark_config.vocab_size}")

    persistent_bitmap = PersistentBitmap(watermark_config.vocab_size, file_path)
    for token in range(watermark_config.vocab_size):
        if token % 1000 == 0:
            elapsed = time.time() - start_time
            print(
                f"Processing token {token}/{watermark_config.vocab_size} ({elapsed:.2f}s elapsed)"
            )

        green_list = watermark_config.gen_green_list(token).bool()
        green_indices = green_list.nonzero().tolist()
        for green_token in green_indices:
            persistent_bitmap.set_bit(green_token, token, True)
    persistent_bitmap.save()

    total_time = time.time() - start_time
    print(f"Preprocessing completed in {total_time:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--bitmap", type=str, required=True)
    args = parser.parse_args()

    watermark_config = WatermarkConfig.from_json(args.config)
    preprocess(watermark_config, args.bitmap)
