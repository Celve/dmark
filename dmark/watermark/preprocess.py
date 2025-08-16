import argparse
import time

import torch
from tqdm import tqdm

from watermark.config import WatermarkConfig
from watermark.persistent_bitmap import PersistentBitmap


def preprocess(watermark_config: WatermarkConfig, file_path: str):
    start_time = time.time()
    print(f"Starting preprocessing for vocab_size={watermark_config.vocab_size}")

    persistent_bitmap = PersistentBitmap(watermark_config.vocab_size, file_path)

    for token in tqdm(range(watermark_config.vocab_size), desc="Processing tokens"):
        green_list = watermark_config.gen_green_list(torch.tensor(token)).bool()
        green_list_ndarray = green_list.numpy()
        persistent_bitmap.set_row(token, green_list_ndarray)

    persistent_bitmap.transpose()
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
