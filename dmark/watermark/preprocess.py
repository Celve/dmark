import argparse
import time

import torch
from tqdm import tqdm

from dmark.watermark.config import WatermarkConfig
from dmark.watermark.persistent_bitmap import PersistentBitmap


def preprocess(watermark_config: WatermarkConfig, file_path: str, device: str = "cuda"):
    start_time = time.time()
    print(f"Starting preprocessing for vocab_size={watermark_config.vocab_size}")

    persistent_bitmap = PersistentBitmap(watermark_config.vocab_size, file_path, initialize=True, device=device)

    for token in tqdm(range(watermark_config.vocab_size), desc="Processing tokens"):
        green_list = watermark_config.gen_green_list(torch.tensor(token)).bool()
        persistent_bitmap.set_row(token, green_list)

    persistent_bitmap.save()

    total_time = time.time() - start_time
    print(f"Preprocessing completed in {total_time:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess and generate bitmap for watermarking")
    
    # Required arguments
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default=".",
        help="Directory to save the bitmap file (default: current directory)"
    )
    
    # Essential parameters for bitmap generation
    parser.add_argument(
        "--vocab_size", 
        type=int, 
        default=126464,
        help="Vocabulary size (default: 126464)"
    )
    parser.add_argument(
        "--ratio", 
        type=float, 
        default=0.5,
        help="Green list ratio (default: 0.5)"
    )
    parser.add_argument(
        "--key", 
        type=int, 
        default=42,
        help="Random seed key for watermark (default: 42)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cpu", "cuda"],
        help="Device to create bitmap on (default: cuda)"
    )
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate bitmap filename based on parameters
    bitmap_filename = f"bitmap_v{args.vocab_size}_r{int(args.ratio*100)}_k{args.key}.bin"
    bitmap_path = os.path.join(args.output_dir, bitmap_filename)
    
    # Create watermark config with only essential parameters
    # Note: delta, prebias, and strategy are not needed for bitmap generation
    watermark_config = WatermarkConfig(
        vocab_size=args.vocab_size,
        ratio=args.ratio,
        delta=2.0,  # Default value, not used in preprocessing
        key=args.key,
        prebias=False,  # Default value, not used in preprocessing
        strategy="normal"  # Default value, not used in preprocessing
    )
    
    print(f"Generating bitmap: vocab_size={args.vocab_size}, ratio={args.ratio}, key={args.key}")
    print(f"Output file: {bitmap_path}")
    print(f"Device: {args.device}")
    
    preprocess(watermark_config, bitmap_path, args.device)
