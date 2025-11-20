import os
from typing import Optional
import time
import argparse

import torch
import numpy as np
from tqdm import tqdm
from pydantic import BaseModel

class PersistentBitmapConfig(BaseModel):
    vocab_size: int
    ratio: float
    key: int

    def gen_green_list(self, prev_token: torch.Tensor, device: str = "cuda") -> torch.Tensor:
        g = torch.Generator(device=device)
        g.manual_seed(prev_token.item() * self.key)
        probs = torch.full((self.vocab_size,), self.ratio, device=device)
        return torch.bernoulli(probs, generator=g)


class PersistentBitmap:
    def __init__(self, vocab_size: int, filepath: str, initialize: bool = False, device: str = "cuda"):
        self.vocab_size = vocab_size
        self.filepath = filepath
        self.device = torch.device(device)

        if initialize:
            self._initialize()
        else:
            if not os.path.exists(filepath):
                raise FileNotFoundError(
                    f"Bitmap file not found: {filepath}\n"
                    f"Please run preprocessing first to generate the bitmap:\n"
                    f"  python -m dmark.watermark.preprocess --output_dir <dir> --vocab_size {vocab_size} --ratio <ratio> --key <key>"
                )
            self._load()

    def _initialize(self):
        self.matrix = torch.zeros((self.vocab_size, self.vocab_size), dtype=torch.bool, device=self.device)
        self._maybe_pin_memory()
        self._save()

    def _load(self):
        # Load packed bits from file
        packed_data = np.fromfile(self.filepath, dtype=np.uint8)
        
        # Calculate expected size
        expected_bytes = (self.vocab_size * self.vocab_size + 7) // 8
        if len(packed_data) != expected_bytes:
            raise ValueError(
                f"Bitmap file size mismatch: expected {expected_bytes} bytes, got {len(packed_data)}"
            )
        
        # Unpack bits to boolean matrix
        # First, unpack all bits
        unpacked_bits = np.unpackbits(packed_data)
        
        # Take only the needed bits (in case of padding)
        needed_bits = self.vocab_size * self.vocab_size
        unpacked_bits = unpacked_bits[:needed_bits]
        
        # Convert to torch tensor and reshape
        self.matrix = (
            torch.from_numpy(unpacked_bits)
            .bool()
            .reshape(self.vocab_size, self.vocab_size)
            .to(self.device)
        )
        self._maybe_pin_memory()

    def _save(self):
        # Convert to numpy for packing
        numpy_matrix = self.matrix.cpu().numpy()
        
        # Flatten the boolean matrix
        flat_matrix = numpy_matrix.flatten()
        
        # Convert to uint8 (0 or 1)
        uint8_matrix = flat_matrix.astype(np.uint8)
        
        # Pack bits (8 bits per byte)
        packed_data = np.packbits(uint8_matrix)
        
        # Save to file
        packed_data.tofile(self.filepath)

    def set_bit(self, x: int, y: int, value: bool):
        if x >= self.vocab_size or y >= self.vocab_size:
            raise IndexError(f"Token index out of bounds: vocab_size={self.vocab_size}")
        self.matrix[x, y] = value

    def get_bit(self, x: int, y: int) -> bool:
        if x >= self.vocab_size or y >= self.vocab_size:
            raise IndexError(f"Token index out of bounds: vocab_size={self.vocab_size}")
        return self.matrix[x, y].item()

    def save(self):
        self._save()

    def get_row(self, x: int) -> torch.Tensor:
        if x >= self.vocab_size:
            raise IndexError(f"Token index out of bounds: vocab_size={self.vocab_size}")
        return self.matrix[x]
    
    def get_rows(self, indices: torch.Tensor) -> torch.Tensor:
        """Efficiently get multiple rows using tensor indexing.
        
        Args:
            indices: 1D tensor of row indices
            
        Returns:
            2D tensor of shape (len(indices), vocab_size)
        """
        # Move indices to the same device as the matrix if needed
        if indices.device != self.matrix.device:
            indices = indices.to(self.matrix.device)
        return self.matrix[indices]
    
    def get_col(self, y: int) -> torch.Tensor:
        if y >= self.vocab_size:
            raise IndexError(f"Token index out of bounds: vocab_size={self.vocab_size}")
        return self.matrix[:, y]
    
    def get_cols(self, indices: torch.Tensor) -> torch.Tensor:
        """Efficiently get multiple columns using tensor indexing.
        
        Args:
            indices: 1D tensor of column indices
            
        Returns:
            2D tensor of shape (vocab_size, len(indices))
        """
        # Move indices to the same device as the matrix if needed
        if indices.device != self.matrix.device:
            indices = indices.to(self.matrix.device)
        return self.matrix[:, indices]

    def set_row(self, x: int, values):
        if x >= self.vocab_size:
            raise IndexError(f"Token index out of bounds: vocab_size={self.vocab_size}")
        
        # Handle both numpy arrays and torch tensors
        if isinstance(values, np.ndarray):
            values = torch.from_numpy(values)
        elif not isinstance(values, torch.Tensor):
            values = torch.tensor(values)
            
        if len(values) != self.vocab_size:
            raise ValueError(f"Values array must have length {self.vocab_size}")
        
        # Move values to the same device as the matrix if needed
        if values.device != self.matrix.device:
            values = values.to(self.matrix.device)
        
        self.matrix[x] = values.bool()
    
    def transpose(self):
        """Transpose the bitmap in-place, swapping x and y dimensions."""
        self.matrix = self.matrix.T
    
    def to(self, device):
        """Move the matrix to a specific device (CPU/GPU)."""
        self.matrix = self.matrix.to(device)
        self.device = torch.device(device)
        self._maybe_pin_memory()
        return self

    def _maybe_pin_memory(self):
        """Pin host memory so host→device transfers can be non-blocking."""
        if self.matrix.device.type == "cpu" and not self.matrix.is_pinned():
            self.matrix = self.matrix.pin_memory()

def preprocess(bitmap_config: PersistentBitmapConfig, file_path: str, device: str = "cuda"):
    start_time = time.time()
    print(f"Starting preprocessing for vocab_size={bitmap_config.vocab_size}")

    persistent_bitmap = PersistentBitmap(bitmap_config.vocab_size, file_path, initialize=True, device=device)

    for token in tqdm(range(bitmap_config.vocab_size), desc="Processing tokens"):
        green_list = bitmap_config.gen_green_list(torch.tensor(token), device=device).bool()
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
    bitmap_config = PersistentBitmapConfig(
        vocab_size=args.vocab_size,
        ratio=args.ratio,
        key=args.key,
    )
    
    print(f"Generating bitmap: vocab_size={args.vocab_size}, ratio={args.ratio}, key={args.key}")
    print(f"Output file: {bitmap_path}")
    print(f"Device: {args.device}")
    
    preprocess(bitmap_config, bitmap_path, args.device)