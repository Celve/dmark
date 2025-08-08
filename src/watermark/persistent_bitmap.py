import os
from typing import Optional

import torch
import numpy as np


class PersistentBitmap:
    def __init__(self, vocab_size: int, filepath: str):
        self.vocab_size = vocab_size
        self.filepath = filepath

        if os.path.exists(filepath):
            self._load()
        else:
            self._initialize()

    def _initialize(self):
        self.matrix = torch.zeros((self.vocab_size, self.vocab_size), dtype=torch.bool)
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
        self.matrix = torch.from_numpy(unpacked_bits).bool().reshape(self.vocab_size, self.vocab_size).to("cuda")

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
        return self.matrix[x].clone()
    
    def get_rows(self, indices: torch.Tensor) -> torch.Tensor:
        """Efficiently get multiple rows using tensor indexing.
        
        Args:
            indices: 1D tensor of row indices
            
        Returns:
            2D tensor of shape (len(indices), vocab_size)
        """
        return self.matrix[indices]
    
    def get_col(self, y: int) -> torch.Tensor:
        if y >= self.vocab_size:
            raise IndexError(f"Token index out of bounds: vocab_size={self.vocab_size}")
        return self.matrix[:, y].clone()
    
    def get_cols(self, indices: torch.Tensor) -> torch.Tensor:
        """Efficiently get multiple columns using tensor indexing.
        
        Args:
            indices: 1D tensor of column indices
            
        Returns:
            2D tensor of shape (vocab_size, len(indices))
        """
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
        
        self.matrix[x] = values.bool()
    
    def transpose(self):
        """Transpose the bitmap in-place, swapping x and y dimensions."""
        self.matrix = self.matrix.T
    
    def to(self, device):
        """Move the matrix to a specific device (CPU/GPU)."""
        self.matrix = self.matrix.to(device)
        return self
