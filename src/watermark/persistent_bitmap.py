import os
from typing import Optional

import numpy as np


class PersistentBitmap:
    def __init__(self, vocab_size: int, filepath: str):
        self.vocab_size = vocab_size
        self.filepath = filepath
        self.total_bits = vocab_size * vocab_size
        self.bytes_needed = (self.total_bits + 7) // 8

        if os.path.exists(filepath):
            self._load()
        else:
            self._initialize()

    def _initialize(self):
        self.bitmap = np.zeros(self.bytes_needed, dtype=np.uint8)
        self._save()

    def _load(self):
        self.bitmap = np.fromfile(self.filepath, dtype=np.uint8)
        if len(self.bitmap) != self.bytes_needed:
            raise ValueError(
                f"Bitmap file size mismatch: expected {self.bytes_needed} bytes, got {len(self.bitmap)}"
            )

    def _save(self):
        self.bitmap.tofile(self.filepath)

    def set_bit(self, x: int, y: int, value: bool):
        if x >= self.vocab_size or y >= self.vocab_size:
            raise IndexError(f"Token index out of bounds: vocab_size={self.vocab_size}")

        bit_index = x * self.vocab_size + y
        byte_index = bit_index // 8
        bit_offset = bit_index % 8

        if value:
            self.bitmap[byte_index] |= 1 << bit_offset
        else:
            self.bitmap[byte_index] &= ~(1 << bit_offset)

    def get_bit(self, x: int, y: int) -> bool:
        if x >= self.vocab_size or y >= self.vocab_size:
            raise IndexError(f"Token index out of bounds: vocab_size={self.vocab_size}")

        bit_index = x * self.vocab_size + y
        byte_index = bit_index // 8
        bit_offset = bit_index % 8

        return bool(self.bitmap[byte_index] & (1 << bit_offset))

    def save(self):
        self._save()

    def get_row(self, x: int) -> np.ndarray:
        if x >= self.vocab_size:
            raise IndexError(f"Token index out of bounds: vocab_size={self.vocab_size}")

        result = np.zeros(self.vocab_size, dtype=bool)
        for i in range(self.vocab_size):
            result[i] = self.get_bit(x, i)
        return result

    def set_row(self, x: int, values: np.ndarray):
        if x >= self.vocab_size:
            raise IndexError(f"Token index out of bounds: vocab_size={self.vocab_size}")
        if len(values) != self.vocab_size:
            raise ValueError(f"Values array must have length {self.vocab_size}")

        for i in range(self.vocab_size):
            self.set_bit(x, i, bool(values[i]))
