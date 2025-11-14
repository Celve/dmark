from abc import ABC
from typing import Any


class Dataset(ABC):
    def sample(self, index: int) -> dict[str, Any]:
        """Return a sample from the dataset as a string."""
        raise NotImplementedError