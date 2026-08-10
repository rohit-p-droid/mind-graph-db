"""Deterministic lightweight feature hash embedding model for testing and offline environments."""

import hashlib
import math
from typing import List

from mind_graph_db.interfaces.embedding import EmbeddingModel


class HashEmbeddingModel(EmbeddingModel):
    """Zero-dependency deterministic hashing embedding model."""

    def __init__(self, dim: int = 384) -> None:
        """Initialize HashEmbeddingModel with output vector dimension.

        Args:
            dim: Dimension size of the output vector (default: 384).
        """
        self._dim = dim

    @property
    def dimension(self) -> int:
        return self._dim

    def encode_text(self, text: str) -> List[float]:
        """Generate a normalized feature hash vector for a text string."""
        if not text:
            return [0.0] * self._dim

        words = text.lower().split()
        vector = [0.0] * self._dim

        for word in words:
            # Generate deterministic index and sign from word hash
            digest = hashlib.sha256(word.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "big") % self._dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[idx] += sign

        # Compute L2 norm and normalize vector
        norm = math.sqrt(sum(v * v for v in vector))
        if norm > 0.0:
            vector = [v / norm for v in vector]
        else:
            # Fallback uniform vector if all zeros
            val = 1.0 / math.sqrt(self._dim)
            vector = [val] * self._dim

        return vector

    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate normalized feature hash vectors for a batch of texts."""
        return [self.encode_text(text) for text in texts]
