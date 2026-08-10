"""EmbeddingModel abstract base class."""

from abc import ABC, abstractmethod
from typing import List


class EmbeddingModel(ABC):
    """Abstract interface for text embedding generation models."""

    @abstractmethod
    def encode_text(self, text: str) -> List[float]:
        """Generate a vector embedding for a single text string.

        Args:
            text: Raw input text string.

        Returns:
            List of floats representing the numerical embedding vector.
        """
        pass

    @abstractmethod
    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate vector embeddings for a batch of text strings.

        Args:
            texts: List of input text strings.

        Returns:
            List of embedding vectors.
        """
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the output dimension size of the embedding model."""
        pass
