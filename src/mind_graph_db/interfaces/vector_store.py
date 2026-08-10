"""VectorStore abstract base class."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from mind_graph_db.core.types import Vector


class VectorStore(ABC):
    """Abstract interface for storing and searching high-dimensional vector embeddings."""

    @abstractmethod
    def add(
        self,
        vector_id: str,
        embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add or update a vector embedding.

        Args:
            vector_id: Unique identifier for the vector.
            embedding: High-dimensional numerical vector representation.
            metadata: Associated metadata key-value pairs.
        """
        pass

    @abstractmethod
    def get(self, vector_id: str) -> Optional[Vector]:
        """Retrieve a vector by ID.

        Args:
            vector_id: Unique identifier for the vector.

        Returns:
            The Vector object if found, otherwise None.
        """
        pass

    @abstractmethod
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Vector, float]]:
        """Search for top_k most similar vectors.

        Args:
            query_embedding: The search query embedding vector.
            top_k: Number of nearest neighbors to retrieve.
            filters: Optional metadata filtering criteria.

        Returns:
            List of tuples containing (Vector, similarity_score).
        """
        pass

    @abstractmethod
    def delete(self, vector_id: str) -> bool:
        """Delete a vector embedding by ID.

        Args:
            vector_id: Unique identifier for the vector.

        Returns:
            True if deleted, False if not found.
        """
        pass

    @abstractmethod
    def count(self) -> int:
        """Return the total number of stored vectors."""
        pass
