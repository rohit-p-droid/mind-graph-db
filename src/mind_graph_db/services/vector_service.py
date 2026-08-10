"""Vector service orchestrating embedding generation and vector store operations."""

from typing import Any, Dict, List, Optional, Tuple

from mind_graph_db.core.types import Document, Vector
from mind_graph_db.interfaces.embedding import EmbeddingModel
from mind_graph_db.interfaces.vector_store import VectorStore


class VectorService:
    """Service connecting Document text, EmbeddingModel generation, and VectorStore persistence."""

    def __init__(self, embedding_model: EmbeddingModel, vector_store: VectorStore) -> None:
        """Initialize VectorService with an embedding model and vector store.

        Args:
            embedding_model: Instance implementing EmbeddingModel interface.
            vector_store: Instance implementing VectorStore interface.
        """
        self.embedding_model = embedding_model
        self.vector_store = vector_store

    def embed_and_store_document(self, document: Document) -> Vector:
        """Generate a vector embedding for a document and store it separately in VectorStore.

        Args:
            document: Document domain model instance.

        Returns:
            The created Vector object.
        """
        embedding = self.embedding_model.encode_text(document.text)
        metadata = dict(document.metadata)
        metadata["doc_id"] = document.id

        self.vector_store.add(
            vector_id=document.id,
            embedding=embedding,
            metadata=metadata,
        )

        return Vector(id=document.id, embedding=embedding, metadata=metadata)

    def search_similar_documents(
        self,
        query_text: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, float]]:
        """Encode query text and perform similarity search against stored document vectors.

        Args:
            query_text: Raw query text string.
            top_k: Maximum number of matching document IDs to return.
            filters: Optional metadata filtering criteria.

        Returns:
            List of tuples containing (document_id, similarity_score).
        """
        query_embedding = self.embedding_model.encode_text(query_text)
        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            filters=filters,
        )

        return [(vector.id, score) for vector, score in results]
