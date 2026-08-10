"""Embedding models and factory functions for Mind Graph DB."""

from typing import Optional

from mind_graph_db.config.settings import get_settings
from mind_graph_db.embeddings.hash_embedding import HashEmbeddingModel
from mind_graph_db.embeddings.sentence_transformer import SentenceTransformerEmbeddingModel
from mind_graph_db.interfaces.embedding import EmbeddingModel


def get_embedding_model(
    model_name: Optional[str] = None,
    fallback_to_hash: bool = True,
) -> EmbeddingModel:
    """Factory function to retrieve an EmbeddingModel instance.

    Attempts to instantiate SentenceTransformerEmbeddingModel using model_name (or Settings).
    If sentence-transformers is not available and fallback_to_hash is True, returns HashEmbeddingModel.

    Args:
        model_name: Pretrained model name or identifier.
        fallback_to_hash: Whether to fall back to HashEmbeddingModel if sentence-transformers is missing.

    Returns:
        Instance implementing EmbeddingModel interface.
    """
    settings = get_settings()
    target_name = model_name or settings.embedding_model_name

    try:
        return SentenceTransformerEmbeddingModel(model_name=target_name)
    except ImportError:
        if fallback_to_hash:
            return HashEmbeddingModel(dim=settings.embedding_dimension)
        raise


__all__ = [
    "EmbeddingModel",
    "HashEmbeddingModel",
    "SentenceTransformerEmbeddingModel",
    "get_embedding_model",
]
