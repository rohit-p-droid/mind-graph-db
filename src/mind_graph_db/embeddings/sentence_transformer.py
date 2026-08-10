"""SentenceTransformers pretrained open-source embedding model implementation."""

from typing import List, Optional

from mind_graph_db.config.settings import get_settings
from mind_graph_db.interfaces.embedding import EmbeddingModel


class SentenceTransformerEmbeddingModel(EmbeddingModel):
    """Embedding model using HuggingFace / SentenceTransformers pretrained models."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        """Initialize SentenceTransformer embedding model.

        Args:
            model_name: Name of sentence-transformer model (e.g. 'all-MiniLM-L6-v2'). If None, uses Settings.
            device: Computing device ('cpu', 'cuda', etc.).
        """
        try:
            from sentence_transformers import SentenceTransformer


        except ImportError as err:
            raise ImportError(
                "sentence-transformers library is not installed. "
                "Please install it using `pip install sentence-transformers`."
            ) from err

        if model_name is None:
            settings = get_settings()
            model_name = settings.embedding_model_name

        self.model_name = model_name
        self.device = device
        self._model = SentenceTransformer(self.model_name, device=self.device)
        self._dimension: int = int(self._model.get_sentence_embedding_dimension())

    @property
    def dimension(self) -> int:
        return self._dimension

    def encode_text(self, text: str) -> List[float]:
        """Encode a single text string into a float vector."""
        embedding = self._model.encode(text, normalize_embeddings=True)
        return [float(x) for x in embedding]

    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        """Encode a list of text strings into float vectors."""
        embeddings = self._model.encode(texts, normalize_embeddings=True)
        return [[float(x) for x in vec] for vec in embeddings]
