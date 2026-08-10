"""Abstract interfaces defining Mind Graph DB extension points."""

from mind_graph_db.interfaces.document_store import DocumentStore
from mind_graph_db.interfaces.embedding import EmbeddingModel
from mind_graph_db.interfaces.graph_store import GraphStore
from mind_graph_db.interfaces.nlp import NLPModel
from mind_graph_db.interfaces.query_engine import QueryEngine
from mind_graph_db.interfaces.vector_store import VectorStore

__all__ = [
    "DocumentStore",
    "VectorStore",
    "GraphStore",
    "EmbeddingModel",
    "NLPModel",
    "QueryEngine",
]
