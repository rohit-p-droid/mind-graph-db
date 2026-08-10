"""Dependency injection providers for FastAPI web app."""

import os
from typing import Optional

from mind_graph_db.embeddings import get_embedding_model
from mind_graph_db.interfaces.document_store import DocumentStore
from mind_graph_db.interfaces.embedding import EmbeddingModel
from mind_graph_db.interfaces.graph_store import GraphStore
from mind_graph_db.interfaces.nlp import NLPModel
from mind_graph_db.interfaces.query_engine import QueryEngine
from mind_graph_db.interfaces.vector_store import VectorStore
from mind_graph_db.nlp import get_nlp_model
from mind_graph_db.pipeline import DocumentIngestionPipeline
from mind_graph_db.query import MindQueryEngine
from mind_graph_db.storage import SQLiteDocumentStore, SQLiteGraphStore, SQLiteVectorStore


class DatabaseContainer:
    """Container holding database storage, model, pipeline, and query components."""

    def __init__(
        self,
        document_store: DocumentStore,
        vector_store: VectorStore,
        graph_store: GraphStore,
        embedding_model: EmbeddingModel,
        nlp_model: NLPModel,
    ) -> None:
        self.document_store = document_store
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.embedding_model = embedding_model
        self.nlp_model = nlp_model
        self.pipeline = DocumentIngestionPipeline(
            document_store=document_store,
            vector_store=vector_store,
            graph_store=graph_store,
            embedding_model=embedding_model,
            nlp_model=nlp_model,
        )
        self.query_engine = MindQueryEngine(
            document_store=document_store,
            vector_store=vector_store,
            graph_store=graph_store,
            embedding_model=embedding_model,
        )


_container_instance: Optional[DatabaseContainer] = None


def init_database(db_dir: str = "./storage") -> DatabaseContainer:
    """Initialize database components and set global singleton container."""
    global _container_instance
    os.makedirs(db_dir, exist_ok=True)

    doc_store = SQLiteDocumentStore(db_path=os.path.join(db_dir, "documents.db"))
    vec_store = SQLiteVectorStore(db_path=os.path.join(db_dir, "vectors.db"))
    graph_store = SQLiteGraphStore(db_path=os.path.join(db_dir, "graph.db"))
    embedding_model = get_embedding_model()
    nlp_model = get_nlp_model()

    _container_instance = DatabaseContainer(
        document_store=doc_store,
        vector_store=vec_store,
        graph_store=graph_store,
        embedding_model=embedding_model,
        nlp_model=nlp_model,
    )
    return _container_instance


def get_database() -> DatabaseContainer:
    """FastAPI dependency yielding current database container."""
    global _container_instance
    if _container_instance is None:
        _container_instance = init_database()
    return _container_instance
