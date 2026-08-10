"""Unit and integration tests for MindQueryEngine query execution."""

from pathlib import Path

from mind_graph_db.core.types import Document, QueryResult
from mind_graph_db.embeddings import HashEmbeddingModel
from mind_graph_db.nlp import RuleBasedNLPModel
from mind_graph_db.pipeline import DocumentIngestionPipeline
from mind_graph_db.query import MindQueryEngine
from mind_graph_db.storage import SQLiteDocumentStore, SQLiteGraphStore, SQLiteVectorStore


def test_mind_query_engine_execution(tmp_path: Path) -> None:
    doc_store = SQLiteDocumentStore(db_path=str(tmp_path / "docs.db"))
    vec_store = SQLiteVectorStore(db_path=str(tmp_path / "vecs.db"))
    graph_store = SQLiteGraphStore(db_path=str(tmp_path / "graph.db"))
    embedding_model = HashEmbeddingModel(dim=128)
    nlp_model = RuleBasedNLPModel()

    pipeline = DocumentIngestionPipeline(
        document_store=doc_store,
        vector_store=vec_store,
        graph_store=graph_store,
        embedding_model=embedding_model,
        nlp_model=nlp_model,
        min_confidence=0.5,
    )

    doc1 = Document(
        id="qdoc-1",
        text="Python programming language is used for artificial intelligence.",
        metadata={"category": "tech", "author": "Alice"},
    )
    doc2 = Document(
        id="qdoc-2",
        text="SQLite database provides fast embedded graph storage in Python.",
        metadata={"category": "db", "author": "Bob"},
    )

    pipeline.ingest_document(doc1)
    pipeline.ingest_document(doc2)

    engine = MindQueryEngine(
        document_store=doc_store,
        vector_store=vec_store,
        graph_store=graph_store,
        embedding_model=embedding_model,
    )

    query_str = (
        'FIND documents '
        'WHERE semantic_match("Python AI") AND metadata.category = "tech" '
        'TRAVERSE 2 HOPS '
        'RETURN documents, entities, relationships'
    )

    result = engine.query(query_str, top_k=5)

    assert isinstance(result, QueryResult)
    assert len(result.documents) >= 1
    assert result.documents[0].id == "qdoc-1"
    assert result.documents[0].metadata["author"] == "Alice"
    assert len(result.entities) >= 1
    assert len(result.scores) >= 1

    doc_store.close()
    vec_store.close()
    graph_store.close()
