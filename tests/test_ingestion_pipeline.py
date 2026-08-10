"""Integration tests for DocumentIngestionPipeline automatic graph discovery."""

from pathlib import Path

from mind_graph_db.core.types import Document
from mind_graph_db.embeddings import HashEmbeddingModel
from mind_graph_db.nlp import RuleBasedNLPModel
from mind_graph_db.pipeline import DocumentIngestionPipeline
from mind_graph_db.storage import SQLiteDocumentStore, SQLiteGraphStore, SQLiteVectorStore


def test_document_ingestion_pipeline_integration(tmp_path: Path) -> None:
    doc_store_path = str(tmp_path / "docs.db")
    vec_store_path = str(tmp_path / "vecs.db")
    graph_store_path = str(tmp_path / "graph.db")

    doc_store = SQLiteDocumentStore(db_path=doc_store_path)
    vec_store = SQLiteVectorStore(db_path=vec_store_path)
    graph_store = SQLiteGraphStore(db_path=graph_store_path)
    embedding_model = HashEmbeddingModel(dim=128)
    nlp_model = RuleBasedNLPModel()

    pipeline = DocumentIngestionPipeline(
        document_store=doc_store,
        vector_store=vec_store,
        graph_store=graph_store,
        embedding_model=embedding_model,
        nlp_model=nlp_model,
        min_confidence=0.5,
        candidate_top_k=5,
    )

    doc1 = Document(
        id="doc-1",
        text="Python is a high-level programming language used by Google.",
        metadata={"author": "Alice"},
    )
    doc2 = Document(
        id="doc-2",
        text="Python integrates with SQLite to manage databases. OpenAI uses Python.",
        metadata={"author": "Bob"},
    )

    # Ingest Document 1
    res1 = pipeline.ingest_document(doc1)
    assert res1.document_id == "doc-1"
    assert res1.entities_extracted >= 1
    assert doc_store.count() == 1
    assert vec_store.count() == 1

    # Ingest Document 2 (triggers candidate discovery & cross-document link)
    res2 = pipeline.ingest_document(doc2)
    assert res2.document_id == "doc-2"
    assert res2.candidate_docs_evaluated >= 1
    assert doc_store.count() == 2
    assert vec_store.count() == 2

    # Verify Document nodes in GraphStore
    node_doc1 = graph_store.get_node("doc-1")
    node_doc2 = graph_store.get_node("doc-2")
    assert node_doc1 is not None
    assert node_doc1.node_type == "DOCUMENT"
    assert node_doc2 is not None
    assert node_doc2.node_type == "DOCUMENT"

    # Verify MENTIONS edges from doc-1
    neighbors_doc1 = graph_store.get_neighbors("doc-1")
    assert len(neighbors_doc1) >= 1
    evidence_texts = [edge.evidence_text for _, edge in neighbors_doc1 if edge.evidence_text]
    assert len(evidence_texts) > 0

    # Verify cross-document vector similarity or entity resolution links
    subgraph = graph_store.query_subgraph(["doc-1", "doc-2"], depth=2)
    assert len(subgraph["nodes"]) >= 3
    assert len(subgraph["relationships"]) >= 2

    doc_store.close()
    vec_store.close()
    graph_store.close()
