"""Unit and integration tests for document update reconciliation and stale edge pruning."""

from pathlib import Path
from time import sleep

from mind_graph_db.core.types import Document
from mind_graph_db.embeddings import HashEmbeddingModel
from mind_graph_db.nlp import RuleBasedNLPModel
from mind_graph_db.pipeline import DocumentIngestionPipeline
from mind_graph_db.storage import SQLiteDocumentStore, SQLiteGraphStore, SQLiteVectorStore


def test_document_update_reconciliation(tmp_path: Path) -> None:
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
    )

    doc = Document(
        id="doc-update-1",
        text="Python is used by Google for backend services.",
        metadata={"version": 1},
    )

    # Initial ingestion
    res_ingest = pipeline.ingest_document(doc)
    assert res_ingest.document_id == "doc-update-1"
    assert res_ingest.entities_extracted >= 2

    initial_neighbors = graph_store.get_neighbors("doc-update-1")
    initial_labels = [node.label for node, _ in initial_neighbors]
    assert "Google" in initial_labels
    assert "Python" in initial_labels

    created_at = doc.created_at

    # Small pause to guarantee timestamp difference
    sleep(0.01)

    # Update document: Replace Google with OpenAI
    doc.text = "Python is used by OpenAI to build artificial intelligence models."
    doc.metadata["version"] = 2

    res_update = pipeline.update_document(doc)

    assert res_update.document_id == "doc-update-1"
    assert res_update.relationships_removed >= 1  # Google edge removed
    assert res_update.relationships_created >= 1  # OpenAI edge added
    assert res_update.relationships_updated >= 1  # Python edge updated

    # Verify DocumentStore update
    retrieved_doc = doc_store.get("doc-update-1")
    assert retrieved_doc is not None
    assert retrieved_doc.text == "Python is used by OpenAI to build artificial intelligence models."
    assert retrieved_doc.updated_at > created_at

    # Verify graph node and edge states
    updated_neighbors = graph_store.get_neighbors("doc-update-1")
    updated_labels = [node.label for node, _ in updated_neighbors]
    assert "OpenAI" in updated_labels
    assert "Python" in updated_labels
    assert "Google" not in updated_labels  # Stale relationship successfully pruned

    doc_store.close()
    vec_store.close()
    graph_store.close()
