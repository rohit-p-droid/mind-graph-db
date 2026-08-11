"""High-stress performance benchmarking and chaos failure recovery tests."""

import time
from pathlib import Path
import pytest
from mind_graph_db.core.types import Document
from mind_graph_db.embeddings import HashEmbeddingModel
from mind_graph_db.nlp import RuleBasedNLPModel
from mind_graph_db.pipeline import DocumentIngestionPipeline
from mind_graph_db.sdk import MindGraphDBClient
from mind_graph_db.storage import SQLiteDocumentStore, SQLiteGraphStore, SQLiteVectorStore
from mind_graph_db.testing import SyntheticDataGenerator


def test_stress_ingestion_throughput_and_latency(tmp_path: Path) -> None:
    doc_store = SQLiteDocumentStore(db_path=str(tmp_path / "stress_docs.db"))
    vec_store = SQLiteVectorStore(db_path=str(tmp_path / "stress_vecs.db"))
    graph_store = SQLiteGraphStore(db_path=str(tmp_path / "stress_graph.db"))
    pipeline = DocumentIngestionPipeline(
        document_store=doc_store,
        vector_store=vec_store,
        graph_store=graph_store,
        embedding_model=HashEmbeddingModel(dim=32),
        nlp_model=RuleBasedNLPModel(),
    )

    generator = SyntheticDataGenerator(seed=500)
    corpus = generator.generate_corpus(profile="quick")  # 100 documents

    start_time = time.time()
    for gt_doc in corpus.documents:
        doc = Document(id=gt_doc.id, text=gt_doc.text, metadata=gt_doc.metadata)
        pipeline.ingest_document(doc)
    elapsed = time.time() - start_time

    throughput = len(corpus.documents) / max(0.001, elapsed)
    assert throughput > 10.0, f"Ingestion throughput {throughput:.1f} docs/sec below threshold 10.0"

    doc_store.close()
    vec_store.close()
    graph_store.close()


def test_chaos_empty_and_oversized_documents(tmp_path: Path) -> None:
    client = MindGraphDBClient(db_dir=str(tmp_path / "chaos_db"))
    client.delete_all()

    # Empty text raises ValueError
    with pytest.raises(ValueError, match="cannot be empty"):
        client.create_document(doc_id="empty-doc", text="   ")

    # Extremely long document text
    long_text = "Machine learning is artificial intelligence. " * 500
    res = client.create_document(doc_id="long-doc", text=long_text)
    assert res.document_id == "long-doc"
    assert res.entities_extracted >= 1
