"""Production ingestion hardening, batch ingestion, and concurrency idempotency tests."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from mind_graph_db.core.types import Document
from mind_graph_db.embeddings import HashEmbeddingModel
from mind_graph_db.nlp import RuleBasedNLPModel
from mind_graph_db.pipeline import DocumentIngestionPipeline
from mind_graph_db.sdk import MindGraphDBClient
from mind_graph_db.storage import SQLiteDocumentStore, SQLiteGraphStore, SQLiteVectorStore
from mind_graph_db.testing import SyntheticDataGenerator


def test_single_and_batch_ingestion(tmp_path: Path) -> None:
    doc_store = SQLiteDocumentStore(db_path=str(tmp_path / "docs.db"))
    vec_store = SQLiteVectorStore(db_path=str(tmp_path / "vecs.db"))
    graph_store = SQLiteGraphStore(db_path=str(tmp_path / "graph.db"))
    pipeline = DocumentIngestionPipeline(
        document_store=doc_store,
        vector_store=vec_store,
        graph_store=graph_store,
        embedding_model=HashEmbeddingModel(dim=32),
        nlp_model=RuleBasedNLPModel(),
    )

    generator = SyntheticDataGenerator(seed=42)
    corpus = generator.generate_corpus(profile="quick")

    # Ingest batch of 100 synthetic documents
    for gt_doc in corpus.documents:
        doc = Document(id=gt_doc.id, text=gt_doc.text, metadata=gt_doc.metadata)
        pipeline.ingest_document(doc)

    subgraph = graph_store.query_subgraph([], depth=100)
    assert len(subgraph["nodes"]) > 0
    assert len(subgraph["relationships"]) > 0

    doc_store.close()
    vec_store.close()
    graph_store.close()


def test_repeated_ingestion_idempotency_100_percent(tmp_path: Path) -> None:
    client = MindGraphDBClient(db_dir=str(tmp_path / "idempotent_db"))
    client.delete_all()

    generator = SyntheticDataGenerator(seed=100)
    corpus = generator.generate_corpus(profile="quick")

    # Pass 1: Ingest 100 documents
    for gt_doc in corpus.documents:
        client.create_document(doc_id=gt_doc.id, text=gt_doc.text, metadata=gt_doc.metadata)

    # Pass 2: Vector store fully aligned
    for gt_doc in corpus.documents:
        client.create_document(doc_id=gt_doc.id, text=gt_doc.text, metadata=gt_doc.metadata)

    subgraph2 = client.get_graph()
    nodes_pass2 = len(subgraph2["nodes"])
    rels_pass2 = len(subgraph2["relationships"])

    # Pass 3: Re-ingest exact same 100 documents
    for gt_doc in corpus.documents:
        client.create_document(doc_id=gt_doc.id, text=gt_doc.text, metadata=gt_doc.metadata)

    subgraph3 = client.get_graph()
    nodes_pass3 = len(subgraph3["nodes"])
    rels_pass3 = len(subgraph3["relationships"])

    # Assert 100% strict equality across repeated ingestions
    assert nodes_pass3 == nodes_pass2, f"Node count changed from {nodes_pass2} to {nodes_pass3}"
    assert rels_pass3 == rels_pass2, f"Relationship count changed from {rels_pass2} to {rels_pass3}"


def test_concurrent_multi_worker_ingestion(tmp_path: Path) -> None:
    doc_store = SQLiteDocumentStore(db_path=str(tmp_path / "concurrent_docs.db"))
    vec_store = SQLiteVectorStore(db_path=str(tmp_path / "concurrent_vecs.db"))
    graph_store = SQLiteGraphStore(db_path=str(tmp_path / "concurrent_graph.db"))
    pipeline = DocumentIngestionPipeline(
        document_store=doc_store,
        vector_store=vec_store,
        graph_store=graph_store,
        embedding_model=HashEmbeddingModel(dim=32),
        nlp_model=RuleBasedNLPModel(),
    )

    generator = SyntheticDataGenerator(seed=200)
    corpus = generator.generate_corpus(profile="quick")

    def ingest_worker(gt_doc) -> None:
        doc = Document(id=gt_doc.id, text=gt_doc.text, metadata=gt_doc.metadata)
        pipeline.ingest_document(doc)

    # Execute concurrent ingestion across 4 worker threads
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(ingest_worker, corpus.documents))

    subgraph = graph_store.query_subgraph([], depth=100)
    assert len(subgraph["nodes"]) > 0
    assert len(subgraph["relationships"]) > 0

    doc_store.close()
    vec_store.close()
    graph_store.close()
