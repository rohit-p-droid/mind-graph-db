"""Unit and integration tests for NLP entity extraction, normalization, explicit MENTIONS, and idempotency."""

from pathlib import Path
from mind_graph_db.core.types import Document
from mind_graph_db.embeddings import HashEmbeddingModel
from mind_graph_db.nlp import RuleBasedNLPModel
from mind_graph_db.pipeline import DocumentIngestionPipeline
from mind_graph_db.storage import SQLiteDocumentStore, SQLiteGraphStore, SQLiteVectorStore


def test_multi_word_entity_extraction_and_normalization() -> None:
    nlp = RuleBasedNLPModel()

    text1 = "Machine learning is a branch of artificial intelligence that enables computers to learn from data."
    entities1 = [e.name for e in nlp.extract_entities(text1)]
    assert "Machine Learning" in entities1
    assert "Artificial Intelligence" in entities1
    assert "Machine" not in entities1  # Fragment suppressed

    text2 = "Deep learning uses neural networks for natural language processing techniques and computer vision."
    entities2 = [e.name for e in nlp.extract_entities(text2)]
    assert "Deep Learning" in entities2
    assert "Natural Language Processing" in entities2  # Normalized from Natural Language Processing Techniques
    assert "Computer Vision" in entities2
    assert "Neural Networks" in entities2

    text3 = "Vector databases store embeddings efficiently."
    entities3 = [e.name for e in nlp.extract_entities(text3)]
    assert "Vector Databases" in entities3
    assert "Embeddings" in entities3
    assert "Vector" not in entities3  # Fragment suppressed


def test_explicit_mentions_confidence_and_provenance(tmp_path: Path) -> None:
    db_dir = str(tmp_path / "explicit_mentions_db")
    doc_store = SQLiteDocumentStore(db_path=str(tmp_path / "explicit_mentions_db" / "documents.db"))
    vec_store = SQLiteVectorStore(db_path=str(tmp_path / "explicit_mentions_db" / "vectors.db"))
    graph_store = SQLiteGraphStore(db_path=str(tmp_path / "explicit_mentions_db" / "graph.db"))
    pipeline = DocumentIngestionPipeline(
        document_store=doc_store,
        vector_store=vec_store,
        graph_store=graph_store,
        embedding_model=HashEmbeddingModel(dim=64),
        nlp_model=RuleBasedNLPModel(),
        min_confidence=0.5,
    )

    doc = Document(
        id="doc1",
        text="Machine learning is a branch of artificial intelligence that enables computers to learn from data.",
        metadata={"category": "Machine Learning"},
    )
    pipeline.ingest_document(doc)

    subgraph = graph_store.query_subgraph([], depth=5)
    mentions_rels = [r for r in subgraph["relationships"] if r.relation_type == "MENTIONS"]
    assert len(mentions_rels) >= 1

    for rel in mentions_rels:
        # All MENTIONS relationships MUST have confidence 1.0
        assert rel.confidence == 1.0, f"Expected MENTIONS confidence 1.0, got {rel.confidence}"
        # Evidence text must quote the supporting sentence
        assert "sentence:" in rel.evidence_text
        assert "Machine learning is a branch of artificial intelligence" in rel.evidence_text

    doc_store.close()
    vec_store.close()
    graph_store.close()


def test_repeated_document_ingestion_idempotency(tmp_path: Path) -> None:
    db_dir = str(tmp_path / "idempotent_db")
    doc_store = SQLiteDocumentStore(db_path=str(tmp_path / "idempotent_db" / "documents.db"))
    vec_store = SQLiteVectorStore(db_path=str(tmp_path / "idempotent_db" / "vectors.db"))
    graph_store = SQLiteGraphStore(db_path=str(tmp_path / "idempotent_db" / "graph.db"))
    pipeline = DocumentIngestionPipeline(
        document_store=doc_store,
        vector_store=vec_store,
        graph_store=graph_store,
        embedding_model=HashEmbeddingModel(dim=64),
        nlp_model=RuleBasedNLPModel(),
        min_confidence=0.5,
    )

    docs = [
        Document(
            id="doc1",
            text="Machine learning is a branch of artificial intelligence that enables computers to learn from data.",
            metadata={"category": "Machine Learning"},
        ),
        Document(
            id="doc2",
            text="Deep learning is a subset of machine learning that uses neural networks.",
            metadata={"category": "Deep Learning"},
        ),
        Document(
            id="doc3",
            text="Natural language processing enables computers to process human language using machine learning.",
            metadata={"category": "NLP"},
        ),
        Document(
            id="doc4",
            text="Computer vision allows machines to analyze visual information.",
            metadata={"category": "Computer Vision"},
        ),
        Document(
            id="doc5",
            text="Vector databases store numerical representations of data using embeddings.",
            metadata={"category": "Databases"},
        ),
    ]

    # Ingest 5 documents 3 times
    for _ in range(3):
        for doc in docs:
            pipeline.ingest_document(doc)

    subgraph = graph_store.query_subgraph([], depth=5)
    relationships = subgraph["relationships"]

    # Verify all MENTIONS edges have confidence 1.0 (no vector similarity MENTIONS)
    mentions_rels = [r for r in relationships if r.relation_type == "MENTIONS"]
    for rel in mentions_rels:
        assert rel.confidence == 1.0, f"Found non-explicit MENTIONS edge with confidence {rel.confidence}"

    # Verify no duplicate relationships exist
    rel_tuples = [(r.source_id, r.target_id, r.relation_type) for r in relationships]
    assert len(rel_tuples) == len(set(rel_tuples)), "Duplicate relationships found in graph"

    doc_store.close()
    vec_store.close()
    graph_store.close()
