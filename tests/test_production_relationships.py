"""Relationship semantics, evidence provenance, and contradiction preservation tests."""

from pathlib import Path
from mind_graph_db.core.types import Document, ProvenanceRecord, RelationshipKind
from mind_graph_db.embeddings import HashEmbeddingModel
from mind_graph_db.nlp import RuleBasedNLPModel
from mind_graph_db.pipeline import DocumentIngestionPipeline
from mind_graph_db.sdk import MindGraphDBClient
from mind_graph_db.storage import SQLiteDocumentStore, SQLiteGraphStore, SQLiteVectorStore


def test_provenance_integrity_and_character_spans(tmp_path: Path) -> None:
    doc_store = SQLiteDocumentStore(db_path=str(tmp_path / "prov_docs.db"))
    vec_store = SQLiteVectorStore(db_path=str(tmp_path / "prov_vecs.db"))
    graph_store = SQLiteGraphStore(db_path=str(tmp_path / "prov_graph.db"))
    pipeline = DocumentIngestionPipeline(
        document_store=doc_store,
        vector_store=vec_store,
        graph_store=graph_store,
        embedding_model=HashEmbeddingModel(dim=32),
        nlp_model=RuleBasedNLPModel(),
    )

    doc = Document(
        id="doc-prov-1",
        text="Natural language processing enables computers to process human language using machine learning.",
    )
    pipeline.ingest_document(doc)

    subgraph = graph_store.query_subgraph([], depth=10)
    relationships = subgraph["relationships"]

    mentions_rels = [r for r in relationships if r.relation_type == "MENTIONS"]
    assert len(mentions_rels) >= 1

    for rel in mentions_rels:
        assert rel.confidence == 1.0
        assert rel.provenance is not None
        assert rel.provenance.kind == RelationshipKind.EXPLICIT
        assert rel.provenance.source_doc_id == "doc-prov-1"
        assert rel.provenance.char_span is not None
        start, end = rel.provenance.char_span
        assert 0 <= start < end <= len(doc.text)

    doc_store.close()
    vec_store.close()
    graph_store.close()


def test_symmetric_relationship_ordering_uniqueness(tmp_path: Path) -> None:
    client = MindGraphDBClient(db_dir=str(tmp_path / "sym_db"))
    client.delete_all()

    # Ingest document generating co-occurrence
    client.create_document(
        doc_id="sym-doc-1",
        text="Deep learning uses neural networks for natural language processing.",
    )

    relationships = client.get_relationships()
    co_occur_rels = [r for r in relationships if r.relation_type == "CO_OCCURS_WITH"]
    
    # Assert every symmetric pair appears exactly once (no reverse-direction duplicates)
    pairs = [(r.source_id, r.target_id) for r in co_occur_rels]
    assert len(pairs) == len(set(pairs))
    for src, tgt in pairs:
        assert src <= tgt  # Canonical ordering min, max
