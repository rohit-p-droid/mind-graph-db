"""Unit tests for ProvenanceRecord, GraphHealthChecker diagnostics, orphan entity pruning, and LLMNLPModel."""

from pathlib import Path
import pytest
from mind_graph_db.core.types import (
    Document,
    Entity,
    ProvenanceRecord,
    RelationshipCandidate,
    RelationshipKind,
    SemanticAnalysisResult,
)
from mind_graph_db.embeddings import HashEmbeddingModel
from mind_graph_db.nlp import LLMNLPModel, RuleBasedNLPModel
from mind_graph_db.pipeline import DocumentIngestionPipeline
from mind_graph_db.sdk import MindGraphDBClient
from mind_graph_db.storage import SQLiteDocumentStore, SQLiteGraphStore, SQLiteVectorStore
from mind_graph_db.utils.health import GraphHealthChecker


def test_provenance_record_creation() -> None:
    prov = ProvenanceRecord(
        kind=RelationshipKind.EXPLICIT,
        source_doc_id="doc1",
        evidence_text="Machine learning is artificial intelligence.",
        char_span=(0, 16),
        confidence=1.0,
        method="nlp_explicit_extraction",
    )

    assert prov.kind == RelationshipKind.EXPLICIT
    assert prov.confidence == 1.0
    assert prov.char_span == (0, 16)
    assert prov.source_doc_id == "doc1"


def test_char_span_validation() -> None:
    with pytest.raises(ValueError, match="char_span start must be >= 0"):
        ProvenanceRecord(char_span=(-1, 10))

    with pytest.raises(ValueError, match="end must be >= start"):
        ProvenanceRecord(char_span=(10, 5))


def test_orphan_entity_cleanup(tmp_path: Path) -> None:
    db_dir = str(tmp_path / "orphan_test_db")
    doc_store = SQLiteDocumentStore(db_path=str(tmp_path / "orphan_test_db" / "documents.db"))
    vec_store = SQLiteVectorStore(db_path=str(tmp_path / "orphan_test_db" / "vectors.db"))
    graph_store = SQLiteGraphStore(db_path=str(tmp_path / "orphan_test_db" / "graph.db"))
    pipeline = DocumentIngestionPipeline(
        document_store=doc_store,
        vector_store=vec_store,
        graph_store=graph_store,
        embedding_model=HashEmbeddingModel(dim=32),
        nlp_model=RuleBasedNLPModel(),
    )

    doc = Document(
        id="unique-doc-1",
        text="Computer vision processes visual data.",
    )
    pipeline.ingest_document(doc)

    subgraph = graph_store.query_subgraph([], depth=5)
    assert len(subgraph["nodes"]) > 0

    # Delete document node
    graph_store.delete_node("unique-doc-1")

    # Assert orphan entities created exclusively for unique-doc-1 are pruned
    subgraph_after = graph_store.query_subgraph([], depth=5)
    assert len(subgraph_after["nodes"]) == 0
    assert len(subgraph_after["relationships"]) == 0

    doc_store.close()
    vec_store.close()
    graph_store.close()


def test_graph_health_checker(tmp_path: Path) -> None:
    db_dir = str(tmp_path / "health_test_db")
    doc_store = SQLiteDocumentStore(db_path=str(tmp_path / "health_test_db" / "documents.db"))
    vec_store = SQLiteVectorStore(db_path=str(tmp_path / "health_test_db" / "vectors.db"))
    graph_store = SQLiteGraphStore(db_path=str(tmp_path / "health_test_db" / "graph.db"))
    pipeline = DocumentIngestionPipeline(
        document_store=doc_store,
        vector_store=vec_store,
        graph_store=graph_store,
        embedding_model=HashEmbeddingModel(dim=32),
        nlp_model=RuleBasedNLPModel(),
    )

    doc = Document(
        id="doc1",
        text="Machine learning enables artificial intelligence systems to extract patterns.",
        metadata={"category": "AI"},
    )
    pipeline.ingest_document(doc)

    report = GraphHealthChecker.check_health(graph_store, doc_store)
    assert report.health_score > 80.0
    assert report.status == "HEALTHY"
    assert report.total_documents == 1
    assert report.total_relationships >= 1
    assert "EXPLICIT" in report.relationship_kinds or "CO_OCCURS_WITH" in report.relationship_kinds

    doc_store.close()
    vec_store.close()
    graph_store.close()


def test_llm_nlp_model_and_contradiction_detection(tmp_path: Path) -> None:
    def custom_llm_extractor(text: str) -> SemanticAnalysisResult:
        ent1 = Entity(name="Product X", type="TECHNOLOGY")
        ent2 = Entity(name="Feature Y", type="CONCEPT")
        cand = RelationshipCandidate(
            source_entity=ent1,
            target_entity=ent2,
            relation_type="CONTRADICTS",
            confidence=0.9,
            evidence_text="Product X does not support Feature Y.",
            provenance=ProvenanceRecord(
                kind=RelationshipKind.CONTRADICTS,
                evidence_text="Product X does not support Feature Y.",
                confidence=0.9,
                method="llm_extraction",
            ),
        )
        return SemanticAnalysisResult(entities=[ent1, ent2], relationship_candidates=[cand])

    llm_model = LLMNLPModel(extractor_fn=custom_llm_extractor, provider_name="test_llm")

    db_dir = str(tmp_path / "llm_test_db")
    doc_store = SQLiteDocumentStore(db_path=str(tmp_path / "llm_test_db" / "documents.db"))
    vec_store = SQLiteVectorStore(db_path=str(tmp_path / "llm_test_db" / "vectors.db"))
    graph_store = SQLiteGraphStore(db_path=str(tmp_path / "llm_test_db" / "graph.db"))
    pipeline = DocumentIngestionPipeline(
        document_store=doc_store,
        vector_store=vec_store,
        graph_store=graph_store,
        embedding_model=HashEmbeddingModel(dim=32),
        nlp_model=llm_model,
    )

    doc = Document(id="contradict-doc", text="Product X does not support Feature Y.")
    pipeline.ingest_document(doc)

    client = MindGraphDBClient(
        container=type(
            "Container",
            (),
            {
                "document_store": doc_store,
                "vector_store": vec_store,
                "graph_store": graph_store,
                "embedding_model": HashEmbeddingModel(dim=32),
            },
        )()
    )

    contradictions = client.find_contradictions()
    assert len(contradictions) >= 1
    assert contradictions[0]["relation_type"] == "CONTRADICTS"

    doc_store.close()
    vec_store.close()
    graph_store.close()


def test_client_explain_and_health_sdk(tmp_path: Path) -> None:
    client = MindGraphDBClient(db_dir=str(tmp_path / "sdk_health_db"))
    client.delete_all()

    client.create_document(
        doc_id="doc-sdk-1",
        text="Deep learning is a subset of machine learning using neural networks.",
    )

    health = client.check_health()
    assert health.health_score >= 80.0
    assert health.total_documents == 1
    assert health.total_relationships >= 1

    relationships = client.get_relationships()
    assert len(relationships) >= 1

    explanation = client.explain_relationship(relationships[0].id)
    assert explanation is not None
    assert "relationship_id" in explanation
    assert "confidence" in explanation
    assert "evidence_text" in explanation

    query_res = client.query("machine learning")
    assert len(query_res.reasoning_paths) >= 1
    assert query_res.reasoning_paths[0]["relevance_score"] >= 0.0
