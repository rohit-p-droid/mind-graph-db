"""Unit tests for SemanticAnalysisService orchestrator."""

from mind_graph_db.core.types import Document, SemanticAnalysisResult
from mind_graph_db.nlp import RuleBasedNLPModel
from mind_graph_db.services.semantic_service import SemanticAnalysisService


def test_semantic_analysis_service_document() -> None:
    nlp_model = RuleBasedNLPModel()
    service = SemanticAnalysisService(nlp_model=nlp_model)

    doc = Document(
        id="doc-semantic-1",
        text="Mind Graph DB is created by Python developers to manage vector graph storage.",
        metadata={"source": "unittest"},
    )

    result = service.analyze_document(doc)

    assert isinstance(result, SemanticAnalysisResult)
    assert result.document_id == "doc-semantic-1"
    assert len(result.entities) >= 1
    assert len(result.concepts) >= 1
    assert isinstance(result.facts, list)
    assert isinstance(result.relationship_candidates, list)
