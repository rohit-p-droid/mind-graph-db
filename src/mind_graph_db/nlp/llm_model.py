"""LLM/SLM extraction engine implementation supporting pluggable structured extraction."""

from typing import Callable, List, Optional
from mind_graph_db.core.types import (
    Entity,
    Fact,
    ProvenanceRecord,
    Relationship,
    RelationshipCandidate,
    RelationshipKind,
    SemanticAnalysisResult,
)
from mind_graph_db.interfaces.nlp import NLPModel
from mind_graph_db.nlp.rule_based_model import RuleBasedNLPModel


class LLMNLPModel(NLPModel):
    """Structured LLM/SLM NLP extraction model wrapper supporting custom extraction functions."""

    def __init__(
        self,
        extractor_fn: Optional[Callable[[str], SemanticAnalysisResult]] = None,
        fallback_model: Optional[NLPModel] = None,
        provider_name: str = "custom_llm",
    ) -> None:
        """Initialize LLMNLPModel with a custom extraction function or fallback model."""
        self.extractor_fn = extractor_fn
        self.fallback_model = fallback_model or RuleBasedNLPModel()
        self.provider_name = provider_name

    def analyze(self, text: str, document_id: Optional[str] = None) -> SemanticAnalysisResult:
        """Analyze text using custom LLM function or fallback engine."""
        if self.extractor_fn:
            try:
                res = self.extractor_fn(text)
                res.document_id = document_id
                return res
            except Exception:
                pass
        return self.fallback_model.analyze(text, document_id=document_id)

    def extract_entities(self, text: str) -> List[Entity]:
        res = self.analyze(text)
        return res.entities

    def extract_concepts(self, text: str) -> List[str]:
        res = self.analyze(text)
        return res.concepts

    def extract_facts(self, text: str) -> List[Fact]:
        res = self.analyze(text)
        return res.facts

    def extract_relationship_candidates(self, text: str) -> List[RelationshipCandidate]:
        res = self.analyze(text)
        return res.relationship_candidates

    def extract_relationships(
        self, text: str, entities: Optional[List[Entity]] = None
    ) -> List[Relationship]:
        res = self.analyze(text)
        return [
            Relationship(
                source_id=c.source_entity.id,
                target_id=c.target_entity.id,
                relation_type=c.relation_type,
                confidence=c.confidence,
                evidence_text=c.evidence_text,
                provenance=c.provenance
                or ProvenanceRecord(
                    kind=RelationshipKind.INFERRED,
                    evidence_text=c.evidence_text,
                    confidence=c.confidence,
                    method=self.provider_name,
                ),
            )
            for c in res.relationship_candidates
        ]
