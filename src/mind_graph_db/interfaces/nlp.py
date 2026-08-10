"""NLPModel abstract base class."""

from abc import ABC, abstractmethod
from typing import List, Optional

from mind_graph_db.core.types import (
    Entity,
    Fact,
    Relationship,
    RelationshipCandidate,
    SemanticAnalysisResult,
)


class NLPModel(ABC):
    """Abstract interface for extracting entities, concepts, facts, and relationships from text."""

    @abstractmethod
    def extract_entities(self, text: str) -> List[Entity]:
        """Extract named entities from raw text.

        Args:
            text: Unstructured input text.

        Returns:
            List of extracted Entity objects.
        """
        pass

    @abstractmethod
    def extract_concepts(self, text: str) -> List[str]:
        """Extract important key concepts and noun phrases from raw text.

        Args:
            text: Unstructured input text.

        Returns:
            List of concept string labels.
        """
        pass

    @abstractmethod
    def extract_facts(self, text: str) -> List[Fact]:
        """Extract Subject-Verb-Object (SVO) factual triples from raw text.

        Args:
            text: Unstructured input text.

        Returns:
            List of Fact objects.
        """
        pass

    @abstractmethod
    def extract_relationship_candidates(self, text: str) -> List[RelationshipCandidate]:
        """Extract candidate entity-to-entity relationships with evidence.

        Args:
            text: Unstructured input text.

        Returns:
            List of RelationshipCandidate objects.
        """
        pass

    @abstractmethod
    def extract_relationships(
        self,
        text: str,
        entities: Optional[List[Entity]] = None,
    ) -> List[Relationship]:
        """Extract semantic relationships between entities in raw text.

        Args:
            text: Unstructured input text.
            entities: Optional pre-extracted entity list. If None, entities will be extracted first.

        Returns:
            List of extracted Relationship objects connecting entities.
        """
        pass

    def analyze(self, text: str, document_id: Optional[str] = None) -> SemanticAnalysisResult:
        """Perform comprehensive semantic analysis on raw text.

        Args:
            text: Unstructured input text.
            document_id: Optional identifier of the originating Document.

        Returns:
            Structured SemanticAnalysisResult object containing entities, concepts, facts, and relationship candidates.
        """
        entities = self.extract_entities(text)
        concepts = self.extract_concepts(text)
        facts = self.extract_facts(text)
        candidates = self.extract_relationship_candidates(text)

        return SemanticAnalysisResult(
            document_id=document_id,
            entities=entities,
            concepts=concepts,
            facts=facts,
            relationship_candidates=candidates,
        )
