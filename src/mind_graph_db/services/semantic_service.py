"""Service for performing semantic analysis on text documents."""

from typing import Optional

from mind_graph_db.core.types import Document, SemanticAnalysisResult
from mind_graph_db.interfaces.nlp import NLPModel


class SemanticAnalysisService:
    """Service orchestrating NLP entity, concept, fact, and relationship candidate extraction."""

    def __init__(self, nlp_model: NLPModel) -> None:
        """Initialize SemanticAnalysisService with an NLPModel instance.

        Args:
            nlp_model: Instance implementing NLPModel interface.
        """
        self.nlp_model = nlp_model

    def analyze_text(self, text: str, document_id: Optional[str] = None) -> SemanticAnalysisResult:
        """Analyze raw text string and return structured semantic analysis.

        Args:
            text: Unstructured text content.
            document_id: Optional ID of originating document.

        Returns:
            Structured SemanticAnalysisResult (entities, concepts, facts, relationship candidates).
        """
        return self.nlp_model.analyze(text, document_id=document_id)

    def analyze_document(self, document: Document) -> SemanticAnalysisResult:
        """Analyze a Document object and return structured semantic analysis.

        Args:
            document: Document domain model instance.

        Returns:
            Structured SemanticAnalysisResult tied to document.id.
        """
        return self.analyze_text(document.text, document_id=document.id)
