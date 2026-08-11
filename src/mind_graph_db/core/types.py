"""Core domain entities and data structures for Mind Graph DB."""

from datetime import datetime, timezone
from enum import Enum
import uuid
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field, field_validator


def _generate_uuid() -> str:
    return str(uuid.uuid4())


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class RelationshipKind(str, Enum):
    """Explicit taxonomy of relationship evidence kinds."""

    EXPLICIT = "EXPLICIT"
    INFERRED = "INFERRED"
    SIMILAR_TO = "SIMILAR_TO"
    CO_OCCURS_WITH = "CO_OCCURS_WITH"
    DERIVED_FROM = "DERIVED_FROM"
    SUPPORTS = "SUPPORTS"
    CONTRADICTS = "CONTRADICTS"
    RELATED_TO = "RELATED_TO"


class Document(BaseModel):
    """Represents an unstructured text document with metadata."""

    id: str = Field(default_factory=_generate_uuid)
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=_utc_now)
    updated_at: datetime = Field(default_factory=_utc_now)

    @field_validator("text")
    @classmethod
    def validate_text_not_empty(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("Document text cannot be empty or whitespace-only.")
        return v

    def touch(self) -> None:
        """Update the updated_at timestamp to the current UTC time."""
        self.updated_at = _utc_now()


class Entity(BaseModel):
    """Represents a named entity extracted from text or defined manually."""

    id: str = Field(default_factory=_generate_uuid)
    name: str
    type: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class GraphNode(BaseModel):
    """Represents a node in the Mind Graph DB knowledge graph (Entity, Document, Concept)."""

    id: str = Field(default_factory=_generate_uuid)
    label: str
    node_type: str = "ENTITY"
    properties: Dict[str, Any] = Field(default_factory=dict)


class ProvenanceRecord(BaseModel):
    """First-class evidence provenance primitive explaining why a relationship exists."""

    kind: RelationshipKind = RelationshipKind.EXPLICIT
    source_doc_id: Optional[str] = None
    evidence_text: Optional[str] = None
    char_span: Optional[Tuple[int, int]] = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    timestamp: datetime = Field(default_factory=_utc_now)
    @field_validator("char_span")
    @classmethod
    def validate_char_span(cls, v: Optional[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
        if v is not None:
            start, end = v
            if start < 0 or end < start:
                raise ValueError("char_span start must be >= 0 and end must be >= start")
        return v


class Relationship(BaseModel):
    """Represents a directed semantic relationship edge between two graph nodes."""

    id: str = Field(default_factory=_generate_uuid)
    source_id: str
    target_id: str
    relation_type: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    evidence_text: Optional[str] = None
    provenance: Optional[ProvenanceRecord] = None
    properties: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TraversalPath(BaseModel):
    """Represents a multi-hop path traversed through the knowledge graph."""

    nodes: List[GraphNode] = Field(default_factory=list)
    edges: List[Relationship] = Field(default_factory=list)
    hop_count: int = 0


class Vector(BaseModel):
    """Represents a numerical embedding vector with metadata."""

    id: str
    embedding: List[float]
    metadata: Dict[str, Any] = Field(default_factory=dict)


class QueryResult(BaseModel):
    """Encapsulates the response from a hybrid query against Mind Graph DB."""

    query: str
    documents: List[Document] = Field(default_factory=list)
    entities: List[Entity] = Field(default_factory=list)
    relationships: List[Relationship] = Field(default_factory=list)
    reasoning_paths: List[Dict[str, Any]] = Field(default_factory=list)
    scores: Dict[str, float] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Fact(BaseModel):
    """Represents a Subject-Verb-Object factual triple extracted from text."""

    subject: str
    predicate: str
    object_: str = Field(default="", alias="object")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RelationshipCandidate(BaseModel):
    """Represents a candidate relationship connecting two entities before graph commit."""

    source_entity: Entity
    target_entity: Entity
    relation_type: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    evidence_text: Optional[str] = None
    provenance: Optional[ProvenanceRecord] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SemanticAnalysisResult(BaseModel):
    """Encapsulates structured semantic extraction results for a document or text."""

    document_id: Optional[str] = None
    entities: List[Entity] = Field(default_factory=list)
    concepts: List[str] = Field(default_factory=list)
    facts: List[Fact] = Field(default_factory=list)
    relationship_candidates: List[RelationshipCandidate] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
