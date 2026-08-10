"""Abstract Syntax Tree (AST) representation for Mind Graph DB query language."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class MindQuery(BaseModel):
    """AST representation of a Mind Graph DB query."""

    target: str = "documents"
    semantic_query: Optional[str] = None
    metadata_filters: Dict[str, Any] = Field(default_factory=dict)
    traverse_hops: int = 0
    allowed_relationship_types: Optional[List[str]] = None
    return_fields: List[str] = Field(
        default_factory=lambda: ["documents", "entities", "relationships", "scores"]
    )
    top_k: int = 10
