"""QueryEngine abstract base class."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from mind_graph_db.core.types import QueryResult


class QueryEngine(ABC):
    """Abstract interface for hybrid semantic + vector + graph query orchestration."""

    @abstractmethod
    def query(
        self,
        query_text: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        include_subgraph: bool = True,
    ) -> QueryResult:
        """Perform a hybrid search across document, vector, and graph stores.

        Args:
            query_text: Natural language query string.
            top_k: Maximum number of relevant document matches to return.
            filters: Optional metadata filters for narrowing search space.
            include_subgraph: Whether to expand and include related graph entities and edges.

        Returns:
            Unified QueryResult object containing matching documents, entities, scores, and metadata.
        """
        pass
