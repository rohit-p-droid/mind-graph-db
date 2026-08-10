"""Query Planner compiling MindQuery AST into executable query plans."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from mind_graph_db.query.ast import MindQuery


class PlanStep(ABC):
    """Abstract step in a compiled query plan."""

    @property
    @abstractmethod
    def name(self) -> str:
        pass


class VectorSearchStep(PlanStep):
    """Execution step performing vector similarity search."""

    def __init__(self, query_text: str, top_k: int) -> None:
        self.query_text = query_text
        self.top_k = top_k

    @property
    def name(self) -> str:
        return "VectorSearchStep"


class MetadataFilterStep(PlanStep):
    """Execution step filtering documents by metadata key-values."""

    def __init__(self, filters: Dict[str, Any]) -> None:
        self.filters = filters

    @property
    def name(self) -> str:
        return "MetadataFilterStep"


class GraphTraverseStep(PlanStep):
    """Execution step performing multi-hop graph traversal from seed nodes."""

    def __init__(self, hops: int, relation_types: Optional[List[str]]) -> None:
        self.hops = hops
        self.relation_types = relation_types

    @property
    def name(self) -> str:
        return "GraphTraverseStep"


class RankAndAssembleStep(PlanStep):
    """Execution step scoring, ranking, and assembling QueryResult."""

    def __init__(self, return_fields: List[str]) -> None:
        self.return_fields = return_fields

    @property
    def name(self) -> str:
        return "RankAndAssembleStep"


class QueryPlan:
    """Compiled query plan containing an ordered sequence of execution steps."""

    def __init__(self, steps: List[PlanStep]) -> None:
        self.steps = steps


class QueryPlanner:
    """Planner transforming MindQuery AST into an executable QueryPlan."""

    def create_plan(self, query_ast: MindQuery) -> QueryPlan:
        """Compile MindQuery AST into an ordered QueryPlan.

        Pipeline stages:
        1. VectorSearchStep (if semantic_query present)
        2. MetadataFilterStep (if metadata_filters present)
        3. GraphTraverseStep (if traverse_hops > 0)
        4. RankAndAssembleStep
        """
        steps: List[PlanStep] = []

        if query_ast.semantic_query:
            steps.append(VectorSearchStep(query_text=query_ast.semantic_query, top_k=query_ast.top_k))

        if query_ast.metadata_filters:
            steps.append(MetadataFilterStep(filters=query_ast.metadata_filters))

        if query_ast.traverse_hops > 0:
            steps.append(
                GraphTraverseStep(
                    hops=query_ast.traverse_hops,
                    relation_types=query_ast.allowed_relationship_types,
                )
            )

        steps.append(RankAndAssembleStep(return_fields=query_ast.return_fields))
        return QueryPlan(steps=steps)
