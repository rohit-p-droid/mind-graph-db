"""Graph health diagnostics and validation utility for Mind Graph DB."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from mind_graph_db.interfaces.document_store import DocumentStore
from mind_graph_db.interfaces.graph_store import GraphStore


class GraphHealthReport(BaseModel):
    """Encapsulates health diagnostic metrics, warnings, and health score for a knowledge graph."""

    health_score: float = Field(..., ge=0.0, le=100.0)
    status: str = "HEALTHY"  # HEALTHY, WARNING, CRITICAL
    total_nodes: int = 0
    total_documents: int = 0
    total_entities: int = 0
    total_relationships: int = 0
    relationship_kinds: Dict[str, int] = Field(default_factory=dict)
    orphan_entities: List[str] = Field(default_factory=list)
    invalid_provenance_count: int = 0
    diagnostics: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)


class GraphHealthChecker:
    """Automated diagnostic checker verifying graph health, orphan nodes, and provenance integrity."""

    @staticmethod
    def check_health(
        graph_store: GraphStore,
        document_store: Optional[DocumentStore] = None,
    ) -> GraphHealthReport:
        """Analyze knowledge graph integrity, relationship kinds, orphan nodes, and provenance completeness."""
        subgraph = graph_store.query_subgraph([], depth=100)
        nodes = subgraph.get("nodes", [])
        relationships = subgraph.get("relationships", [])

        total_nodes = len(nodes)
        total_relationships = len(relationships)

        doc_nodes = [n for n in nodes if n.node_type == "DOCUMENT"]
        entity_nodes = [n for n in nodes if n.node_type == "ENTITY"]

        total_documents = len(doc_nodes)
        total_entities = len(entity_nodes)

        # Count relationship kinds
        kind_counts: Dict[str, int] = {}
        invalid_prov_count = 0

        connected_node_ids = set()
        for rel in relationships:
            connected_node_ids.add(rel.source_id)
            connected_node_ids.add(rel.target_id)

            kind_val = rel.provenance.kind.value if hasattr(rel.provenance.kind, "value") else str(rel.provenance.kind)
            kind_counts[kind_val] = kind_counts.get(kind_val, 0) + 1

            if not rel.evidence_text and not rel.provenance:
                invalid_prov_count += 1

        # Detect orphan entities
        orphan_entities = [e.label for e in entity_nodes if e.id not in connected_node_ids]

        # Calculate health score
        diagnostics = []
        recommendations = []
        penalty = 0.0

        if orphan_entities:
            penalty += min(30.0, len(orphan_entities) * 5.0)
            diagnostics.append(f"Found {len(orphan_entities)} orphan entity nodes with 0 graph relationships: {orphan_entities[:5]}")
            recommendations.append("Run ingestion pipeline to resolve orphan entities or prune unused nodes.")

        if invalid_prov_count > 0:
            penalty += min(20.0, invalid_prov_count * 5.0)
            diagnostics.append(f"Found {invalid_prov_count} relationships without evidence provenance.")
            recommendations.append("Ensure relationship creation attaches a ProvenanceRecord.")

        health_score = max(0.0, min(100.0, 100.0 - penalty))
        status = "HEALTHY" if health_score >= 85.0 else ("WARNING" if health_score >= 60.0 else "CRITICAL")

        return GraphHealthReport(
            health_score=round(health_score, 1),
            status=status,
            total_nodes=total_nodes,
            total_documents=total_documents,
            total_entities=total_entities,
            total_relationships=total_relationships,
            relationship_kinds=kind_counts,
            orphan_entities=orphan_entities,
            invalid_provenance_count=invalid_prov_count,
            diagnostics=diagnostics,
            recommendations=recommendations,
        )
