"""GraphStore abstract base class."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from mind_graph_db.core.types import Entity, GraphNode, Relationship, TraversalPath


class GraphStore(ABC):
    """Abstract interface for managing knowledge graphs, nodes, and semantic edges."""

    @abstractmethod
    def add_node(self, node: GraphNode) -> str:
        """Add or update a node (Entity, Document, Concept) in the graph.

        Args:
            node: GraphNode object to insert.

        Returns:
            The node ID.
        """
        pass

    @abstractmethod
    def get_node(self, node_id: str) -> Optional[GraphNode]:
        """Retrieve a node by ID.

        Args:
            node_id: Unique node identifier.

        Returns:
            The GraphNode if found, otherwise None.
        """
        pass

    @abstractmethod
    def get_node_by_label(self, label: str, node_type: str = "ENTITY") -> Optional[GraphNode]:
        """Retrieve a node by label and node_type (for entity resolution).

        Args:
            label: Case-insensitive label/name string.
            node_type: Node type string ('ENTITY', 'DOCUMENT', 'CONCEPT').

        Returns:
            The GraphNode if found, otherwise None.
        """
        pass


    @abstractmethod
    def delete_node(self, node_id: str) -> bool:
        """Delete a node and all connected incoming/outgoing relationship edges.

        Args:
            node_id: Unique node identifier.

        Returns:
            True if deleted, False if node ID not found.
        """
        pass

    @abstractmethod
    def add_relationship(self, relationship: Relationship) -> None:
        """Add a directed relationship edge between two nodes in the graph.

        Args:
            relationship: Relationship object to insert.
        """
        pass

    @abstractmethod
    def delete_relationship(
        self,
        source_id: str,
        target_id: str,
        relation_type: Optional[str] = None,
    ) -> bool:
        """Remove a relationship edge between two nodes.

        Args:
            source_id: Source node ID.
            target_id: Target node ID.
            relation_type: Optional specific relation type filter.

        Returns:
            True if deleted, False if matching edge not found.
        """
        pass

    @abstractmethod
    def get_neighbors(
        self,
        node_id: str,
        relation_types: Optional[List[str]] = None,
        max_depth: int = 1,
    ) -> List[Tuple[GraphNode, Relationship]]:
        """Find neighboring nodes connected to the specified starting node.

        Args:
            node_id: Starting node identifier.
            relation_types: Optional list of relationship types to filter by.
            max_depth: Traversal depth limit.

        Returns:
            List of (Neighbor GraphNode, Relationship edge) tuples.
        """
        pass

    @abstractmethod
    def traverse(
        self,
        start_node_id: str,
        max_hops: int = 2,
        relation_types: Optional[List[str]] = None,
    ) -> List[TraversalPath]:
        """Traverse the graph using Breadth-First Search up to max_hops limit.

        Args:
            start_node_id: Starting node ID.
            max_hops: Maximum hop count depth.
            relation_types: Optional list of allowed relationship types.

        Returns:
            List of TraversalPath objects representing multi-hop paths.
        """
        pass

    @abstractmethod
    def query_subgraph(
        self,
        start_node_ids: List[str],
        depth: int = 2,
    ) -> Dict[str, Any]:
        """Extract a subgraph starting from a set of seed nodes.

        Args:
            start_node_ids: List of seed node IDs.
            depth: Traversal depth.

        Returns:
            Subgraph dictionary representation containing 'nodes' and 'relationships'.
        """
        pass

    # Backwards compatibility wrappers for Entity
    def add_entity(self, entity: Entity) -> str:
        """Add or update an Entity node (wrapper around add_node)."""
        node = GraphNode(
            id=entity.id,
            label=entity.name,
            node_type="ENTITY",
            properties={"entity_type": entity.type, **entity.metadata},
        )
        return self.add_node(node)

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Retrieve an Entity node by ID (wrapper around get_node)."""
        node = self.get_node(entity_id)
        if node is None:
            return None
        entity_type = str(node.properties.get("entity_type", "CONCEPT"))
        meta = {k: v for k, v in node.properties.items() if k != "entity_type"}
        return Entity(id=node.id, name=node.label, type=entity_type, metadata=meta)

    def delete_entity(self, entity_id: str) -> bool:
        """Remove an Entity node (wrapper around delete_node)."""
        return self.delete_node(entity_id)
