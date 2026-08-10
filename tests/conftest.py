"""Pytest configuration and test fixtures for Mind Graph DB."""

from typing import Any, Dict, Iterator, List, Optional, Tuple

import pytest

from mind_graph_db.config.settings import Settings
from mind_graph_db.core.types import (
    Document,
    Entity,
    Fact,
    GraphNode,
    QueryResult,
    Relationship,
    RelationshipCandidate,
    TraversalPath,
    Vector,
)


from mind_graph_db.interfaces.document_store import DocumentStore
from mind_graph_db.interfaces.embedding import EmbeddingModel
from mind_graph_db.interfaces.graph_store import GraphStore
from mind_graph_db.interfaces.nlp import NLPModel
from mind_graph_db.interfaces.query_engine import QueryEngine
from mind_graph_db.interfaces.vector_store import VectorStore
from mind_graph_db.storage.sqlite_document_store import SQLiteDocumentStore
from mind_graph_db.storage.sqlite_graph_store import SQLiteGraphStore
from mind_graph_db.storage.sqlite_vector_store import SQLiteVectorStore





class InMemoryDocumentStore(DocumentStore):
    """Stub implementation of DocumentStore using in-memory dictionary."""

    def __init__(self) -> None:
        self._docs: Dict[str, Document] = {}

    def put(self, document: Document) -> str:
        self._docs[document.id] = document
        return document.id

    def get(self, document_id: str) -> Optional[Document]:
        return self._docs.get(document_id)

    def update(self, document: Document) -> bool:
        if document.id in self._docs:
            document.touch()
            self._docs[document.id] = document
            return True
        return False

    def delete(self, document_id: str) -> bool:
        if document_id in self._docs:
            del self._docs[document_id]
            return True
        return False


    def list_documents(self, limit: int = 100, offset: int = 0) -> List[Document]:
        items = list(self._docs.values())
        return items[offset : offset + limit]

    def count(self) -> int:
        return len(self._docs)


class InMemoryVectorStore(VectorStore):
    """Stub implementation of VectorStore using in-memory dictionary."""

    def __init__(self) -> None:
        self._vectors: Dict[str, Vector] = {}

    def add(
        self,
        vector_id: str,
        embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._vectors[vector_id] = Vector(
            id=vector_id,
            embedding=embedding,
            metadata=metadata or {},
        )

    def get(self, vector_id: str) -> Optional[Vector]:
        return self._vectors.get(vector_id)

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Vector, float]]:
        # Returns raw dummy similarity score of 1.0 for first top_k
        items = list(self._vectors.values())[:top_k]
        return [(vec, 1.0) for vec in items]

    def delete(self, vector_id: str) -> bool:
        if vector_id in self._vectors:
            del self._vectors[vector_id]
            return True
        return False

    def count(self) -> int:
        return len(self._vectors)


class InMemoryGraphStore(GraphStore):
    """Stub implementation of GraphStore using in-memory dicts."""

    def __init__(self) -> None:
        self._nodes: Dict[str, GraphNode] = {}
        self._relationships: List[Relationship] = []

    def add_node(self, node: GraphNode) -> str:
        self._nodes[node.id] = node
        return node.id

    def get_node(self, node_id: str) -> Optional[GraphNode]:
        return self._nodes.get(node_id)

    def get_node_by_label(self, label: str, node_type: str = "ENTITY") -> Optional[GraphNode]:
        clean_target = label.strip().lower()
        for node in self._nodes.values():
            if node.label.lower() == clean_target and node.node_type == node_type:
                return node
        return None


    def delete_node(self, node_id: str) -> bool:
        if node_id in self._nodes:
            del self._nodes[node_id]
            self._relationships = [
                r for r in self._relationships if r.source_id != node_id and r.target_id != node_id
            ]
            return True
        return False

    def add_relationship(self, relationship: Relationship) -> None:
        self._relationships.append(relationship)

    def delete_relationship(
        self,
        source_id: str,
        target_id: str,
        relation_type: Optional[str] = None,
    ) -> bool:
        orig_len = len(self._relationships)
        self._relationships = [
            r for r in self._relationships
            if not (r.source_id == source_id and r.target_id == target_id and (relation_type is None or r.relation_type == relation_type))
        ]
        return len(self._relationships) < orig_len

    def get_neighbors(
        self,
        node_id: str,
        relation_types: Optional[List[str]] = None,
        max_depth: int = 1,
    ) -> List[Tuple[GraphNode, Relationship]]:
        res: List[Tuple[GraphNode, Relationship]] = []
        for rel in self._relationships:
            target_id = None
            if rel.source_id == node_id:
                target_id = rel.target_id
            elif rel.target_id == node_id:
                target_id = rel.source_id

            if target_id and target_id in self._nodes:
                if relation_types is None or rel.relation_type in relation_types:
                    res.append((self._nodes[target_id], rel))
        return res

    def traverse(
        self,
        start_node_id: str,
        max_hops: int = 2,
        relation_types: Optional[List[str]] = None,
    ) -> List[TraversalPath]:
        start_node = self.get_node(start_node_id)
        if start_node is None:
            return []

        paths: List[TraversalPath] = []
        queue: List[Tuple[str, List[GraphNode], List[Relationship], int]] = [
            (start_node_id, [start_node], [], 0)
        ]
        visited = {start_node_id}


        while queue:
            curr_id, curr_nodes, curr_edges, curr_hops = queue.pop(0)

            if curr_hops > 0:
                paths.append(TraversalPath(nodes=curr_nodes, edges=curr_edges, hop_count=curr_hops))

            if curr_hops >= max_hops:
                continue

            neighbors = self.get_neighbors(curr_id, relation_types=relation_types)
            for neighbor_node, edge in neighbors:
                if neighbor_node.id not in visited:
                    visited.add(neighbor_node.id)
                    queue.append((neighbor_node.id, curr_nodes + [neighbor_node], curr_edges + [edge], curr_hops + 1))

        return paths

    def query_subgraph(
        self,
        start_node_ids: List[str],
        depth: int = 2,
    ) -> Dict[str, Any]:
        matched_nodes = [self._nodes[nid] for nid in start_node_ids if nid in self._nodes] if start_node_ids else list(self._nodes.values())
        return {
            "nodes": matched_nodes,
            "relationships": self._relationships,
        }




class DummyEmbeddingModel(EmbeddingModel):
    """Stub implementation of EmbeddingModel."""

    def __init__(self, dim: int = 4) -> None:
        self._dim = dim

    def encode_text(self, text: str) -> List[float]:
        return [0.1] * self._dim

    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        return [[0.1] * self._dim for _ in texts]

    @property
    def dimension(self) -> int:
        return self._dim


class DummyNLPModel(NLPModel):
    """Stub implementation of NLPModel."""

    def extract_entities(self, text: str) -> List[Entity]:
        return [Entity(name="TestEntity", type="CONCEPT")]

    def extract_concepts(self, text: str) -> List[str]:
        return ["test_concept"]

    def extract_facts(self, text: str) -> List[Fact]:
        return [Fact(subject="Test", predicate="is", object="Sample")]

    def extract_relationship_candidates(self, text: str) -> List[RelationshipCandidate]:
        return []

    def extract_relationships(
        self,
        text: str,
        entities: Optional[List[Entity]] = None,
    ) -> List[Relationship]:
        return []



class DummyQueryEngine(QueryEngine):
    """Stub implementation of QueryEngine."""

    def query(
        self,
        query_text: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        include_subgraph: bool = True,
    ) -> QueryResult:
        return QueryResult(query=query_text)


@pytest.fixture
def sample_settings() -> Settings:
    return Settings(environment="testing", log_level="DEBUG")


@pytest.fixture
def in_memory_doc_store() -> DocumentStore:
    return InMemoryDocumentStore()


@pytest.fixture
def in_memory_vector_store() -> VectorStore:
    return InMemoryVectorStore()


@pytest.fixture
def in_memory_graph_store() -> GraphStore:
    return InMemoryGraphStore()


@pytest.fixture
def dummy_embedding_model() -> EmbeddingModel:
    return DummyEmbeddingModel(dim=4)


@pytest.fixture
def dummy_nlp_model() -> NLPModel:
    return DummyNLPModel()


@pytest.fixture
def dummy_query_engine() -> QueryEngine:
    return DummyQueryEngine()


@pytest.fixture
def sqlite_doc_store(tmp_path: Any) -> Iterator[SQLiteDocumentStore]:
    db_file = str(tmp_path / "test_docs.db")
    store = SQLiteDocumentStore(db_path=db_file)
    yield store
    store.close()


@pytest.fixture
def sqlite_graph_store(tmp_path: Any) -> Iterator[SQLiteGraphStore]:
    db_file = str(tmp_path / "test_graph.db")
    store = SQLiteGraphStore(db_path=db_file)
    yield store
    store.close()



