"""Unit tests for SQLiteGraphStore node creation, relationships, traversal, and deletion."""

from pathlib import Path
from mind_graph_db.core.types import GraphNode, Relationship
from mind_graph_db.storage.sqlite_graph_store import SQLiteGraphStore


def test_sqlite_graph_store_node_creation(sqlite_graph_store: SQLiteGraphStore) -> None:
    node_ent = GraphNode(id="n-1", label="Python", node_type="ENTITY", properties={"category": "language"})
    node_doc = GraphNode(id="n-doc1", label="Python Spec Doc", node_type="DOCUMENT", properties={"author": "Guido"})

    sqlite_graph_store.add_node(node_ent)
    sqlite_graph_store.add_node(node_doc)

    retrieved_ent = sqlite_graph_store.get_node("n-1")
    assert retrieved_ent is not None
    assert retrieved_ent.label == "Python"
    assert retrieved_ent.node_type == "ENTITY"
    assert retrieved_ent.properties == {"category": "language"}

    retrieved_doc = sqlite_graph_store.get_node("n-doc1")
    assert retrieved_doc is not None
    assert retrieved_doc.node_type == "DOCUMENT"
    assert retrieved_doc.properties["author"] == "Guido"


def test_sqlite_graph_store_relationships_and_neighbors(sqlite_graph_store: SQLiteGraphStore) -> None:
    n1 = GraphNode(id="n1", label="Mind Graph DB", node_type="ENTITY")
    n2 = GraphNode(id="n2", label="SQLite", node_type="ENTITY")
    n3 = GraphNode(id="n3", label="Vector Index", node_type="CONCEPT")

    sqlite_graph_store.add_node(n1)
    sqlite_graph_store.add_node(n2)
    sqlite_graph_store.add_node(n3)

    rel1 = Relationship(
        id="rel-1",
        source_id="n1",
        target_id="n2",
        relation_type="USES_STORAGE",
        confidence=0.95,
        evidence_text="Mind Graph DB uses SQLite for persistent storage.",
    )
    rel2 = Relationship(
        id="rel-2",
        source_id="n1",
        target_id="n3",
        relation_type="CONTAINS_INDEX",
        confidence=0.90,
    )

    sqlite_graph_store.add_relationship(rel1)
    sqlite_graph_store.add_relationship(rel2)

    neighbors = sqlite_graph_store.get_neighbors("n1")
    assert len(neighbors) == 2
    labels = [node.label for node, _ in neighbors]
    assert "SQLite" in labels
    assert "Vector Index" in labels

    # Test filtering neighbors by relation type
    filtered = sqlite_graph_store.get_neighbors("n1", relation_types=["USES_STORAGE"])
    assert len(filtered) == 1
    assert filtered[0][0].label == "SQLite"
    assert filtered[0][1].evidence_text == "Mind Graph DB uses SQLite for persistent storage."


def test_sqlite_graph_store_deletions(sqlite_graph_store: SQLiteGraphStore) -> None:
    n1 = GraphNode(id="n10", label="Node A", node_type="ENTITY")
    n2 = GraphNode(id="n20", label="Node B", node_type="ENTITY")
    sqlite_graph_store.add_node(n1)
    sqlite_graph_store.add_node(n2)

    rel = Relationship(source_id="n10", target_id="n20", relation_type="LINKED_TO")
    sqlite_graph_store.add_relationship(rel)

    # Delete relationship
    assert sqlite_graph_store.delete_relationship("n10", "n20", relation_type="LINKED_TO") is True
    assert len(sqlite_graph_store.get_neighbors("n10")) == 0

    # Re-add relationship and test node deletion cascade
    sqlite_graph_store.add_relationship(rel)
    assert len(sqlite_graph_store.get_neighbors("n10")) == 1

    assert sqlite_graph_store.delete_node("n10") is True
    assert sqlite_graph_store.get_node("n10") is None
    assert len(sqlite_graph_store.get_neighbors("n20")) == 0


def test_sqlite_graph_store_multi_hop_traversal(sqlite_graph_store: SQLiteGraphStore) -> None:
    # Construct a 3-hop graph chain: A -> B -> C -> D
    nA = GraphNode(id="nA", label="A", node_type="ENTITY")
    nB = GraphNode(id="nB", label="B", node_type="ENTITY")
    nC = GraphNode(id="nC", label="C", node_type="ENTITY")
    nD = GraphNode(id="nD", label="D", node_type="ENTITY")

    for n in [nA, nB, nC, nD]:
        sqlite_graph_store.add_node(n)

    r1 = Relationship(source_id="nA", target_id="nB", relation_type="STEP_1")
    r2 = Relationship(source_id="nB", target_id="nC", relation_type="STEP_2")
    r3 = Relationship(source_id="nC", target_id="nD", relation_type="STEP_3")

    for r in [r1, r2, r3]:
        sqlite_graph_store.add_relationship(r)

    # 1-hop traversal from A
    paths_1 = sqlite_graph_store.traverse("nA", max_hops=1)
    assert len(paths_1) == 1
    assert paths_1[0].hop_count == 1
    assert paths_1[0].nodes[-1].id == "nB"

    # 2-hop traversal from A
    paths_2 = sqlite_graph_store.traverse("nA", max_hops=2)
    assert len(paths_2) == 2
    max_hop_path = max(paths_2, key=lambda p: p.hop_count)
    assert max_hop_path.hop_count == 2
    assert max_hop_path.nodes[-1].id == "nC"

    # 3-hop traversal from A
    paths_3 = sqlite_graph_store.traverse("nA", max_hops=3)
    assert len(paths_3) == 3
    final_path = max(paths_3, key=lambda p: p.hop_count)
    assert final_path.hop_count == 3
    assert final_path.nodes[-1].id == "nD"


def test_sqlite_graph_store_persistence(tmp_path: Path) -> None:
    db_file = str(tmp_path / "persistent_graph.db")

    store1 = SQLiteGraphStore(db_path=db_file)
    n1 = GraphNode(id="pn1", label="Persistent Node 1", node_type="ENTITY")
    n2 = GraphNode(id="pn2", label="Persistent Node 2", node_type="ENTITY")
    store1.add_node(n1)
    store1.add_node(n2)
    rel = Relationship(source_id="pn1", target_id="pn2", relation_type="PERSISTENT_LINK")
    store1.add_relationship(rel)
    store1.close()

    store2 = SQLiteGraphStore(db_path=db_file)
    retrieved_n1 = store2.get_node("pn1")
    assert retrieved_n1 is not None
    assert retrieved_n1.label == "Persistent Node 1"

    neighbors = store2.get_neighbors("pn1")
    assert len(neighbors) == 1
    assert neighbors[0][0].label == "Persistent Node 2"
    assert neighbors[0][1].relation_type == "PERSISTENT_LINK"
    store2.close()
