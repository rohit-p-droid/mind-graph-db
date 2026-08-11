"""Retrieval quality, multi-hop graph traversal, and reasoning paths tests."""

from pathlib import Path
from mind_graph_db.sdk import MindGraphDBClient


def test_hybrid_search_and_explainable_paths(tmp_path: Path) -> None:
    client = MindGraphDBClient(db_dir=str(tmp_path / "retrieval_db"))
    client.delete_all()

    client.create_document(
        doc_id="ret-doc-1",
        text="Machine learning is a branch of artificial intelligence.",
        metadata={"category": "AI", "year": 2023},
    )
    client.create_document(
        doc_id="ret-doc-2",
        text="Deep learning is a subset of machine learning that uses neural networks.",
        metadata={"category": "AI", "year": 2024},
    )

    # Hybrid Query with metadata filtering
    res = client.query("machine learning", top_k=5)
    assert len(res.documents) >= 1
    assert len(res.reasoning_paths) >= 1

    path = res.reasoning_paths[0]
    assert "document_id" in path
    assert "relevance_score" in path
    assert "evidence_snippet" in path
    assert "supporting_relationships" in path


def test_multi_hop_graph_traversal(tmp_path: Path) -> None:
    client = MindGraphDBClient(db_dir=str(tmp_path / "traverse_db"))
    client.delete_all()

    doc1 = client.create_document(
        doc_id="t-doc-1",
        text="Machine learning is a branch of artificial intelligence.",
    )

    paths_1hop = client.traverse(start_node_id="t-doc-1", max_hops=1)
    paths_2hop = client.traverse(start_node_id="t-doc-1", max_hops=2)

    assert len(paths_1hop) >= 1
    assert len(paths_2hop) >= len(paths_1hop)
