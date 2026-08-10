"""Unit tests for SQLiteVectorStore persistent vector store implementation."""

import pytest
from pathlib import Path
from mind_graph_db.storage.sqlite_vector_store import SQLiteVectorStore



def test_sqlite_vector_store_crud(tmp_path: Path) -> None:
    db_file = str(tmp_path / "test_vectors.db")
    store = SQLiteVectorStore(db_path=db_file)

    # Add
    store.add("vec-1", [1.0, 0.0, 0.0], metadata={"category": "tech"})
    store.add("vec-2", [0.0, 1.0, 0.0], metadata={"category": "finance"})
    store.add("vec-3", [0.9, 0.1, 0.0], metadata={"category": "tech"})

    assert store.count() == 3

    # Get
    vec1 = store.get("vec-1")
    assert vec1 is not None
    assert vec1.id == "vec-1"
    assert vec1.embedding == [1.0, 0.0, 0.0]
    assert vec1.metadata == {"category": "tech"}

    # Search similarity ranking
    results = store.search(query_embedding=[1.0, 0.0, 0.0], top_k=2)
    assert len(results) == 2
    assert results[0][0].id == "vec-1"
    assert pytest.approx(results[0][1], abs=1e-4) == 1.0
    assert results[1][0].id == "vec-3"

    # Search with filter
    filtered = store.search(query_embedding=[0.0, 1.0, 0.0], top_k=10, filters={"category": "finance"})
    assert len(filtered) == 1
    assert filtered[0][0].id == "vec-2"

    # Delete
    assert store.delete("vec-1") is True
    assert store.get("vec-1") is None
    assert store.count() == 2

    store.close()


def test_sqlite_vector_store_persistence(tmp_path: Path) -> None:
    db_file = str(tmp_path / "persistent_vectors.db")

    store1 = SQLiteVectorStore(db_path=db_file)
    store1.add("vec-p1", [0.5, 0.5, 0.0], metadata={"tag": "persistent"})
    store1.close()

    store2 = SQLiteVectorStore(db_path=db_file)
    assert store2.count() == 1
    vec = store2.get("vec-p1")
    assert vec is not None
    assert vec.embedding == [0.5, 0.5, 0.0]
    assert vec.metadata == {"tag": "persistent"}
    store2.close()
