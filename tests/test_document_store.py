"""Unit tests for Document domain model and SQLiteDocumentStore persistent storage."""

from pathlib import Path
import time
import pytest

from mind_graph_db.core.types import Document
from mind_graph_db.storage.sqlite_document_store import SQLiteDocumentStore


def test_document_model_validation() -> None:

    # Test auto UUID generation
    doc = Document(text="Valid document text")
    assert doc.id is not None
    assert len(doc.id) > 0
    assert doc.created_at is not None
    assert doc.updated_at is not None

    # Test empty text validation
    with pytest.raises(ValueError, match="cannot be empty"):
        Document(text="")

    with pytest.raises(ValueError, match="cannot be empty"):
        Document(text="   \n \t ")

    # Test touch method
    orig_updated = doc.updated_at
    time.sleep(0.01)
    doc.touch()
    assert doc.updated_at > orig_updated


def test_sqlite_document_store_crud(sqlite_doc_store: SQLiteDocumentStore) -> None:
    # Create (put)
    doc = Document(text="First test document", metadata={"author": "Alice", "version": 1})
    doc_id = sqlite_doc_store.put(doc)
    assert doc_id == doc.id
    assert sqlite_doc_store.count() == 1

    # Read (get)
    retrieved = sqlite_doc_store.get(doc_id)
    assert retrieved is not None
    assert retrieved.id == doc.id
    assert retrieved.text == "First test document"
    assert retrieved.metadata == {"author": "Alice", "version": 1}

    # Update
    retrieved.text = "Updated test document"
    retrieved.metadata["version"] = 2
    orig_updated = retrieved.updated_at
    time.sleep(0.01)

    updated_success = sqlite_doc_store.update(retrieved)
    assert updated_success is True

    after_update = sqlite_doc_store.get(doc_id)
    assert after_update is not None
    assert after_update.text == "Updated test document"
    assert after_update.metadata["version"] == 2
    assert after_update.updated_at > orig_updated

    # Delete
    deleted_success = sqlite_doc_store.delete(doc_id)
    assert deleted_success is True
    assert sqlite_doc_store.get(doc_id) is None
    assert sqlite_doc_store.count() == 0


def test_sqlite_document_store_persistence(tmp_path: Path) -> None:
    db_file = str(tmp_path / "persistent_docs.db")


    # Write document in first store instance
    store1 = SQLiteDocumentStore(db_path=db_file)
    doc = Document(id="doc-100", text="Persistent content", metadata={"key": "value"})
    store1.put(doc)
    store1.close()

    # Re-open store from same file
    store2 = SQLiteDocumentStore(db_path=db_file)
    assert store2.count() == 1
    retrieved = store2.get("doc-100")
    assert retrieved is not None
    assert retrieved.text == "Persistent content"
    assert retrieved.metadata == {"key": "value"}
    store2.close()


def test_sqlite_document_store_pagination_and_count(sqlite_doc_store: SQLiteDocumentStore) -> None:
    for i in range(5):
        doc = Document(text=f"Document number {i}", metadata={"idx": i})
        sqlite_doc_store.put(doc)

    assert sqlite_doc_store.count() == 5

    page1 = sqlite_doc_store.list_documents(limit=3, offset=0)
    assert len(page1) == 3

    page2 = sqlite_doc_store.list_documents(limit=3, offset=3)
    assert len(page2) == 2


def test_sqlite_document_store_edge_cases(sqlite_doc_store: SQLiteDocumentStore) -> None:
    # Get non-existent
    assert sqlite_doc_store.get("non-existent-id") is None

    # Update non-existent
    fake_doc = Document(id="fake-id", text="Some text")
    assert sqlite_doc_store.update(fake_doc) is False

    # Delete non-existent
    assert sqlite_doc_store.delete("non-existent-id") is False
