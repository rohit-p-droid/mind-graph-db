"""Unit and integration tests for Python Client SDK (MindGraphDBClient)."""

from pathlib import Path
from mind_graph_db.api import create_app
from mind_graph_db.api.deps import DatabaseContainer
from mind_graph_db.embeddings import HashEmbeddingModel
from mind_graph_db.nlp import RuleBasedNLPModel
from mind_graph_db.sdk import MindGraphDBClient
from mind_graph_db.storage import SQLiteDocumentStore, SQLiteGraphStore, SQLiteVectorStore


def test_sdk_in_process_mode(tmp_path: Path) -> None:
    db_dir = str(tmp_path / "sdk_storage")
    doc_store = SQLiteDocumentStore(db_path=str(tmp_path / "sdk_storage" / "documents.db"))
    vec_store = SQLiteVectorStore(db_path=str(tmp_path / "sdk_storage" / "vectors.db"))
    graph_store = SQLiteGraphStore(db_path=str(tmp_path / "sdk_storage" / "graph.db"))
    container = DatabaseContainer(
        document_store=doc_store,
        vector_store=vec_store,
        graph_store=graph_store,
        embedding_model=HashEmbeddingModel(dim=64),
        nlp_model=RuleBasedNLPModel(),
    )

    client = MindGraphDBClient(container=container)

    # 1. Create document
    res_create = client.create_document(
        text="Mind Graph DB is built using Python and SQLite.",
        metadata={"author": "Dev"},
        doc_id="sdk-doc-1",
    )
    assert res_create.document_id == "sdk-doc-1"
    assert res_create.entities_extracted >= 1

    # 2. Get document
    doc = client.get_document("sdk-doc-1")
    assert doc is not None
    assert doc.text == "Mind Graph DB is built using Python and SQLite."

    # 3. Update document
    res_update = client.update_document(
        document_id="sdk-doc-1",
        text="Mind Graph DB is built using Python, SQLite, and FastAPI.",
        metadata={"author": "Dev", "version": 2},
    )
    assert res_update.document_id == "sdk-doc-1"

    # 4. Semantic Search
    search_results = client.semantic_search("Python SQLite", top_k=5)
    assert len(search_results) >= 1
    assert search_results[0]["document_id"] == "sdk-doc-1"

    # 5. Domain Query
    q_res = client.query("FIND documents WHERE semantic_match('Python')", top_k=5)
    assert len(q_res.documents) >= 1

    # 6. Graph Traversal
    paths = client.traverse("sdk-doc-1", max_hops=1)
    assert isinstance(paths, list)

    # 7. Delete Document
    deleted = client.delete_document("sdk-doc-1")
    assert deleted is True
    assert client.get_document("sdk-doc-1") is None
