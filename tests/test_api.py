"""Integration tests for FastAPI REST Web Service endpoints using TestClient."""

from pathlib import Path
from fastapi.testclient import TestClient
import pytest

from mind_graph_db.api import create_app
from mind_graph_db.api.deps import DatabaseContainer
from mind_graph_db.embeddings import HashEmbeddingModel
from mind_graph_db.nlp import RuleBasedNLPModel
from mind_graph_db.storage import SQLiteDocumentStore, SQLiteGraphStore, SQLiteVectorStore


@pytest.fixture
def test_client(tmp_path: Path) -> TestClient:
    db_dir = str(tmp_path / "api_storage")
    app = create_app(db_dir=db_dir)

    # Set custom test models for fast deterministic tests
    doc_store = SQLiteDocumentStore(db_path=str(tmp_path / "api_storage" / "documents.db"))
    vec_store = SQLiteVectorStore(db_path=str(tmp_path / "api_storage" / "vectors.db"))
    graph_store = SQLiteGraphStore(db_path=str(tmp_path / "api_storage" / "graph.db"))
    embedding_model = HashEmbeddingModel(dim=64)
    nlp_model = RuleBasedNLPModel()

    from mind_graph_db.api import deps
    deps._container_instance = DatabaseContainer(
        document_store=doc_store,
        vector_store=vec_store,
        graph_store=graph_store,
        embedding_model=embedding_model,
        nlp_model=nlp_model,
    )

    return TestClient(app)


def test_api_health_check(test_client: TestClient) -> None:
    resp = test_client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_api_document_crud_and_search(test_client: TestClient) -> None:
    # 1. Create document
    create_resp = test_client.post(
        "/api/v1/documents",
        json={"id": "api-doc-1", "text": "Python is used by Google for backend services.", "metadata": {"env": "test"}},
    )
    assert create_resp.status_code == 201
    res_data = create_resp.json()
    assert res_data["document_id"] == "api-doc-1"
    assert res_data["entities_extracted"] >= 1

    # 2. Get document
    get_resp = test_client.get("/api/v1/documents/api-doc-1")
    assert get_resp.status_code == 200
    doc_data = get_resp.json()
    assert doc_data["id"] == "api-doc-1"
    assert doc_data["text"] == "Python is used by Google for backend services."

    # 3. Update document
    put_resp = test_client.put(
        "/api/v1/documents/api-doc-1",
        json={"text": "Python is used by OpenAI to build AI models.", "metadata": {"env": "test", "ver": 2}},
    )
    assert put_resp.status_code == 200
    update_data = put_resp.json()
    assert update_data["document_id"] == "api-doc-1"

    # 4. Semantic Search
    search_resp = test_client.post("/api/v1/search", json={"query": "Python AI", "top_k": 5})
    assert search_resp.status_code == 200
    results = search_resp.json()
    assert len(results) >= 1
    assert results[0]["document_id"] == "api-doc-1"

    # 5. Raw Query
    query_resp = test_client.post(
        "/api/v1/query",
        json={"query": 'FIND documents WHERE semantic_match("Python") RETURN documents, entities', "top_k": 5},
    )
    assert query_resp.status_code == 200
    q_data = query_resp.json()
    assert len(q_data["documents"]) >= 1

    # 6. Delete document
    del_resp = test_client.delete("/api/v1/documents/api-doc-1")
    assert del_resp.status_code == 200
    assert del_resp.json()["status"] == "deleted"

    # Verify 404 after deletion
    assert test_client.get("/api/v1/documents/api-doc-1").status_code == 404
