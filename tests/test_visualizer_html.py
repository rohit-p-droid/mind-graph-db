"""Unit and integration tests for HTML Graph Visualizer & REST endpoint."""

from pathlib import Path
from fastapi.testclient import TestClient

from mind_graph_db.api import create_app
from mind_graph_db.api.deps import DatabaseContainer
from mind_graph_db.core.types import Document
from mind_graph_db.embeddings import HashEmbeddingModel
from mind_graph_db.nlp import RuleBasedNLPModel
from mind_graph_db.pipeline import DocumentIngestionPipeline
from mind_graph_db.sdk import MindGraphDBClient
from mind_graph_db.storage import SQLiteDocumentStore, SQLiteGraphStore, SQLiteVectorStore
from mind_graph_db.utils import GraphVisualizer


def test_graph_visualizer_export_html(tmp_path: Path) -> None:
    doc_store = SQLiteDocumentStore(db_path=str(tmp_path / "docs.db"))
    vec_store = SQLiteVectorStore(db_path=str(tmp_path / "vecs.db"))
    graph_store = SQLiteGraphStore(db_path=str(tmp_path / "graph.db"))

    pipeline = DocumentIngestionPipeline(
        document_store=doc_store,
        vector_store=vec_store,
        graph_store=graph_store,
        embedding_model=HashEmbeddingModel(dim=32),
        nlp_model=RuleBasedNLPModel(),
    )

    pipeline.ingest_document(Document(id="vdoc-1", text="Python is used for machine learning."))

    html_file = tmp_path / "custom_graph.html"
    html_content = GraphVisualizer.export_html(graph_store, output_path=str(html_file))

    assert "Mind Graph DB Explorer" in html_content
    assert "vis.Network" in html_content
    assert "vdoc-1" in html_content
    assert html_file.exists()

    doc_store.close()
    vec_store.close()
    graph_store.close()


def test_sdk_visualize_method(tmp_path: Path) -> None:
    doc_store = SQLiteDocumentStore(db_path=str(tmp_path / "docs.db"))
    vec_store = SQLiteVectorStore(db_path=str(tmp_path / "vecs.db"))
    graph_store = SQLiteGraphStore(db_path=str(tmp_path / "graph.db"))
    container = DatabaseContainer(
        document_store=doc_store,
        vector_store=vec_store,
        graph_store=graph_store,
        embedding_model=HashEmbeddingModel(dim=32),
        nlp_model=RuleBasedNLPModel(),
    )

    client = MindGraphDBClient(container=container)
    client.create_document(text="Tesla builds EV cars in Berlin.")

    out_file = tmp_path / "sdk_graph.html"
    abs_path = client.visualize(output_html_path=str(out_file), auto_open=False)

    assert Path(abs_path).exists()
    content = Path(abs_path).read_text(encoding="utf-8")
    assert "Mind Graph DB Explorer" in content


def test_api_visualize_route(tmp_path: Path) -> None:
    app = create_app(db_dir=str(tmp_path / "api_vis"))
    test_client = TestClient(app)

    resp = test_client.get("/visualize")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "Mind Graph DB Explorer" in resp.text
