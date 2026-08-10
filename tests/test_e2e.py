"""Full End-to-End Integration Test Suite for Mind Graph DB."""

from pathlib import Path

from mind_graph_db.core.types import Document, QueryResult
from mind_graph_db.embeddings import HashEmbeddingModel
from mind_graph_db.nlp import RuleBasedNLPModel
from mind_graph_db.pipeline import DocumentIngestionPipeline
from mind_graph_db.query import MindQueryEngine
from mind_graph_db.storage import SQLiteDocumentStore, SQLiteGraphStore, SQLiteVectorStore
from mind_graph_db.utils import GraphVisualizer


def test_end_to_end_mind_graph_db_workflow(tmp_path: Path) -> None:
    doc_store = SQLiteDocumentStore(db_path=str(tmp_path / "docs.db"))
    vec_store = SQLiteVectorStore(db_path=str(tmp_path / "vecs.db"))
    graph_store = SQLiteGraphStore(db_path=str(tmp_path / "graph.db"))
    embedding_model = HashEmbeddingModel(dim=128)
    nlp_model = RuleBasedNLPModel()

    pipeline = DocumentIngestionPipeline(
        document_store=doc_store,
        vector_store=vec_store,
        graph_store=graph_store,
        embedding_model=embedding_model,
        nlp_model=nlp_model,
        min_confidence=0.5,
    )

    query_engine = MindQueryEngine(
        document_store=doc_store,
        vector_store=vec_store,
        graph_store=graph_store,
        embedding_model=embedding_model,
    )

    # 1. Ingest initial documents
    d1 = Document(
        id="e2e-doc-1",
        text="Mind Graph DB is an open source hybrid database written in Python.",
        metadata={"category": "db", "author": "Alice"},
    )
    d2 = Document(
        id="e2e-doc-2",
        text="Python integrates with SQLite for embedded persistence. Google uses Python.",
        metadata={"category": "tech", "author": "Bob"},
    )

    res1 = pipeline.ingest_document(d1)
    res2 = pipeline.ingest_document(d2)

    assert res1.document_id == "e2e-doc-1"
    assert res2.document_id == "e2e-doc-2"
    assert doc_store.count() == 2
    assert vec_store.count() == 2

    # 2. Verify graph structure & visualizer
    ascii_out = GraphVisualizer.render_ascii(graph_store, seed_node_ids=["e2e-doc-1"], max_depth=2)
    assert "e2e-doc-1" in ascii_out
    assert "Nodes" in ascii_out

    json_export = GraphVisualizer.export_json(graph_store)
    assert "nodes" in json_export
    assert "relationships" in json_export

    dot_export = GraphVisualizer.export_dot(graph_store)
    assert "digraph MindGraphDB" in dot_export

    # 3. Update document content & verify stale edge pruning
    d2.text = "Python integrates with SQLite for embedded persistence. OpenAI uses Python."
    res_up = pipeline.update_document(d2)

    assert res_up.relationships_removed >= 1  # Google edge removed
    assert res_up.relationships_created >= 1  # OpenAI edge added

    # 4. Execute domain query
    q_str = (
        'FIND documents '
        'WHERE semantic_match("Python database") AND metadata.category = "db" '
        'TRAVERSE 2 HOPS '
        'RETURN documents, entities, relationships'
    )
    q_res = query_engine.query(q_str, top_k=5)

    assert isinstance(q_res, QueryResult)
    assert len(q_res.documents) >= 1
    assert q_res.documents[0].id == "e2e-doc-1"
    assert len(q_res.entities) >= 1
    assert len(q_res.relationships) >= 1

    # Verify evidence provenance text attached to relationship edges
    for rel in q_res.relationships:
        assert rel.evidence_text is not None

    doc_store.close()
    vec_store.close()
    graph_store.close()
