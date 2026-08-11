"""Security robustness, injection vector protection, and API quality boundary tests."""

from pathlib import Path
from mind_graph_db.sdk import MindGraphDBClient


def test_sql_injection_and_label_sanitization(tmp_path: Path) -> None:
    client = MindGraphDBClient(db_dir=str(tmp_path / "sec_db"))
    client.delete_all()

    # Document text containing SQL injection strings
    malicious_text = "Machine learning '; DROP TABLE nodes; -- is artificial intelligence."
    res = client.create_document(doc_id="sec-doc-1", text=malicious_text)
    assert res.document_id == "sec-doc-1"

    # Query with SQL injection string
    query_res = client.query("'; DROP TABLE relationships; --")
    assert query_res is not None

    # Assert nodes and relationships tables remain intact
    health = client.check_health()
    assert health.status == "HEALTHY"


def test_nonexistent_relationship_explanation(tmp_path: Path) -> None:
    client = MindGraphDBClient(db_dir=str(tmp_path / "sec_db2"))
    client.delete_all()

    explanation = client.explain_relationship("invalid-nonexistent-id")
    assert explanation is None
