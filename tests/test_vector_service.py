"""Integration tests for VectorService document embedding and similarity search."""

from pathlib import Path
from mind_graph_db.core.types import Document
from mind_graph_db.embeddings import HashEmbeddingModel
from mind_graph_db.services.vector_service import VectorService
from mind_graph_db.storage.sqlite_vector_store import SQLiteVectorStore


def test_vector_service_embed_and_search(tmp_path: Path) -> None:
    db_file = str(tmp_path / "service_vectors.db")
    vector_store = SQLiteVectorStore(db_path=db_file)
    embedding_model = HashEmbeddingModel(dim=128)

    service = VectorService(embedding_model=embedding_model, vector_store=vector_store)

    doc1 = Document(
        id="doc-python",
        text="Python is an interpreted high-level general-purpose programming language.",
        metadata={"category": "programming"},
    )
    doc2 = Document(
        id="doc-graph",
        text="Graph databases use graph structures for semantic queries with nodes and edges.",
        metadata={"category": "database"},
    )

    # Embed and store documents
    vec1 = service.embed_and_store_document(doc1)
    vec2 = service.embed_and_store_document(doc2)

    assert vec1.id == "doc-python"
    assert vec2.id == "doc-graph"
    assert vector_store.count() == 2

    # Verify vector is stored separately from document text
    stored_vec1 = vector_store.get("doc-python")
    assert stored_vec1 is not None
    assert stored_vec1.metadata["doc_id"] == "doc-python"

    # Perform similarity search returning document IDs and scores
    matches = service.search_similar_documents("Python programming language", top_k=2)
    assert len(matches) == 2
    top_doc_id, top_score = matches[0]
    assert top_doc_id == "doc-python"
    assert top_score > 0.0

    vector_store.close()
