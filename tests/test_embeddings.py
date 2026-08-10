"""Unit tests for embedding models and embedding factory."""

from mind_graph_db.embeddings import HashEmbeddingModel, get_embedding_model
from mind_graph_db.storage.sqlite_vector_store import _cosine_similarity


def test_hash_embedding_model_dimension() -> None:
    model = HashEmbeddingModel(dim=128)
    assert model.dimension == 128

    vec = model.encode_text("Hello world")
    assert len(vec) == 128

    batch = model.encode_batch(["Text one", "Text two"])
    assert len(batch) == 2
    assert len(batch[0]) == 128
    assert len(batch[1]) == 128


def test_hash_embedding_similarity_ordering() -> None:
    model = HashEmbeddingModel(dim=256)

    vec_db1 = model.encode_text("database relational sql query")
    vec_db2 = model.encode_text("database SQL queries tables")
    vec_food = model.encode_text("pizza pasta restaurant recipe")

    sim_similar = _cosine_similarity(vec_db1, vec_db2)
    sim_unrelated = _cosine_similarity(vec_db1, vec_food)

    assert sim_similar > sim_unrelated


def test_get_embedding_model_factory() -> None:
    # Testing fallback to HashEmbeddingModel when sentence-transformers is missing or for hash fallback
    model = get_embedding_model(fallback_to_hash=True)
    assert model.dimension > 0

    vec = model.encode_text("Test factory embedding")
    assert len(vec) == model.dimension
