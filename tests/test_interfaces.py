"""Tests verifying abstract interface compliance and domain objects."""

from mind_graph_db.core.types import Document, Entity, QueryResult, Relationship, Vector
from mind_graph_db.interfaces.document_store import DocumentStore
from mind_graph_db.interfaces.embedding import EmbeddingModel
from mind_graph_db.interfaces.graph_store import GraphStore
from mind_graph_db.interfaces.nlp import NLPModel
from mind_graph_db.interfaces.query_engine import QueryEngine
from mind_graph_db.interfaces.vector_store import VectorStore


def test_domain_types_creation() -> None:
    doc = Document(text="Artificial Intelligence and Graph Databases")
    assert doc.id is not None
    assert doc.text == "Artificial Intelligence and Graph Databases"

    entity = Entity(name="Mind Graph DB", type="DATABASE")
    assert entity.id is not None
    assert entity.name == "Mind Graph DB"

    rel = Relationship(
        source_id="e1",
        target_id="e2",
        relation_type="DEPENDS_ON",
        confidence=0.95,
    )
    assert rel.source_id == "e1"
    assert rel.confidence == 0.95

    vec = Vector(id="v1", embedding=[0.1, 0.2, 0.3])
    assert vec.id == "v1"
    assert len(vec.embedding) == 3

    res = QueryResult(query="test query", documents=[doc], entities=[entity])
    assert res.query == "test query"
    assert len(res.documents) == 1
    assert len(res.entities) == 1


def test_document_store_interface(in_memory_doc_store: DocumentStore) -> None:
    doc = Document(text="Sample document content")
    doc_id = in_memory_doc_store.put(doc)
    assert doc_id == doc.id
    assert in_memory_doc_store.count() == 1

    retrieved = in_memory_doc_store.get(doc_id)
    assert retrieved is not None
    assert retrieved.text == "Sample document content"

    listed = in_memory_doc_store.list_documents(limit=10)
    assert len(listed) == 1

    deleted = in_memory_doc_store.delete(doc_id)
    assert deleted is True
    assert in_memory_doc_store.count() == 0


def test_vector_store_interface(in_memory_vector_store: VectorStore) -> None:
    in_memory_vector_store.add("vec1", [0.1, 0.2, 0.3, 0.4])
    assert in_memory_vector_store.count() == 1

    retrieved = in_memory_vector_store.get("vec1")
    assert retrieved is not None
    assert retrieved.embedding == [0.1, 0.2, 0.3, 0.4]

    results = in_memory_vector_store.search([0.1, 0.2, 0.3, 0.4], top_k=5)
    assert len(results) == 1
    assert results[0][0].id == "vec1"
    assert results[0][1] == 1.0

    assert in_memory_vector_store.delete("vec1") is True
    assert in_memory_vector_store.count() == 0


def test_graph_store_interface(in_memory_graph_store: GraphStore) -> None:
    e1 = Entity(id="e1", name="GraphDB", type="TECHNOLOGY")
    e2 = Entity(id="e2", name="Python", type="LANGUAGE")

    in_memory_graph_store.add_entity(e1)
    in_memory_graph_store.add_entity(e2)

    rel = Relationship(source_id="e1", target_id="e2", relation_type="WRITTEN_IN")
    in_memory_graph_store.add_relationship(rel)

    neighbors = in_memory_graph_store.get_neighbors("e1")
    assert len(neighbors) == 1
    assert neighbors[0][0].label == "Python"
    assert neighbors[0][1].relation_type == "WRITTEN_IN"

    subgraph = in_memory_graph_store.query_subgraph(["e1"])
    assert len(subgraph["nodes"]) == 1
    assert len(subgraph["relationships"]) == 1

    assert in_memory_graph_store.delete_entity("e1") is True
    assert in_memory_graph_store.get_entity("e1") is None



def test_embedding_model_interface(dummy_embedding_model: EmbeddingModel) -> None:
    assert dummy_embedding_model.dimension == 4

    vec = dummy_embedding_model.encode_text("test string")
    assert len(vec) == 4

    batch_vecs = dummy_embedding_model.encode_batch(["s1", "s2"])
    assert len(batch_vecs) == 2
    assert len(batch_vecs[0]) == 4


def test_nlp_model_interface(dummy_nlp_model: NLPModel) -> None:
    entities = dummy_nlp_model.extract_entities("Sample text")
    assert len(entities) == 1
    assert entities[0].name == "TestEntity"

    rels = dummy_nlp_model.extract_relationships("Sample text", entities)
    assert isinstance(rels, list)


def test_query_engine_interface(dummy_query_engine: QueryEngine) -> None:
    result = dummy_query_engine.query("What is Mind Graph DB?")
    assert isinstance(result, QueryResult)
    assert result.query == "What is Mind Graph DB?"
