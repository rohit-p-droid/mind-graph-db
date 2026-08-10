"""End-to-End Demonstration Script for Mind Graph DB."""

import os
import shutil
from pathlib import Path

from mind_graph_db.core.types import Document
from mind_graph_db.embeddings import HashEmbeddingModel
from mind_graph_db.nlp import RuleBasedNLPModel
from mind_graph_db.pipeline import DocumentIngestionPipeline
from mind_graph_db.query import MindQueryEngine
from mind_graph_db.storage import SQLiteDocumentStore, SQLiteGraphStore, SQLiteVectorStore
from mind_graph_db.utils import GraphVisualizer


def run_demo() -> None:
    demo_dir = "./storage_demo"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir, exist_ok=True)

    print("===========================================================")
    print("      MIND GRAPH DB - END-TO-END DEMONSTRATION")
    print("===========================================================")

    # 1. Initialize Storage & Model Components
    doc_store = SQLiteDocumentStore(db_path=os.path.join(demo_dir, "documents.db"))
    vec_store = SQLiteVectorStore(db_path=os.path.join(demo_dir, "vectors.db"))
    graph_store = SQLiteGraphStore(db_path=os.path.join(demo_dir, "graph.db"))
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

    # 2. Ingest Multiple Documents (Step 1 - 5)
    print("\n--- Step 1: Ingesting Documents & Auto-Building Graph ---")

    doc1 = Document(
        id="doc-1",
        text="Mind Graph DB is an open source hybrid database written in Python.",
        metadata={"category": "database", "author": "Alice"},
    )
    doc2 = Document(
        id="doc-2",
        text="Python integrates with SQLite engine for graph persistence. Google uses Python for backend services.",
        metadata={"category": "technology", "author": "Bob"},
    )
    doc3 = Document(
        id="doc-3",
        text="Mind Graph DB provides vector embeddings and semantic graph search.",
        metadata={"category": "database", "author": "Charlie"},
    )

    res1 = pipeline.ingest_document(doc1)
    res2 = pipeline.ingest_document(doc2)
    res3 = pipeline.ingest_document(doc3)

    print(f"Doc 1 Ingested: {res1.entities_extracted} entities, {res1.relationships_created} edges created.")
    print(f"Doc 2 Ingested: {res2.entities_extracted} entities, {res2.relationships_created} edges created.")
    print(f"Doc 3 Ingested: {res3.entities_extracted} entities, {res3.relationships_created} edges created.")

    # 3. Render ASCII Graph Visualization
    print("\n--- Step 2: Knowledge Graph ASCII Visualization ---")
    ascii_graph = GraphVisualizer.render_ascii(graph_store, seed_node_ids=["doc-1", "doc-2"], max_depth=2)
    print(ascii_graph)

    # 4. Update Document & Stale Relationship Pruning (Step 6 & 7)
    print("\n--- Step 3: Updating Document & Reconciling Relationships ---")
    print("Updating Doc 2: Replacing 'Google' with 'OpenAI'...")
    doc2.text = "Python integrates with SQLite engine for graph persistence. OpenAI uses Python to train AI models."
    doc2.metadata["version"] = 2

    res_update = pipeline.update_document(doc2)
    print(f"Doc 2 Updated! Removed Edges: {res_update.relationships_removed} | Created Edges: {res_update.relationships_created} | Preserved Edges: {res_update.relationships_updated}")

    # 5. Execute Domain Query (Step 8 & 9)
    print("\n--- Step 4: Executing Domain Query with Evidence Provenance ---")
    query_str = (
        'FIND documents '
        'WHERE semantic_match("Python database") AND metadata.category = "database" '
        'TRAVERSE 2 HOPS '
        'RETURN documents, entities, relationships'
    )
    print(f"Query: {query_str}")
    query_res = query_engine.query(query_str, top_k=5)

    print(f"\nMatching Documents ({len(query_res.documents)} found):")
    for d in query_res.documents:
        score = query_res.scores.get(d.id, 0.0)
        print(f"  • ID: {d.id} | Score: {score:.3f} | Text: '{d.text}'")

    print(f"\nDiscovered Entity Nodes ({len(query_res.entities)} found):")
    for e in query_res.entities:
        print(f"  • ID: {e.id} | Name: '{e.name}' | Type: '{e.type}'")

    print(f"\nTraversed Relationship Edges & Evidence Provenance ({len(query_res.relationships)} found):")
    for r in query_res.relationships:
        print(f"  • Edge: '{r.source_id}' --({r.relation_type}, conf={r.confidence:.2f})--> '{r.target_id}'")
        if r.evidence_text:
            print(f"    Evidence Provenance: '{r.evidence_text}'")

    doc_store.close()
    vec_store.close()
    graph_store.close()
    print("\n===========================================================")
    print("      DEMONSTRATION COMPLETED SUCCESSFULLY")
    print("===========================================================")


if __name__ == "__main__":
    run_demo()
