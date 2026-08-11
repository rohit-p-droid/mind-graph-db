"""Demonstration script using MindGraphDBClient SDK for data operations."""

import os
import shutil
from mind_graph_db.sdk import MindGraphDBClient


def run_sdk_demo() -> None:
    demo_dir = "./demo_storage"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir, exist_ok=True)

    print("===========================================================")
    print("   MIND GRAPH DB - CLIENT SDK END-TO-END DEMONSTRATION")
    print("===========================================================")

    # 1. Initialize SDK Client in local in-process mode
    client = MindGraphDBClient(db_dir=demo_dir)
    print("\n--- Step 1: SDK Client Initialized ---")

    # 2. Add Rows (Create & Ingest Documents)
    print("\n--- Step 2: Adding Rows (Document Ingestion & Auto Graph Construction) ---")
    res1 = client.create_document(
        doc_id="doc-1",
        text="Mind Graph DB is an open source hybrid database written in Python.",
        metadata={"category": "database", "author": "Alice"},
    )
    res2 = client.create_document(
        doc_id="doc-2",
        text="Python integrates with SQLite engine for graph persistence. Google uses Python for backend services.",
        metadata={"category": "technology", "author": "Bob"},
    )
    res3 = client.create_document(
        doc_id="doc-3",
        text="Mind Graph DB provides vector embeddings and semantic graph search.",
        metadata={"category": "database", "author": "Charlie"},
    )

    print(f"Added Doc 1: Extracted {res1.entities_extracted} entities, created {res1.relationships_created} edges.")
    print(f"Added Doc 2: Extracted {res2.entities_extracted} entities, created {res2.relationships_created} edges.")
    print(f"Added Doc 3: Extracted {res3.entities_extracted} entities, created {res3.relationships_created} edges.")

    # 3. View Rows (Get Document by ID)
    print("\n--- Step 3: Viewing Rows (Retrieving Documents by ID) ---")
    doc1 = client.get_document("doc-1")
    doc2 = client.get_document("doc-2")

    if doc1:
        print(f"Fetched Doc 1 -> ID: '{doc1.id}' | Text: '{doc1.text}' | Metadata: {doc1.metadata}")
    if doc2:
        print(f"Fetched Doc 2 -> ID: '{doc2.id}' | Text: '{doc2.text}' | Metadata: {doc2.metadata}")

    # 4. Perform Semantic Search Across Rows
    print("\n--- Step 4: Semantic Vector Search ---")
    search_results = client.semantic_search(query_text="Python database", top_k=2)
    for idx, match in enumerate(search_results, 1):
        print(f"  {idx}. Doc ID: '{match['document_id']}' | Score: {match['score']:.4f} | Text: '{match['text']}'")

    # 5. Update Rows & Reconcile Relationships
    print("\n--- Step 5: Updating Rows (Updating Text & Metadata) ---")
    print("Updating Doc 2: Replacing 'Google' with 'OpenAI'...")
    res_update = client.update_document(
        document_id="doc-2",
        text="Python integrates with SQLite engine for graph persistence. OpenAI uses Python to train AI models.",
        metadata={"category": "technology", "author": "Bob", "version": 2},
    )
    print(f"Updated Doc 2: Removed {res_update.relationships_removed} obsolete edges, created {res_update.relationships_created} new edges.")

    updated_doc2 = client.get_document("doc-2")
    if updated_doc2:
        print(f"Verified Updated Doc 2 -> Text: '{updated_doc2.text}'")

    # 6. Hybrid Domain Query & Provenance Retrieval
    print("\n--- Step 6: Hybrid Domain Query Execution ---")
    query_str = 'FIND documents WHERE semantic_match("Python database") AND metadata.category = "database" TRAVERSE 2 HOPS RETURN documents, entities, relationships'
    print(f"Query: {query_str}")
    query_res = client.query(query_str, top_k=5)

    print(f"Matching Documents Found: {len(query_res.documents)}")
    print(f"Entities Discovered: {len(query_res.entities)}")
    print(f"Relationship Edges Traversed: {len(query_res.relationships)}")
    for r in query_res.relationships:
        print(f"  • Edge: '{r.source_id}' --({r.relation_type}, conf={r.confidence:.2f})--> '{r.target_id}'")
        if r.evidence_text:
            print(f"    Evidence Provenance: '{r.evidence_text}'")

    # 7. Export Interactive HTML Graph Visualization using Client SDK
    html_file = "examples/graph_visualization.html"
    saved_path = client.visualize(output_html_path=html_file, auto_open=False)
    print(f"\n--- Step 7: Visualization Exported via SDK to '{saved_path}' ---")

    client.close()
    print("\n===========================================================")
    print("      SDK DEMONSTRATION COMPLETED SUCCESSFULLY")
    print("===========================================================")


if __name__ == "__main__":
    run_sdk_demo()
