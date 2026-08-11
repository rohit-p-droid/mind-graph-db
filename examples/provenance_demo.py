"""Mind Graph DB - Provenance-First Hybrid Knowledge Database Demonstration.

Showcases 'Knowledge You Can Inspect & Trust':
- Evidence-backed relationships with explicit ProvenanceRecord metadata
- Graph health & diagnostics checks
- Explainable hybrid queries returning reasoning paths
"""

from mind_graph_db.sdk import MindGraphDBClient


def main() -> None:
    client = MindGraphDBClient(db_dir="./storage")

    # Clear previous database state
    client.delete_all()

    print("--- 1. Ingesting Enterprise Knowledge Documents ---")
    doc1 = client.create_document(
        doc_id="doc1",
        text="Machine learning is a branch of artificial intelligence that enables computers to learn from data.",
        metadata={"category": "Machine Learning", "author": "Andrew Ng"},
    )
    print(f"Ingested Document '{doc1.document_id}'")

    doc2 = client.create_document(
        doc_id="doc2",
        text="Deep learning is a subset of machine learning that uses neural networks with multiple layers.",
        metadata={"category": "Deep Learning", "author": "Ian Goodfellow"},
    )
    print(f"Ingested Document '{doc2.document_id}'")

    print("\n--- 2. Inspecting Evidence Provenance Records ---")
    relationships = client.get_relationships()
    print(f"Total Discovered Graph Relationships: {len(relationships)}")
    for i, rel in enumerate(relationships[:5], 1):
        explanation = client.explain_relationship(rel.id)
        if explanation:
            print(f"\n[{i}] Edge: '{explanation['source']}' --[{explanation['relation_type']}]--> '{explanation['target']}'")
            print(f"    Confidence: {explanation['confidence']:.2f}")
            print(f"    Evidence Text: '{explanation['evidence_text']}'")
            if explanation['provenance']:
                prov = explanation['provenance']
                print(f"    Provenance Kind: {prov.get('kind')}")
                print(f"    Extraction Method: {prov.get('method')}")
                print(f"    Char Span Offset: {prov.get('char_span')}")

    print("\n--- 3. Running Automated Graph Health Diagnostics ---")
    health = client.check_health()
    print(f"Graph Health Score: {health.health_score}% | Status: {health.status}")
    print(f"Nodes: {health.total_nodes} (Documents: {health.total_documents}, Entities: {health.total_entities})")
    print(f"Relationships by Kind: {health.relationship_kinds}")
    if health.diagnostics:
        print(f"Diagnostics: {health.diagnostics}")

    print("\n--- 4. Explainable Hybrid Query with Reasoning Paths ---")
    query_res = client.query("machine learning", top_k=2)
    print(f"Query: '{query_res.query}'")
    for path in query_res.reasoning_paths:
        print(f"\nDocument ID: {path['document_id']} | Relevance Score: {path['relevance_score']:.3f}")
        print(f"Snippet: '{path['evidence_snippet']}'")
        print(f"Supporting Relationships Count: {len(path['supporting_relationships'])}")


if __name__ == "__main__":
    main()
