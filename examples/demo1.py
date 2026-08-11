from mind_graph_db.sdk import MindGraphDBClient


client = MindGraphDBClient(
    db_dir="./storage"
)

# Start clean
client.delete_all()

doc1 = client.create_document(
    doc_id="doc1",
    text="Machine learning is a branch of artificial intelligence that enables computers to learn from data and make predictions without being explicitly programmed.",
    metadata={
        "title": "Introduction to Machine Learning",
        "author": "Andrew Ng",
        "category": "Machine Learning",
        "year": 2023
    }
)

doc2 = client.create_document(
    doc_id="doc2",
    text="Deep learning is a subset of machine learning that uses neural networks with multiple layers to learn complex patterns from large datasets.",
    metadata={
        "title": "Deep Learning Fundamentals",
        "author": "Ian Goodfellow",
        "category": "Deep Learning",
        "year": 2022
    }
)

doc3 = client.create_document(
    doc_id="doc3",
    text="Natural language processing enables computers to understand, process, and generate human language using computational and machine learning techniques.",
    metadata={
        "title": "Natural Language Processing",
        "author": "Christopher Manning",
        "category": "NLP",
        "year": 2021
    }
)

doc4 = client.create_document(
    doc_id="doc4",
    text="Computer vision allows machines to analyze and understand visual information from images and videos.",
    metadata={
        "title": "Computer Vision Basics",
        "author": "David Forsyth",
        "category": "Computer Vision",
        "year": 2020
    }
)

doc5 = client.create_document(
    doc_id="doc5",
    text="Vector databases store numerical representations of data and enable efficient similarity searches using embeddings.",
    metadata={
        "title": "Vector Database Overview",
        "author": "Test Author",
        "category": "Databases",
        "year": 2024
    }
)

documents = client.get_all_documents()
print(f"Total Stored Documents: {len(documents)}")

# Fetch and print all discovered graph relationships
relationships = client.get_relationships()
print(f"\n--- Discovered Graph Relationships ({len(relationships)} total) ---")
for rel in relationships:
    print(f"Edge: '{rel.source_id}' --[{rel.relation_type}, conf={rel.confidence:.2f}]--> '{rel.target_id}'")
    if rel.evidence_text:
        print(f"   Evidence Provenance: '{rel.evidence_text}'")

client.visualize(output_html_path="graph.html", auto_open=False)
