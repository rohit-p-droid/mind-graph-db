from mind_graph_db.sdk import MindGraphDBClient

client = MindGraphDBClient(db_dir="./storage")

document1 = client.create_document(
    text="""Tesla is expanding its manufacturing operations in Germany.
The company is planning a new battery manufacturing facility in Berlin.""",
    metadata={
        "source": "test",
        "category": "company"
    }
)

docs = client.semantic_search("expanding", top_k=5)
print("Semantic Search Results:", docs)

# Generate interactive visual graph HTML file
html_path = client.visualize(output_html_path="graph_visualization.html", auto_open=False)
print(f"Visual Graph HTML generated at: {html_path}")

