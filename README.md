# Mind Graph DB

**Mind Graph DB** is a new hybrid semantic + vector + graph database built in Python.

It seamlessly combines unstructured text storage, automated vector embedding generation, open-source NLP entity and relationship extraction, and graph traversal into a single unified query engine.

---

## Key Concepts & Architecture

Mind Graph DB is architected around three foundational storage tiers and two model abstractions, coordinated by a central query engine:

```
                      +-----------------------------+
                      |      Client Application     |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |         QueryEngine         |
                      +--------------+--------------+
                                     |
        +----------------------------+----------------------------+
        |                            |                            |
        v                            v                            v
+---------------+            +---------------+            +---------------+
| DocumentStore |            |  VectorStore  |            |  GraphStore   |
| (Raw Text &   |            |  (Embeddings  |            |  (Entities &  |
|  Metadata)    |            |   & ANN)      |            |  Relations)   |
+---------------+            +---------------+            +---------------+
        ^                            ^                            ^
        |                            |                            |
        +------------------+---------+----------+-----------------+
                           |                    |
                           v                    v
                   +---------------+    +---------------+
                   | EmbeddingModel|    |   NLPModel    |
                   | (Dense Vecs)  |    | (NER & RE)    |
                   +---------------+    +---------------+
```

### Component Breakdown

1. **`DocumentStore`**: Handles raw text persistence, metadata indexing, and document lifecycle management.
2. **`VectorStore`**: Manages high-dimensional dense vectors for semantic similarity searches.
3. **`GraphStore`**: Maintains nodes (entities) and directed edge relationships for multi-hop graph queries.
4. **`EmbeddingModel`**: Abstract interface for text embedding models (e.g. Sentence Transformers, HuggingFace transformers).
5. **`NLPModel`**: Abstract interface for Natural Language Processing (Named Entity Recognition & Relation Extraction).
6. **`QueryEngine`**: Unified query interface that orchestrates vector retrieval, graph expansion, and text filtering.

---

## Document Storage Layer (Step 2)

The Document Storage Layer manages raw unstructured text documents and metadata behind the abstract `DocumentStore` interface.

### Domain Model: `Document`
- **`id`**: Unique string identifier (automatically generated via UUIDv4 if omitted).
- **`text`**: Non-empty text string (validated on instantiation).
- **`metadata`**: JSON-serializable key-value pair dictionary.
- **`created_at`**: UTC timestamp recording initial storage creation time.
- **`updated_at`**: UTC timestamp recording last modification time.

### Storage Engine: `SQLiteDocumentStore`
A lightweight, zero-dependency local disk persistent storage engine using SQLite (`sqlite3`):
- **Database Table Schema**:
  ```sql
  CREATE TABLE IF NOT EXISTS documents (
      id TEXT PRIMARY KEY,
      text TEXT NOT NULL,
      metadata TEXT NOT NULL,
      created_at TEXT NOT NULL,
      updated_at TEXT NOT NULL
  );
  ```
- **CRUD Operations**: Full support for `put()` (insert/upsert), `get()`, `update()`, `delete()`, `list_documents()` (paginated), and `count()`.

---

## Embedding & Vector Storage Layer (Step 3)

The Embedding & Vector Storage Layer generates high-dimensional vector representations of document text and provides local vector similarity indexing.

### Embedding Models & Factory
- **`SentenceTransformerEmbeddingModel`**: Open-source pretrained models (e.g. `"all-MiniLM-L6-v2"`, `"bge-small-en-v1.5"`) via `sentence-transformers`.
- **`HashEmbeddingModel`**: Deterministic feature-hashing model for zero-dependency offline execution and fast testing.
- **`get_embedding_model()`**: Factory function retrieving configured embedding model according to application `Settings`.

### Persistent Vector Storage: `SQLiteVectorStore`
Vectors are stored in a dedicated local vector index separate from raw document text:
- **Database Table Schema**:
  ```sql
  CREATE TABLE IF NOT EXISTS vectors (
      id TEXT PRIMARY KEY,
      embedding TEXT NOT NULL,
      metadata TEXT NOT NULL
  );
  ```
- **Cosine Similarity Search**:
  Calculates exact cosine similarity scores:
  $$\text{similarity}(u, v) = \frac{u \cdot v}{\|u\| \|v\|}$$
- **Metadata Filtering**: Filters candidates by metadata key-value constraints prior to score ranking.

### Vector Service (`VectorService`)
Connects document text ingestion, vector embedding generation, and similarity search:

```python
from mind_graph_db.core.types import Document
from mind_graph_db.embeddings import get_embedding_model
from mind_graph_db.services import VectorService
from mind_graph_db.storage import SQLiteVectorStore

# Initialize components
embedding_model = get_embedding_model("all-MiniLM-L6-v2")
vector_store = SQLiteVectorStore(db_path="./storage/vectors.db")
service = VectorService(embedding_model=embedding_model, vector_store=vector_store)

# Embed & store document vector separately
doc = Document(id="doc-1", text="Vector search retrieves semantically similar documents.")
service.embed_and_store_document(doc)

# Search similar documents (returns list of (document_id, similarity_score))
matches = service.search_similar_documents(query_text="semantic vector retrieval", top_k=5)
for doc_id, score in matches:
    print(f"Doc ID: {doc_id}, Score: {score:.4f}")

vector_store.close()
```

---

## Quick Start

### Installation

Clone the repository and install in editable mode with development dependencies:

```bash
pip install -e .[dev]
```

### Running Tests

Execute the test suite using `pytest`:

```bash
pytest
```

Or run static type checking with `mypy`:

```bash
mypy src tests
```

## Technical Documentation

Detailed architecture specifications, sequence diagrams, database DDL schemas, and step-by-step guides are maintained in the [`docs/`](file:///f:/SAAS/mind-graph-db/docs/INDEX.md) folder:

- **[Architecture Index & Master Overview](file:///f:/SAAS/mind-graph-db/docs/INDEX.md)**: System sequence diagrams and architecture map.
- **[Step 1: Foundation Technical Spec](file:///f:/SAAS/mind-graph-db/docs/01_foundation.md)**: Project layout, ABC interfaces, Pydantic settings, and domain types.
- **[Step 2: Document Storage Technical Spec](file:///f:/SAAS/mind-graph-db/docs/02_document_store.md)**: Document model validation, SQLite document storage engine, and CRUD data flow.
- **[Step 3: Embedding & Vector Storage Technical Spec](file:///f:/SAAS/mind-graph-db/docs/03_embedding_vector_store.md)**: Pretrained embedding models, SQLite vector index, cosine similarity math, and `VectorService`.
- **[Step 4: Semantic Analysis Layer Technical Spec](file:///f:/SAAS/mind-graph-db/docs/04_semantic_analysis.md)**: Pretrained NLP models, NER entity extraction, SVO facts, relationship candidates, and `SemanticAnalysisService`.
- **[Step 5: Custom Graph Storage Layer Technical Spec](file:///f:/SAAS/mind-graph-db/docs/05_graph_store.md)**: Custom `SQLiteGraphStore` local graph engine, Entity & Document nodes, relationship edges with evidence provenance, and multi-hop BFS traversal.
- **[Step 6: Automatic Knowledge Graph Discovery Technical Spec](file:///f:/SAAS/mind-graph-db/docs/06_ingestion_pipeline.md)**: Automated 10-step `DocumentIngestionPipeline`, vector candidate discovery, entity resolution, and evidence provenance logging.
- **[Step 7: Document Updates & Stale Relationship Cleanup Technical Spec](file:///f:/SAAS/mind-graph-db/docs/07_document_updates.md)**: `DocumentIngestionPipeline.update_document`, embedding vector re-indexing, graph relationship reconciliation, and obsolete edge pruning.
- **[Step 8: Custom Domain Query Language & Execution Engine Technical Spec](file:///f:/SAAS/mind-graph-db/docs/08_query_engine.md)**: Custom domain query language parser, AST, 4-step query planner, and `MindQueryEngine`.
- **[Step 9: Python Client SDK & FastAPI Web Service Technical Spec](file:///f:/SAAS/mind-graph-db/docs/09_sdk_and_api.md)**: FastAPI REST endpoints, OpenAPI schemas, and `MindGraphDBClient` Python SDK.
- **[Step 10: End-to-End Demonstration & Complete Architecture Technical Spec](file:///f:/SAAS/mind-graph-db/docs/10_end_to_end_demonstration.md)**: Runnable E2E demo script, integration test suite, and `GraphVisualizer` debugging utility.
- **[User Manual & Complete Usage Guide](file:///f:/SAAS/mind-graph-db/docs/USER_MANUAL.md)**: Comprehensive user manual covering installation, Python SDK, REST API, Domain Query Language, Document Updates, and Graph Visualizer.

---

## Project Layout

```
mind-graph-db/
├── pyproject.toml
├── README.md
├── examples/        # Runnable end-to-end demonstration scripts
│   └── demo.py
├── docs/            # Complete step-by-step technical documentation & diagrams
│   ├── INDEX.md
│   ├── USER_MANUAL.md
│   ├── 01_foundation.md
│   ├── 02_document_store.md
│   ├── 03_embedding_vector_store.md
│   ├── 04_semantic_analysis.md
│   ├── 05_graph_store.md
│   ├── 06_ingestion_pipeline.md
│   ├── 07_document_updates.md
│   ├── 08_query_engine.md
│   ├── 09_sdk_and_api.md
│   └── 10_end_to_end_demonstration.md







├── src/
│   └── mind_graph_db/
│       ├── config/          # Application settings & environment parsing
│       ├── logging/         # Logging infrastructure
│       ├── core/            # Domain models (Document, Entity, Relation, Vector)
│       ├── interfaces/      # Storage, Model, and Query engine interfaces (ABCs)
│       ├── embeddings/      # Embedding models & factory (SentenceTransformer, Hash)
│       ├── storage/         # Storage engines (SQLiteDocumentStore, SQLiteVectorStore)
│       └── services/        # Service orchestrators (VectorService)
└── tests/                   # Pytest suite & verification
```

---

## License

MIT License.

