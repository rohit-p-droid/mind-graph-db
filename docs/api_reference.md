# Mind Graph DB - API Reference

## MindGraphDBClient (Python SDK)

```python
from mind_graph_db.sdk import MindGraphDBClient

client = MindGraphDBClient(db_dir="./storage")
```

### Methods

#### `create_document(doc_id: str, text: str, metadata: Optional[Dict[str, Any]] = None) -> IngestionResult`
Ingest a document into the database, running vector embedding, entity resolution, and provenance graph creation.

#### `query(query_string: str, top_k: int = 10) -> QueryResult`
Execute a hybrid search query, returning matching documents, entities, relationships, and explainable `reasoning_paths`.

#### `traverse(start_node_id: str, max_hops: int = 2) -> List[TraversalPath]`
Perform multi-hop BFS graph traversal starting from a node.

#### `check_health() -> GraphHealthReport`
Run automated graph health diagnostics returning graph health score ($0-100\%$), status, relationship counts by kind, and orphan entity lists.

#### `explain_relationship(relationship_id: str) -> Optional[Dict[str, Any]]`
Retrieve evidence text, character offsets, confidence, and provenance record for a graph edge.

#### `find_contradictions() -> List[Dict[str, Any]]`
Scan the graph for contradictory relationships (`CONTRADICTS`), returning evidence quotes and supporting context.

#### `delete_all()`
Purge all nodes, edges, vectors, and documents cleanly.

---

## REST API Specification

- `GET /health`: System health check.
- `GET /api/v1/health/graph`: Graph diagnostic health report.
- `POST /api/v1/documents`: Ingest document.
- `GET /api/v1/documents`: List documents.
- `DELETE /api/v1/documents`: Purge all database storage.
- `POST /api/v1/query`: Execute hybrid search query.
- `GET /api/v1/graph`: Retrieve nodes and relationships subgraph.
- `GET /api/v1/visualize`: Export interactive HTML graph visualization.
