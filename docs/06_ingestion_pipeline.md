# Step 6: Automatic Knowledge Graph Discovery & Ingestion Pipeline Technical Specification

This document details the **Automatic Knowledge Graph Discovery & Ingestion Pipeline** (`DocumentIngestionPipeline`) implemented in Step 6.

---

## 1. Automated Pipeline Architecture

The ingestion pipeline automates raw text document storage, vector embedding indexing, semantic extraction, vector candidate discovery, cross-document entity resolution, confidence scoring, evidence provenance logging, and knowledge graph persistence.

```mermaid
graph TD
    User[Client Application] --> Pipeline[DocumentIngestionPipeline]
    
    subgraph Ingestion Steps
        Pipeline --> Step1[1. Store Document text in DocumentStore]
        Step1 --> Step2[2. Encode & Store Vector in VectorStore]
        Step2 --> Step3[3. Extract Entities & Facts via NLPModel]
        Step3 --> Step4[4. Create DOCUMENT Node in GraphStore]
        Step4 --> Step5[5. Entity Resolution & Add MENTIONS Edges]
        Step5 --> Step6[6. Vector Similarity Candidate Search]
        Step6 --> Step7[7. Evaluate Candidate Documents]
        Step7 --> Step8[8. Compute Confidence Scores]
        Step8 --> Step9[9. Attach Evidence Provenance]
        Step9 --> Step10[10. Commit Edges to GraphStore]
    end

    Step10 --> Result[IngestionResult]
```

---

## 2. The 10-Step Ingestion Workflow

```mermaid
sequenceDiagram
    autonumber
    actor App as Client Application
    participant Pipeline as DocumentIngestionPipeline
    participant DS as SQLiteDocumentStore
    participant VS as SQLiteVectorStore
    participant GS as SQLiteGraphStore
    participant NLP as NLPModel
    participant EM as EmbeddingModel

    App->>Pipeline: ingest_document(Document)
    Pipeline->>DS: put(Document) [Step 1]
    
    Pipeline->>EM: encode_text(document.text) [Step 2]
    EM-->>Pipeline: embedding_vector
    Pipeline->>VS: add(document.id, embedding_vector) [Step 2]
    
    Pipeline->>NLP: analyze(document.text) [Step 3]
    NLP-->>Pipeline: SemanticAnalysisResult
    
    Pipeline->>GS: add_node(DOCUMENT node) [Step 4]
    Pipeline->>GS: add_node(ENTITY nodes) & add_relationship(MENTIONS) [Step 5]
    
    Pipeline->>VS: search(embedding_vector, top_k=5) [Step 6]
    VS-->>Pipeline: Candidate Vectors & Similarity Scores
    
    loop For Each Candidate Document
        Pipeline->>GS: Compare entities & Evaluate vector similarity [Step 7]
        Pipeline->>Pipeline: Compute confidence score [Step 8]
        Pipeline->>GS: add_relationship(SIMILAR_TO / MENTIONS) with evidence_text [Step 9 & 10]
    end

    Pipeline-->>App: IngestionResult
```

---

## 3. Candidate Discovery & Entity Resolution

### Vector Candidate Discovery ($O(1)$ ANN / Top-K Search)
Instead of performing an expensive $O(N)$ pairwise comparison against every document in the database, the pipeline queries `VectorStore.search(query_embedding, top_k=5)`. Only top candidate documents exceeding `min_confidence` are evaluated.

### Entity Resolution
Entities extracted from new documents are normalized to canonical labels. Entity nodes are inserted into `GraphStore` as `GraphNode(node_type="ENTITY")`. Incoming documents create `"MENTIONS"` edges targeting resolved entity nodes.

### Evidence Provenance Logging
Every relationship committed into `GraphStore` attaches explicit evidence text in its `evidence_text` attribute:
- **`MENTIONS` Edges**: `f"Document '{doc_id}' explicitly mentions entity '{entity_name}'"`
- **`SIMILAR_TO` Edges**: `f"Vector cosine similarity ({sim_score:.3f}) between doc '{doc_id}' and doc '{cand_doc_id}'"`
- **Cross-Document Entity Links**: `f"Entity '{entity_name}' resolved across vector similar doc '{cand_doc_id}' (similarity={sim_score:.3f})"`

---

## 4. Usage Code Examples

```python
from mind_graph_db.core.types import Document
from mind_graph_db.embeddings import get_embedding_model
from mind_graph_db.nlp import get_nlp_model
from mind_graph_db.pipeline import DocumentIngestionPipeline
from mind_graph_db.storage import SQLiteDocumentStore, SQLiteGraphStore, SQLiteVectorStore

# 1. Initialize storage & model components
doc_store = SQLiteDocumentStore(db_path="./storage/documents.db")
vec_store = SQLiteVectorStore(db_path="./storage/vectors.db")
graph_store = SQLiteGraphStore(db_path="./storage/graph.db")

embedding_model = get_embedding_model("all-MiniLM-L6-v2")
nlp_model = get_nlp_model("en_core_web_sm")

# 2. Instantiate pipeline
pipeline = DocumentIngestionPipeline(
    document_store=doc_store,
    vector_store=vec_store,
    graph_store=graph_store,
    embedding_model=embedding_model,
    nlp_model=nlp_model,
    min_confidence=0.6,
    candidate_top_k=5,
)

# 3. Ingest documents
doc1 = Document(text="Python is used by Google for AI applications.")
res1 = pipeline.ingest_document(doc1)

doc2 = Document(text="Python integrates with SQLite to store semantic data. OpenAI uses Python.")
res2 = pipeline.ingest_document(doc2)

print(f"Ingested Doc 2 | Relationships Created: {res2.relationships_created}")

# 4. Explore automatically discovered knowledge graph
paths = graph_store.traverse(doc1.id, max_hops=2)
for path in paths:
    print(f"Path Hop {path.hop_count}: {[n.label for n in path.nodes]}")

doc_store.close()
vec_store.close()
graph_store.close()
```
