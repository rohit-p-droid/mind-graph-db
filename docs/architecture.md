# Mind Graph DB - System Architecture & Design

**Mind Graph DB** is a provenance-first, hybrid knowledge graph and vector database built on the core principle: **"Knowledge You Can Inspect & Trust"**.

---

## High-Level Architecture Diagram

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                   MindGraphDB Client                                   │
│                        (In-Process Python SDK or REST HTTP API Client)                  │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                 Mind Graph DB Core Engine                              │
│                                                                                        │
│   ┌────────────────────────┐  ┌─────────────────────────┐  ┌────────────────────────┐   │
│   │   Document Store       │  │      Vector Store       │  │      Graph Store       │   │
│   │ (SQLite / In-Memory)   │  │  (SQLite / Cosine SIM)  │  │  (SQLite Graph Engine) │   │
│   └───────────┬────────────┘  └────────────┬────────────┘  └───────────┬────────────┘   │
│               │                            │                           │                │
│               └────────────────────────────┼───────────────────────────┘                │
│                                            ▼                                            │
│   ┌────────────────────────────────────────────────────────────────────────────────┐   │
│   │                           10-Step Ingestion Pipeline                           │   │
│   │                                                                                │   │
│   │   1. Document Storage ──► 2. Vector Embedding ──► 3. NLP Extraction             │   │
│   │   4. Node Creation   ──► 5. Explicit MENTIONS  ──► 6. Vector SIMILAR_TO         │   │
│   │   7. Sentence CO_OCCURS_WITH ──► 8. Canonical Idempotency Commit               │   │
│   └────────────────────────────────────────┬───────────────────────────────────────┘   │
│                                            │                                            │
│                                            ▼                                            │
│   ┌────────────────────────────────────────────────────────────────────────────────┐   │
│   │                           Explainable MindQueryEngine                          │   │
│   │                                                                                │   │
│   │   - Hybrid Lexical & Vector Filtering                                          │   │
│   │   - Multi-Hop Graph Traversal (BFS)                                            │   │
│   │   - Reasoning Paths & Supporting Evidence Snippets                             │   │
│   └────────────────────────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Core Subsystems

### 1. Storage Subsystem (`src/mind_graph_db/storage/`)
- **`SQLiteDocumentStore`**: Handles document lifecycle, text storage, metadata JSON, and timestamps. Features thread-safe `threading.Lock()` concurrency protection.
- **`SQLiteVectorStore`**: Stores high-dimensional dense vector embeddings and executes cosine similarity searches with metadata filtering.
- **`SQLiteGraphStore`**: Manages GraphNodes (`ENTITY`, `DOCUMENT`) and Relationships (`MENTIONS`, `CO_OCCURS_WITH`, `SIMILAR_TO`, `CONTRADICTS`, `SUPPORTS`). Enforces `UNIQUE(source_id, target_id, relation_type)` and canonical pair ordering `min(src, tgt), max(src, tgt)` for symmetric edges.

### 2. Ingestion & NLP Subsystem (`src/mind_graph_db/pipeline/` & `nlp/`)
- **`DocumentIngestionPipeline`**: Orchestrates 10-step ingestion with idempotent updates and orphan node pruning upon document deletion.
- **`RuleBasedNLPModel`**: Multi-word entity resolution engine (`ENTITY_CANONICAL_MAP`) preventing entity fragment duplicates (e.g. `Machine Learning` vs `Machine`).
- **`LLMNLPModel`**: Pluggable structured extraction engine allowing custom LLM/SLM extractors while preserving evidence spans.

### 3. Query & Health Subsystem (`src/mind_graph_db/query/` & `utils/`)
- **`MindQueryEngine`**: Lexical AST query parser and hybrid search engine constructing `reasoning_paths` with relevance scores and evidence text snippets.
- **`GraphHealthChecker`**: Automated health checker calculating graph health score ($0-100\%$), identifying orphan entities, and reporting relationship breakdown by kind.
