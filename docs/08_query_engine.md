# Step 8: Custom Domain Query Language & Execution Engine Technical Specification

This document details the **Custom Domain-Specific Query Language & Query Execution Engine** (`MindQueryEngine`) implemented in Step 8.

---

## 1. Query Engine Architecture

The Query Engine parses custom domain query strings, compiles them into optimized multi-stage execution plans, executes hybrid vector search, metadata filtering, and multi-hop graph traversal, and ranks results.

```mermaid
graph TD
    User[Client Application] --> Parser[QueryParser & QueryLexer]
    Parser --> AST[MindQuery AST]
    AST --> Planner[QueryPlanner]
    Planner --> Plan[QueryPlan Execution Steps]
    
    subgraph Execution Steps
        Plan --> Step1[1. VectorSearchStep]
        Plan --> Step2[2. MetadataFilterStep]
        Plan --> Step3[3. GraphTraverseStep]
        Plan --> Step4[4. RankAndAssembleStep]
    end
    
    Step1 --> VectorStore[(VectorStore)]
    Step2 --> DocumentStore[(DocumentStore)]
    Step3 --> GraphStore[(GraphStore)]
    Step4 --> Result[QueryResult Response]
```

---

## 2. Query Language EBNF Grammar

```ebnf
Query ::= FindClause WhereClause? TraverseClause? ReturnClause?

FindClause ::= "FIND" Target
Target     ::= "documents" | "entities" | "subgraph"

WhereClause ::= "WHERE" ConditionExpr
ConditionExpr ::= Condition ("AND" Condition)*
Condition   ::= SemanticMatchCond | MetadataCond

SemanticMatchCond ::= "semantic_match" "(" STRING_LITERAL ")"
MetadataCond     ::= "metadata." IDENTIFIER "=" (STRING_LITERAL | NUMBER)

TraverseClause ::= "TRAVERSE" INT_LITERAL "HOPS" ("WITH RELATIONSHIPS" StringList)?
ReturnClause   ::= "RETURN" ReturnList
ReturnList     ::= ReturnItem ("," ReturnItem)*
ReturnItem     ::= "documents" | "entities" | "relationships" | "scores"
```

---

## 3. Query Execution Sequence Diagram

```mermaid
sequenceDiagram
    autonumber
    actor App as Client Application
    participant Engine as MindQueryEngine
    participant Parser as QueryParser
    participant Planner as QueryPlanner
    participant VS as VectorStore
    participant DS as DocumentStore
    participant GS as GraphStore

    App->>Engine: query("FIND documents WHERE semantic_match('...') AND metadata.category = 'EV' TRAVERSE 2 HOPS")
    Engine->>Parser: parse(query_text)
    Parser-->>Engine: MindQuery AST
    Engine->>Planner: create_plan(MindQuery AST)
    Planner-->>Engine: QueryPlan (4 Steps)
    
    Engine->>VS: search(query_embedding, top_k)
    VS-->>Engine: Candidate Vectors & Scores
    
    Engine->>DS: get(doc_id) & Apply Metadata Filters
    DS-->>Engine: Filtered Documents
    
    Engine->>GS: traverse(doc_id, max_hops=2)
    GS-->>Engine: GraphNodes & Relationship Edges
    
    Engine->>Engine: Rank Documents by Hybrid Scores & Assemble
    Engine-->>App: QueryResult
```

---

## 4. Usage Code Examples

```python
from mind_graph_db.embeddings import get_embedding_model
from mind_graph_db.query import MindQueryEngine
from mind_graph_db.storage import SQLiteDocumentStore, SQLiteGraphStore, SQLiteVectorStore

# 1. Initialize components
doc_store = SQLiteDocumentStore(db_path="./storage/documents.db")
vec_store = SQLiteVectorStore(db_path="./storage/vectors.db")
graph_store = SQLiteGraphStore(db_path="./storage/graph.db")
embedding_model = get_embedding_model("all-MiniLM-L6-v2")

# 2. Instantiate Query Engine
engine = MindQueryEngine(
    document_store=doc_store,
    vector_store=vec_store,
    graph_store=graph_store,
    embedding_model=embedding_model,
)

# 3. Submit custom domain query
query_str = (
    'FIND documents '
    'WHERE semantic_match("Tesla battery production") AND metadata.category = "EV" '
    'TRAVERSE 2 HOPS WITH RELATIONSHIPS ["MENTIONS", "SIMILAR_TO"] '
    'RETURN documents, entities, relationships'
)

result = engine.query(query_str, top_k=5)

print("Matching Documents:", [d.id for d in result.documents])
print("Entities:", [e.name for e in result.entities])
print("Traversed Edges:", [r.relation_type for r in result.relationships])

doc_store.close()
vec_store.close()
graph_store.close()
```
