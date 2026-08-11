# Mind Graph DB - Provenance & Value Proposition Guide

## Why Mind Graph DB Exists: Knowledge You Can Inspect & Trust

Traditional graph databases (such as Neo4j) store edges as simple structural connections without built-in evidence context. Traditional vector databases (such as Pinecone) return nearest-neighbor document chunks without explaining how concepts relate structurally.

**Mind Graph DB** bridges this gap by making **Evidence Provenance** a first-class database primitive.

---

## The `ProvenanceRecord` Primitive

Every relationship returned by Mind Graph DB includes a `ProvenanceRecord`:

```python
class ProvenanceRecord(BaseModel):
    kind: RelationshipKind        # EXPLICIT, INFERRED, SIMILAR_TO, CO_OCCURS_WITH, CONTRADICTS, SUPPORTS
    source_doc_id: Optional[str]  # Source document ID
    evidence_text: Optional[str]  # Quoted supporting sentence text
    char_span: Optional[Tuple[int, int]]  # Character span offset [start, end]
    confidence: float             # Confidence rating 0.0 to 1.0
    timestamp: datetime           # ISO timestamp
    method: str                   # Extraction method (e.g. nlp_explicit_extraction)
```

---

## Strict Graph Semantic Rules

1. **Explicit Text Mentions (`MENTIONS`)**:
   - MUST ONLY be created when the entity explicitly appears in document text (`confidence=1.0`).
   - Includes exact character span offsets `[start, end]` and quoted sentence evidence.
   - Vector similarity is **NEVER** silently converted into explicit factual `MENTIONS`.

2. **Vector Discovery (`SIMILAR_TO`)**:
   - Vector similarity candidate search is strictly isolated to `SIMILAR_TO` edges.

3. **Symmetric Co-occurrence (`CO_OCCURS_WITH`)**:
   - Created between distinct entities co-occurring in the same sentence.
   - Saved with canonical pair ordering `min(src_id, tgt_id), max(src_id, tgt_id)` to prevent reverse-direction duplicates.

4. **Contradiction Detection (`CONTRADICTS`)**:
   - Conflicting statements across documents are captured as `CONTRADICTS` relationships and inspectable via `client.find_contradictions()`.
