"""FastAPI router declaring REST endpoints for Mind Graph DB."""

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field


from mind_graph_db.api.deps import DatabaseContainer, get_database
from mind_graph_db.core.types import Document, QueryResult, TraversalPath
from mind_graph_db.pipeline import IngestionResult

router = APIRouter()


class CreateDocumentRequest(BaseModel):
    id: Optional[str] = None
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class UpdateDocumentRequest(BaseModel):
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SearchRequest(BaseModel):
    query: str
    top_k: int = 10


class QueryRequest(BaseModel):
    query: str
    top_k: int = 10


class TraverseRequest(BaseModel):
    start_node_id: str
    max_hops: int = 2


@router.get("/documents", response_model=List[Document])
def list_documents(
    limit: int = 100,
    offset: int = 0,
    db: DatabaseContainer = Depends(get_database),
) -> List[Document]:
    """Retrieve a paginated list of all stored documents."""
    return db.document_store.list_documents(limit=limit, offset=offset)


@router.get("/graph")
def get_graph_data(
    db: DatabaseContainer = Depends(get_database),
) -> Dict[str, Any]:
    """Retrieve live graph structure as JSON for UI visualizer."""
    from mind_graph_db.utils import GraphVisualizer
    return GraphVisualizer.export_json(db.graph_store)


@router.post("/documents", response_model=IngestionResult, status_code=status.HTTP_201_CREATED)
def create_document(
    req: CreateDocumentRequest,
    db: DatabaseContainer = Depends(get_database),
) -> IngestionResult:
    """Create and ingest a new document through the automatic discovery pipeline."""
    doc = Document(id=req.id, text=req.text, metadata=req.metadata) if req.id else Document(text=req.text, metadata=req.metadata)
    return db.pipeline.ingest_document(doc)


@router.get("/documents/{document_id}", response_model=Document)
def get_document(
    document_id: str,
    db: DatabaseContainer = Depends(get_database),
) -> Document:
    """Retrieve a document by unique ID."""
    doc = db.document_store.get(document_id)
    if doc is None:
        raise HTTPException(status_code=404, detail=f"Document '{document_id}' not found.")
    return doc


@router.put("/documents/{document_id}", response_model=IngestionResult)
def update_document(
    document_id: str,
    req: UpdateDocumentRequest,
    db: DatabaseContainer = Depends(get_database),
) -> IngestionResult:
    """Update an existing document, regenerate embeddings, and reconcile graph relationships."""
    existing_doc = db.document_store.get(document_id)
    if existing_doc is None:
        raise HTTPException(status_code=404, detail=f"Document '{document_id}' not found.")

    existing_doc.text = req.text
    if req.metadata:
        existing_doc.metadata.update(req.metadata)

    return db.pipeline.update_document(existing_doc)


@router.delete("/documents/{document_id}")
def delete_document(
    document_id: str,
    db: DatabaseContainer = Depends(get_database),
) -> Dict[str, Any]:
    """Delete a document and clean up associated vectors and graph nodes."""
    doc = db.document_store.get(document_id)
    if doc is None:
        raise HTTPException(status_code=404, detail=f"Document '{document_id}' not found.")

    db.document_store.delete(document_id)
    db.vector_store.delete(document_id)
    db.graph_store.delete_node(document_id)

    return {"status": "deleted", "document_id": document_id}


@router.delete("/documents")
def delete_all_documents(
    db: DatabaseContainer = Depends(get_database),
) -> Dict[str, Any]:
    """Delete all documents, vectors, and graph nodes in the database."""
    docs = db.document_store.list_documents(limit=10000)
    count = len(docs)
    for doc in docs:
        db.document_store.delete(doc.id)
        db.vector_store.delete(doc.id)
    if hasattr(db.graph_store, "clear"):
        db.graph_store.clear()
    else:
        for doc in docs:
            db.graph_store.delete_node(doc.id)

    return {"status": "deleted_all", "count": count}


@router.post("/search")
def semantic_search(
    req: SearchRequest,
    db: DatabaseContainer = Depends(get_database),
) -> List[Dict[str, Any]]:
    """Perform semantic vector similarity search."""
    query_vec = db.embedding_model.encode_text(req.query)
    results = db.vector_store.search(query_vec, top_k=req.top_k)
    output: List[Dict[str, Any]] = []
    for vec, score in results:
        doc = db.document_store.get(vec.id)
        output.append(
            {
                "document_id": vec.id,
                "score": float(score),
                "text": doc.text if doc else "",
                "metadata": doc.metadata if doc else {},
            }
        )
    return output


@router.post("/query", response_model=QueryResult)
def raw_query(
    req: QueryRequest,
    db: DatabaseContainer = Depends(get_database),
) -> QueryResult:
    """Submit a Mind Graph DB domain query string."""
    return db.query_engine.query(req.query, top_k=req.top_k)


@router.post("/graph/traverse", response_model=List[TraversalPath])
def traverse_graph(
    req: TraverseRequest,
    db: DatabaseContainer = Depends(get_database),
) -> List[TraversalPath]:
    """Perform multi-hop Breadth-First Search graph traversal from a starting node."""
    node = db.graph_store.get_node(req.start_node_id)
    if node is None:
        raise HTTPException(status_code=404, detail=f"Starting node '{req.start_node_id}' not found.")
    return db.graph_store.traverse(req.start_node_id, max_hops=req.max_hops)


@router.get("/graph/relationships/{relationship_id}/evidence")
def get_relationship_evidence(
    relationship_id: str,
    db: DatabaseContainer = Depends(get_database),
) -> Dict[str, Any]:
    """Retrieve evidence provenance for a relationship edge."""
    # Query graph store neighbors or edges for match
    subgraph = db.graph_store.query_subgraph([], depth=1)
    for rel in subgraph["relationships"]:
        if rel.id == relationship_id:
            return {
                "relationship_id": rel.id,
                "source_id": rel.source_id,
                "target_id": rel.target_id,
                "relation_type": rel.relation_type,
                "confidence": rel.confidence,
                "evidence_text": rel.evidence_text,
                "properties": rel.properties,
            }
    raise HTTPException(status_code=404, detail=f"Relationship '{relationship_id}' not found.")


@router.get("/health/graph")
def check_graph_health(
    db: DatabaseContainer = Depends(get_database),
) -> Dict[str, Any]:
    """Run automated graph health diagnostics, checking orphan nodes, provenance, and integrity."""
    from mind_graph_db.utils.health import GraphHealthChecker
    report = GraphHealthChecker.check_health(db.graph_store, db.document_store)
    return report.model_dump(mode="json")


@router.get("/visualize", response_class=HTMLResponse)
def get_graph_visualization(
    db: DatabaseContainer = Depends(get_database),
) -> HTMLResponse:
    """Render interactive web application graph visualizer dashboard."""
    from fastapi.responses import HTMLResponse
    from mind_graph_db.utils import GraphVisualizer

    html_str = GraphVisualizer.export_html(db.graph_store, output_path=None)
    return HTMLResponse(content=html_str)

