"""Python Client SDK for Mind Graph DB supporting HTTP REST and in-process execution modes."""

from typing import Any, Dict, List, Optional
import httpx

from mind_graph_db.api.deps import DatabaseContainer, init_database
from mind_graph_db.core.types import Document, QueryResult, Relationship, TraversalPath
from mind_graph_db.pipeline import IngestionResult
from mind_graph_db.utils.health import GraphHealthChecker, GraphHealthReport


class MindGraphDBClient:
    """Python Client SDK for interacting with Mind Graph DB REST web service or in-process database engine."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        container: Optional[DatabaseContainer] = None,
        db_dir: str = "./storage",
    ) -> None:
        """Initialize MindGraphDBClient.

        Args:
            base_url: Optional base URL of remote FastAPI service (e.g. 'http://localhost:8000').
            container: Optional pre-initialized in-process DatabaseContainer instance.
            db_dir: Storage directory path for local in-process mode if base_url is None.
        """
        self.base_url = base_url
        self._http_client: Optional[httpx.Client] = None
        self._container: Optional[DatabaseContainer] = None

        if self.base_url:
            self._http_client = httpx.Client(base_url=self.base_url, timeout=30.0)
        else:
            self._container = container or init_database(db_dir=db_dir)


    def create_document(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        doc_id: Optional[str] = None,
    ) -> IngestionResult:
        """Create and ingest a new document."""
        meta = metadata or {}
        if self._http_client:
            resp = self._http_client.post(
                "/api/v1/documents",
                json={"id": doc_id, "text": text, "metadata": meta},
            )
            resp.raise_for_status()
            return IngestionResult(**resp.json())
        else:
            assert self._container is not None
            doc = Document(id=doc_id, text=text, metadata=meta) if doc_id else Document(text=text, metadata=meta)
            return self._container.pipeline.ingest_document(doc)

    def get_document(self, document_id: str) -> Optional[Document]:
        """Retrieve a document by ID."""
        if self._http_client:
            resp = self._http_client.get(f"/api/v1/documents/{document_id}")
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return Document(**resp.json())
        else:
            assert self._container is not None
            return self._container.document_store.get(document_id)

    def update_document(
        self,
        document_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> IngestionResult:
        """Update an existing document and reconcile graph relationships."""
        meta = metadata or {}
        if self._http_client:
            resp = self._http_client.put(
                f"/api/v1/documents/{document_id}",
                json={"text": text, "metadata": meta},
            )
            resp.raise_for_status()
            return IngestionResult(**resp.json())
        else:
            assert self._container is not None
            doc = self._container.document_store.get(document_id)
            if doc is None:
                raise ValueError(f"Document '{document_id}' not found.")
            doc.text = text
            if meta:
                doc.metadata.update(meta)
            return self._container.pipeline.update_document(doc)

    def delete_document(self, document_id: str) -> bool:
        """Delete a document and clean up vectors and graph nodes."""
        if self._http_client:
            resp = self._http_client.delete(f"/api/v1/documents/{document_id}")
            if resp.status_code == 404:
                return False
            resp.raise_for_status()
            return True
        else:
            assert self._container is not None
            doc = self._container.document_store.get(document_id)
            if doc is None:
                return False
            self._container.document_store.delete(document_id)
            self._container.vector_store.delete(document_id)
            self._container.graph_store.delete_node(document_id)
            return True

    def list_documents(self, limit: int = 100, offset: int = 0) -> List[Document]:
        """Retrieve a paginated list of all stored documents."""
        if self._http_client:
            resp = self._http_client.get(
                "/api/v1/documents",
                params={"limit": limit, "offset": offset},
            )
            resp.raise_for_status()
            return [Document(**d) for d in resp.json()]
        else:
            assert self._container is not None
            return self._container.document_store.list_documents(limit=limit, offset=offset)

    def get_all_documents(self) -> List[Document]:
        """Retrieve all documents stored in the database."""
        return self.list_documents(limit=10000, offset=0)

    def get_all(self) -> List[Document]:
        """Retrieve all documents stored in the database (alias for get_all_documents)."""
        return self.get_all_documents()

    def delete_all_documents(self) -> int:
        """Delete all documents, vectors, and graph nodes in the database."""
        if self._http_client:
            resp = self._http_client.delete("/api/v1/documents")
            resp.raise_for_status()
            data = resp.json()
            return int(data.get("count", 0))
        else:
            assert self._container is not None
            docs = self._container.document_store.list_documents(limit=10000)
            count = len(docs)
            for doc in docs:
                self._container.document_store.delete(doc.id)
                self._container.vector_store.delete(doc.id)
            if hasattr(self._container.graph_store, "clear"):
                self._container.graph_store.clear()
            else:
                for doc in docs:
                    self._container.graph_store.delete_node(doc.id)
            return count

    def delete_all(self) -> int:
        """Delete all documents, vectors, and graph nodes in the database (alias for delete_all_documents)."""
        return self.delete_all_documents()

    def semantic_search(self, query_text: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Perform semantic similarity vector search."""
        if self._http_client:
            resp = self._http_client.post(
                "/api/v1/search",
                json={"query": query_text, "top_k": top_k},
            )
            resp.raise_for_status()
            return list(resp.json())
        else:
            assert self._container is not None
            query_vec = self._container.embedding_model.encode_text(query_text)
            results = self._container.vector_store.search(query_vec, top_k=top_k)
            out: List[Dict[str, Any]] = []
            for vec, score in results:
                doc = self._container.document_store.get(vec.id)
                out.append(
                    {
                        "document_id": vec.id,
                        "score": float(score),
                        "text": doc.text if doc else "",
                        "metadata": doc.metadata if doc else {},
                    }
                )
            return out

    def query(self, query_str: str, top_k: int = 10) -> QueryResult:
        """Execute a Mind Graph DB domain query."""
        if self._http_client:
            resp = self._http_client.post(
                "/api/v1/query",
                json={"query": query_str, "top_k": top_k},
            )
            resp.raise_for_status()
            return QueryResult(**resp.json())
        else:
            assert self._container is not None
            return self._container.query_engine.query(query_str, top_k=top_k)

    def traverse(self, start_node_id: str, max_hops: int = 2) -> List[TraversalPath]:
        """Perform multi-hop graph traversal."""
        if self._http_client:
            resp = self._http_client.post(
                "/api/v1/graph/traverse",
                json={"start_node_id": start_node_id, "max_hops": max_hops},
            )
            resp.raise_for_status()
            return [TraversalPath(**p) for p in resp.json()]
        else:
            assert self._container is not None
            return self._container.graph_store.traverse(start_node_id, max_hops=max_hops)

    def get_graph(self, start_node_ids: Optional[List[str]] = None, depth: int = 5) -> Dict[str, Any]:
        """Retrieve graph nodes and relationships structure."""
        if self._http_client:
            resp = self._http_client.get("/api/v1/graph")
            resp.raise_for_status()
            return dict(resp.json())
        else:
            assert self._container is not None
            subgraph = self._container.graph_store.query_subgraph(start_node_ids or [], depth=depth)
            return {
                "nodes": subgraph.get("nodes", []),
                "relationships": subgraph.get("relationships", []),
            }

    def check_health(self) -> GraphHealthReport:
        """Run automated graph health diagnostics, checking orphan nodes, provenance, and integrity."""
        if self._http_client:
            resp = self._http_client.get("/api/v1/health/graph")
            resp.raise_for_status()
            return GraphHealthReport(**resp.json())
        else:
            assert self._container is not None
            return GraphHealthChecker.check_health(
                self._container.graph_store,
                self._container.document_store,
            )

    def validate_graph(self) -> GraphHealthReport:
        """Alias for check_health()."""
        return self.check_health()

    def explain_relationship(self, relationship_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve the evidence provenance record and supporting context for a relationship ID."""
        rels = self.get_relationships()
        target_rel = next((r for r in rels if r.id == relationship_id), None)
        if not target_rel:
            return None

        src_node = self._container.graph_store.get_node(target_rel.source_id) if self._container else None
        tgt_node = self._container.graph_store.get_node(target_rel.target_id) if self._container else None

        return {
            "relationship_id": target_rel.id,
            "source": src_node.label if src_node else target_rel.source_id,
            "target": tgt_node.label if tgt_node else target_rel.target_id,
            "relation_type": target_rel.relation_type,
            "confidence": target_rel.confidence,
            "evidence_text": target_rel.evidence_text,
            "provenance": target_rel.provenance.model_dump(mode="json") if target_rel.provenance else None,
        }

    def find_contradictions(self) -> List[Dict[str, Any]]:
        """Find contradictory relationships or conflicting facts in the knowledge graph."""
        rels = self.get_relationships()
        contradictions = []

        for rel in rels:
            is_contradict = (
                rel.relation_type.upper() == "CONTRADICTS"
                or (rel.provenance and str(rel.provenance.kind) in ("CONTRADICTS", "RelationshipKind.CONTRADICTS"))
            )
            if is_contradict:
                src_node = self._container.graph_store.get_node(rel.source_id) if self._container else None
                tgt_node = self._container.graph_store.get_node(rel.target_id) if self._container else None
                contradictions.append({
                    "relationship_id": rel.id,
                    "source": src_node.label if src_node else rel.source_id,
                    "target": tgt_node.label if tgt_node else rel.target_id,
                    "relation_type": rel.relation_type,
                    "confidence": rel.confidence,
                    "evidence_text": rel.evidence_text,
                    "provenance": rel.provenance.model_dump(mode="json") if rel.provenance else None,
                })

        return contradictions

    def get_relationships(self, start_node_ids: Optional[List[str]] = None) -> List[Relationship]:
        """Retrieve all graph relationships."""
        if self._http_client:
            resp = self._http_client.get("/api/v1/graph")
            resp.raise_for_status()
            raw_rels = resp.json().get("relationships", [])
            return [Relationship(**r) for r in raw_rels]
        else:
            assert self._container is not None
            subgraph = self._container.graph_store.query_subgraph(start_node_ids or [], depth=5)
            return list(subgraph.get("relationships", []))

    def get_relationship_evidence(self, relationship_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve relationship provenance evidence."""
        if self._http_client:
            resp = self._http_client.get(f"/api/v1/graph/relationships/{relationship_id}/evidence")
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return dict(resp.json())
        else:
            assert self._container is not None
            subgraph = self._container.graph_store.query_subgraph([], depth=1)
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
            return None

    def visualize(self, output_html_path: str = "graph.html", auto_open: bool = True) -> str:
        """Generate interactive HTML graph visualizer and optionally open in browser.

        Args:
            output_html_path: File path to save generated HTML application.
            auto_open: If True, automatically launches default web browser displaying the graph.

        Returns:
            Absolute path to generated HTML file.
        """
        import os
        import webbrowser
        from mind_graph_db.utils import GraphVisualizer

        if self._http_client:
            resp = self._http_client.get("/visualize")
            resp.raise_for_status()
            html_content = resp.text
            abs_path = os.path.abspath(output_html_path)
            with open(abs_path, "w", encoding="utf-8") as f:
                f.write(html_content)
        else:
            assert self._container is not None
            abs_path = os.path.abspath(output_html_path)
            GraphVisualizer.export_html(self._container.graph_store, output_path=abs_path)

        if auto_open:
            webbrowser.open(f"file:///{abs_path}")

        return abs_path

    def close(self) -> None:
        """Close underlying HTTP client connection if active."""
        if self._http_client:
            self._http_client.close()

