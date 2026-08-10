"""Python Client SDK for Mind Graph DB supporting HTTP REST and in-process execution modes."""

from typing import Any, Dict, List, Optional
import httpx

from mind_graph_db.api.deps import DatabaseContainer, init_database
from mind_graph_db.core.types import Document, QueryResult, TraversalPath
from mind_graph_db.pipeline import IngestionResult


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

