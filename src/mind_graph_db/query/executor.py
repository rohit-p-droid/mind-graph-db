"""MindQueryEngine implementing hybrid vector, metadata, and graph traversal query execution."""

from typing import Any, Dict, List, Optional, Set, Tuple

from mind_graph_db.core.types import Document, Entity, QueryResult, Relationship
from mind_graph_db.interfaces.document_store import DocumentStore
from mind_graph_db.interfaces.embedding import EmbeddingModel
from mind_graph_db.interfaces.graph_store import GraphStore
from mind_graph_db.interfaces.query_engine import QueryEngine
from mind_graph_db.interfaces.vector_store import VectorStore
from mind_graph_db.query.ast import MindQuery
from mind_graph_db.query.parser import QueryParser
from mind_graph_db.query.planner import QueryPlanner


class MindQueryEngine(QueryEngine):
    """Engine executing hybrid semantic, metadata, and multi-hop graph queries against Mind Graph DB."""

    def __init__(
        self,
        document_store: DocumentStore,
        vector_store: VectorStore,
        graph_store: GraphStore,
        embedding_model: EmbeddingModel,
    ) -> None:
        """Initialize MindQueryEngine with required storage and model components."""
        self.document_store = document_store
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.embedding_model = embedding_model
        self.parser = QueryParser()
        self.planner = QueryPlanner()

    def execute_ast(self, query_ast: MindQuery) -> QueryResult:
        """Execute a parsed MindQuery AST and return a structured QueryResult."""
        plan = self.planner.create_plan(query_ast)

        scores_map: Dict[str, float] = {}
        candidate_doc_ids: List[str] = []

        # Stage 1: Vector Semantic Search
        if query_ast.semantic_query:
            query_vec = self.embedding_model.encode_text(query_ast.semantic_query)
            search_results = self.vector_store.search(query_vec, top_k=query_ast.top_k * 2)
            for vec, score in search_results:
                candidate_doc_ids.append(vec.id)
                scores_map[vec.id] = float(score)
        else:
            docs = self.document_store.list_documents(limit=query_ast.top_k * 2)
            candidate_doc_ids = [d.id for d in docs]
            for d_id in candidate_doc_ids:
                scores_map[d_id] = 1.0

        # Stage 2: Metadata Filtering
        matched_documents: List[Document] = []
        for doc_id in candidate_doc_ids:
            doc = self.document_store.get(doc_id)
            if doc is None:
                continue

            matches_filters = True
            for k, expected_val in query_ast.metadata_filters.items():
                if doc.metadata.get(k) != expected_val:
                    matches_filters = False
                    break

            if matches_filters:
                matched_documents.append(doc)

        # Truncate to top_k
        matched_documents = matched_documents[: query_ast.top_k]

        # Stage 3: Graph Traversal
        matched_entities: Dict[str, Entity] = {}
        matched_relationships: Dict[str, Relationship] = {}

        for doc in matched_documents:
            if query_ast.traverse_hops > 0:
                paths = self.graph_store.traverse(
                    doc.id,
                    max_hops=query_ast.traverse_hops,
                    relation_types=query_ast.allowed_relationship_types,
                )
                for path in paths:
                    for node in path.nodes:
                        if node.node_type == "ENTITY":
                            ent_type = str(node.properties.get("entity_type", "CONCEPT"))
                            matched_entities[node.id] = Entity(
                                id=node.id,
                                name=node.label,
                                type=ent_type,
                                metadata=node.properties,
                            )
                    for edge in path.edges:
                        matched_relationships[edge.id] = edge
            else:
                # Direct neighbor retrieval
                neighbors = self.graph_store.get_neighbors(
                    doc.id,
                    relation_types=query_ast.allowed_relationship_types,
                )
                for node, edge in neighbors:
                    if node.node_type == "ENTITY":
                        ent_type = str(node.properties.get("entity_type", "CONCEPT"))
                        matched_entities[node.id] = Entity(
                            id=node.id,
                            name=node.label,
                            type=ent_type,
                            metadata=node.properties,
                        )
                    matched_relationships[edge.id] = edge

        # Stage 4: Result Ranking & Explainable Reasoning Paths
        matched_documents.sort(key=lambda d: scores_map.get(d.id, 0.0), reverse=True)

        reasoning_paths: List[Dict[str, Any]] = []
        for doc in matched_documents:
            doc_score = scores_map.get(doc.id, 0.0)
            doc_rels = [
                rel.model_dump(mode="json")
                for rel in matched_relationships.values()
                if rel.source_id == doc.id or rel.target_id == doc.id
            ]
            reasoning_paths.append({
                "document_id": doc.id,
                "relevance_score": doc_score,
                "evidence_snippet": doc.text[:140] + ("..." if len(doc.text) > 140 else ""),
                "supporting_relationships": doc_rels,
            })

        return QueryResult(
            query=query_ast.semantic_query or query_ast.target,
            documents=matched_documents if "documents" in query_ast.return_fields else [],
            entities=list(matched_entities.values()) if "entities" in query_ast.return_fields else [],
            relationships=list(matched_relationships.values()) if "relationships" in query_ast.return_fields else [],
            reasoning_paths=reasoning_paths,
            scores={d.id: scores_map.get(d.id, 0.0) for d in matched_documents},
            metadata={"plan_steps": [s.name for s in plan.steps]},
        )

    def query(
        self,
        query_text: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        include_subgraph: bool = True,
    ) -> QueryResult:
        """Parse query string, apply top_k and filters, execute plan, and return QueryResult."""
        ast = self.parser.parse(query_text)
        ast.top_k = top_k
        if filters:
            ast.metadata_filters.update(filters)
        return self.execute_ast(ast)
