"""Document ingestion pipeline automating storage, vector indexing, NLP analysis, entity resolution, and graph discovery."""

import re
from typing import Any, Dict, List, Optional, Set, Tuple
from pydantic import BaseModel, Field

from mind_graph_db.core.types import Document, GraphNode, ProvenanceRecord, Relationship, RelationshipKind
from mind_graph_db.interfaces.document_store import DocumentStore
from mind_graph_db.interfaces.embedding import EmbeddingModel
from mind_graph_db.interfaces.graph_store import GraphStore
from mind_graph_db.interfaces.nlp import NLPModel
from mind_graph_db.interfaces.vector_store import VectorStore


class IngestionResult(BaseModel):
    """Summary result returned after processing or updating a document through the ingestion pipeline."""

    document_id: str
    vector_id: str
    entities_extracted: int = 0
    facts_extracted: int = 0
    relationships_created: int = 0
    relationships_updated: int = 0
    relationships_removed: int = 0
    candidate_docs_evaluated: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DocumentIngestionPipeline:
    """Automated 10-step pipeline orchestrating document ingestion, vector indexing, entity resolution, and knowledge graph construction."""

    def __init__(
        self,
        document_store: DocumentStore,
        vector_store: VectorStore,
        graph_store: GraphStore,
        embedding_model: EmbeddingModel,
        nlp_model: NLPModel,
        min_confidence: float = 0.6,
        candidate_top_k: int = 5,
    ) -> None:
        """Initialize the document ingestion pipeline with required components.

        Args:
            document_store: DocumentStore instance.
            vector_store: VectorStore instance.
            graph_store: GraphStore instance.
            embedding_model: EmbeddingModel instance.
            nlp_model: NLPModel instance.
            min_confidence: Minimum confidence threshold for committing graph relationships.
            candidate_top_k: Number of vector candidate documents to evaluate.
        """
        self.document_store = document_store
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.embedding_model = embedding_model
        self.nlp_model = nlp_model
        self.min_confidence = min_confidence
        self.candidate_top_k = candidate_top_k

    def ingest_document(self, document: Document) -> IngestionResult:
        """Process a document through the 10-step automatic ingestion and graph discovery pipeline.

        1. Store the document.
        2. Generate its embedding and store in vector index.
        3. Extract entities, concepts, and facts via NLP model.
        4. Create Document node in knowledge graph.
        5. Perform entity resolution & add MENTIONS edges.
        6. Discover candidate documents via vector search.
        7. Analyze document against candidates for cross-document links.
        8. Compute confidence scores for relationship candidates.
        9. Store evidence showing why each relationship was created.
        10. Commit valid nodes and edges into the knowledge graph.

        Args:
            document: Document model instance to ingest.

        Returns:
            IngestionResult summarizing pipeline execution metrics.
        """
        # Step 1: Store document
        doc_id = self.document_store.put(document)

        # Step 2: Generate embedding & store vector
        embedding = self.embedding_model.encode_text(document.text)
        vector_meta = dict(document.metadata)
        vector_meta["doc_id"] = doc_id
        self.vector_store.add(vector_id=doc_id, embedding=embedding, metadata=vector_meta)

        # Step 3: Extract entities and semantic information
        semantic_result = self.nlp_model.analyze(document.text, document_id=doc_id)

        # Step 4: Create DOCUMENT node in GraphStore
        doc_label = document.text[:40] + ("..." if len(document.text) > 40 else "")
        doc_node = GraphNode(
            id=doc_id,
            label=doc_label,
            node_type="DOCUMENT",
            properties=dict(document.metadata),
        )
        self.graph_store.add_node(doc_node)

        rel_created_count = 0

        # Step 5: Entity Resolution & Document MENTIONS Edges
        resolved_entities: Dict[str, str] = {}  # Map clean entity name -> entity node ID
        sentences = [s.strip() for s in re.split(r"[.!?]\s+", document.text) if s.strip()]

        for entity in semantic_result.entities:
            clean_name = entity.name.strip()
            existing_node = self.graph_store.get_node_by_label(clean_name, node_type="ENTITY")
            if existing_node:
                ent_id = existing_node.id
            else:
                ent_id = entity.id
                ent_node = GraphNode(
                    id=ent_id,
                    label=clean_name,
                    node_type="ENTITY",
                    properties={"entity_type": entity.type, **entity.metadata},
                )
                self.graph_store.add_node(ent_node)

            resolved_entities[clean_name.lower()] = ent_id

            # Find supporting sentence in document text
            supporting_sentence = next(
                (s for s in sentences if clean_name.lower() in s.lower()),
                document.text,
            )
            char_start = document.text.lower().find(clean_name.lower())
            char_span = (char_start, char_start + len(clean_name)) if char_start != -1 else None

            prov = ProvenanceRecord(
                kind=RelationshipKind.EXPLICIT,
                source_doc_id=doc_id,
                evidence_text=supporting_sentence,
                char_span=char_span,
                confidence=1.0,
                method="nlp_explicit_extraction",
            )

            # Create explicit MENTIONS edge with 1.0 confidence and sentence provenance
            mentions_edge = Relationship(
                source_id=doc_id,
                target_id=ent_id,
                relation_type="MENTIONS",
                confidence=1.0,
                evidence_text=f"Document '{doc_id}' explicitly mentions entity '{clean_name}' in sentence: '{supporting_sentence}'",
                provenance=prov,
            )
            self.graph_store.add_relationship(mentions_edge)
            rel_created_count += 1

        # Step 6: Vector Candidate Discovery (avoiding O(N) full DB search)
        vector_candidates = self.vector_store.search(
            query_embedding=embedding,
            top_k=self.candidate_top_k + 1,
        )
        # Exclude self from candidates
        valid_candidates = [(vec, score) for vec, score in vector_candidates if vec.id != doc_id]

        # Step 7, 8, 9 & 10: Candidate Evaluation, Confidence Scoring & Evidence Provenance
        for cand_vec, sim_score in valid_candidates:
            cand_doc_id = cand_vec.id
            conf_score = max(0.0, min(1.0, float(sim_score)))

            # Create SIMILAR_TO relationship if vector similarity meets threshold
            if sim_score >= self.min_confidence:
                sim_prov = ProvenanceRecord(
                    kind=RelationshipKind.SIMILAR_TO,
                    source_doc_id=doc_id,
                    evidence_text=f"Vector cosine similarity: {sim_score:.3f}",
                    confidence=conf_score,
                    method="vector_cosine_similarity",
                )
                sim_edge = Relationship(
                    source_id=doc_id,
                    target_id=cand_doc_id,
                    relation_type="SIMILAR_TO",
                    confidence=conf_score,
                    evidence_text=(
                        f"Vector cosine similarity ({sim_score:.3f}) "
                        f"between doc '{doc_id}' and doc '{cand_doc_id}'"
                    ),
                    provenance=sim_prov,
                )
                self.graph_store.add_relationship(sim_edge)
                rel_created_count += 1


        # Store intra-document relationship candidates meeting confidence threshold
        for candidate in semantic_result.relationship_candidates:
            if candidate.confidence >= self.min_confidence:
                src_name = candidate.source_entity.name.strip()
                tgt_name = candidate.target_entity.name.strip()

                src_existing = self.graph_store.get_node_by_label(src_name, node_type="ENTITY")
                if src_existing:
                    src_id = src_existing.id
                else:
                    src_id = candidate.source_entity.id
                    self.graph_store.add_node(
                        GraphNode(id=src_id, label=src_name, node_type="ENTITY")
                    )

                tgt_existing = self.graph_store.get_node_by_label(tgt_name, node_type="ENTITY")
                if tgt_existing:
                    tgt_id = tgt_existing.id
                else:
                    tgt_id = candidate.target_entity.id
                    self.graph_store.add_node(
                        GraphNode(id=tgt_id, label=tgt_name, node_type="ENTITY")
                    )

                co_prov = ProvenanceRecord(
                    kind=RelationshipKind.CO_OCCURS_WITH,
                    source_doc_id=doc_id,
                    evidence_text=candidate.evidence_text or f"Extracted from document '{doc_id}'",
                    confidence=candidate.confidence,
                    method="sentence_co_occurrence",
                )
                rel_edge = Relationship(
                    source_id=src_id,
                    target_id=tgt_id,
                    relation_type=candidate.relation_type,
                    confidence=candidate.confidence,
                    evidence_text=candidate.evidence_text or f"Extracted from document '{doc_id}'",
                    provenance=co_prov,
                )
                self.graph_store.add_relationship(rel_edge)
                rel_created_count += 1

        return IngestionResult(
            document_id=doc_id,
            vector_id=doc_id,
            entities_extracted=len(semantic_result.entities),
            facts_extracted=len(semantic_result.facts),
            relationships_created=rel_created_count,
            candidate_docs_evaluated=len(valid_candidates),
        )

    def update_document(self, document: Document) -> IngestionResult:
        """Update an existing document, regenerate embeddings, re-analyze semantics, prune stale edges, and preserve valid relationships.

        Args:
            document: Document model instance with updated text/metadata.

        Returns:
            IngestionResult summarizing update execution metrics.
        """
        # Step 1: Touch timestamp and update DocumentStore
        document.touch()
        self.document_store.update(document)

        doc_id = document.id

        # Step 2: Regenerate embedding & update VectorStore
        embedding = self.embedding_model.encode_text(document.text)
        vector_meta = dict(document.metadata)
        vector_meta["doc_id"] = doc_id
        vector_meta["updated_at"] = document.updated_at.isoformat()
        self.vector_store.add(vector_id=doc_id, embedding=embedding, metadata=vector_meta)

        # Step 3: Fetch existing graph edges connected to document.id
        old_neighbors = self.graph_store.get_neighbors(doc_id)
        old_edges_map: Dict[Tuple[str, str, str], Relationship] = {}
        for _, edge in old_neighbors:
            old_edges_map[(edge.source_id, edge.target_id, edge.relation_type)] = edge

        # Step 4: Re-run semantic analysis
        semantic_result = self.nlp_model.analyze(document.text, document_id=doc_id)

        # Step 5: Update DOCUMENT node in GraphStore
        doc_label = document.text[:40] + ("..." if len(document.text) > 40 else "")
        doc_node = GraphNode(
            id=doc_id,
            label=doc_label,
            node_type="DOCUMENT",
            properties={"updated_at": document.updated_at.isoformat(), **document.metadata},
        )
        self.graph_store.add_node(doc_node)

        # Build candidate new relationships set
        new_edges_map: Dict[Tuple[str, str, str], Relationship] = {}

        # Entity Resolution for updated text
        resolved_entities: Dict[str, str] = {}
        for entity in semantic_result.entities:
            clean_name = entity.name.strip()
            existing_node = self.graph_store.get_node_by_label(clean_name, node_type="ENTITY")
            if existing_node:
                ent_id = existing_node.id
            else:
                ent_id = entity.id
                ent_node = GraphNode(
                    id=ent_id,
                    label=clean_name,
                    node_type="ENTITY",
                    properties={"entity_type": entity.type, **entity.metadata},
                )
                self.graph_store.add_node(ent_node)

            resolved_entities[clean_name.lower()] = ent_id

            mentions_edge = Relationship(
                source_id=doc_id,
                target_id=ent_id,
                relation_type="MENTIONS",
                confidence=1.0,
                evidence_text=f"Document '{doc_id}' explicitly mentions entity '{clean_name}' (updated at {document.updated_at.isoformat()})",
            )
            new_edges_map[(doc_id, ent_id, "MENTIONS")] = mentions_edge


        # Vector Candidate Discovery for updated content
        vector_candidates = self.vector_store.search(
            query_embedding=embedding,
            top_k=self.candidate_top_k + 1,
        )
        valid_candidates = [(vec, score) for vec, score in vector_candidates if vec.id != doc_id]

        for cand_vec, sim_score in valid_candidates:
            cand_doc_id = cand_vec.id
            conf_score = max(0.0, min(1.0, float(sim_score)))
            if sim_score >= self.min_confidence:
                sim_edge = Relationship(
                    source_id=doc_id,
                    target_id=cand_doc_id,
                    relation_type="SIMILAR_TO",
                    confidence=conf_score,
                    evidence_text=(
                        f"Vector cosine similarity ({sim_score:.3f}) "
                        f"between updated doc '{doc_id}' and doc '{cand_doc_id}'"
                    ),
                )
                new_edges_map[(doc_id, cand_doc_id, "SIMILAR_TO")] = sim_edge


        # Intra-document candidates
        for candidate in semantic_result.relationship_candidates:
            if candidate.confidence >= self.min_confidence:
                src_name = candidate.source_entity.name.strip()
                tgt_name = candidate.target_entity.name.strip()

                src_existing = self.graph_store.get_node_by_label(src_name, node_type="ENTITY")
                if src_existing:
                    src_id = src_existing.id
                else:
                    src_id = candidate.source_entity.id
                    self.graph_store.add_node(
                        GraphNode(id=src_id, label=src_name, node_type="ENTITY")
                    )

                tgt_existing = self.graph_store.get_node_by_label(tgt_name, node_type="ENTITY")
                if tgt_existing:
                    tgt_id = tgt_existing.id
                else:
                    tgt_id = candidate.target_entity.id
                    self.graph_store.add_node(
                        GraphNode(id=tgt_id, label=tgt_name, node_type="ENTITY")
                    )

                rel_edge = Relationship(
                    source_id=src_id,
                    target_id=tgt_id,
                    relation_type=candidate.relation_type,
                    confidence=candidate.confidence,
                    evidence_text=candidate.evidence_text or f"Extracted from updated document '{doc_id}'",
                )
                new_edges_map[(src_id, tgt_id, candidate.relation_type)] = rel_edge

        # Step 6: Reconcile Graph Relationships (Prune Obsolete, Preserve Valid, Commit New)
        created_count = 0
        updated_count = 0
        removed_count = 0

        # Delete edges present in old state but missing from new state
        for key, old_edge in old_edges_map.items():
            if key not in new_edges_map:
                self.graph_store.delete_relationship(
                    source_id=old_edge.source_id,
                    target_id=old_edge.target_id,
                    relation_type=old_edge.relation_type,
                )
                removed_count += 1

        # Add or update new state edges
        for key, new_edge in new_edges_map.items():
            if key in old_edges_map:
                updated_count += 1
            else:
                created_count += 1
            self.graph_store.add_relationship(new_edge)

        return IngestionResult(
            document_id=doc_id,
            vector_id=doc_id,
            entities_extracted=len(semantic_result.entities),
            facts_extracted=len(semantic_result.facts),
            relationships_created=created_count,
            relationships_updated=updated_count,
            relationships_removed=removed_count,
            candidate_docs_evaluated=len(valid_candidates),
        )
