"""Synthetic knowledge corpus generator and independent ground-truth dataset framework for Mind Graph DB."""

import random
from typing import Any, Dict, List, Optional, Set, Tuple
from pydantic import BaseModel, Field

from mind_graph_db.core.types import Document, Entity, ProvenanceRecord, RelationshipKind


class GroundTruthEntity(BaseModel):
    """Ground-truth expected entity representation."""

    name: str
    canonical_name: str
    type: str
    aliases: List[str] = Field(default_factory=list)


class GroundTruthRelationship(BaseModel):
    """Ground-truth expected relationship representation."""

    source_entity: str
    target_entity: str
    relation_type: str
    kind: RelationshipKind = RelationshipKind.EXPLICIT
    confidence: float = 1.0
    evidence_text: str
    doc_id: str


class GroundTruthDocument(BaseModel):
    """Ground-truth expected document model with embedded ground-truth expectations."""

    id: str
    text: str
    domain: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    expected_entities: List[GroundTruthEntity] = Field(default_factory=list)
    expected_relationships: List[GroundTruthRelationship] = Field(default_factory=list)
    is_duplicate: bool = False
    has_contradiction: bool = False


class GroundTruthCorpus(BaseModel):
    """Encapsulates a complete synthetic corpus and independent ground-truth dataset."""

    documents: List[GroundTruthDocument] = Field(default_factory=list)
    entities: List[GroundTruthEntity] = Field(default_factory=list)
    relationships: List[GroundTruthRelationship] = Field(default_factory=list)
    profile: str = "quick"
    seed: int = 42


class SyntheticDataGenerator:
    """Configurable deterministic synthetic data generator producing realistic knowledge corpora."""

    DOMAINS = [
        "artificial_intelligence",
        "distributed_systems",
        "databases",
        "cybersecurity",
        "cloud_computing",
        "biology",
        "finance",
        "software_engineering",
    ]

    CONCEPTS = {
        "artificial_intelligence": [
            ("Machine Learning", ["ML", "machine learning techniques", "machine learning methodology"]),
            ("Deep Learning", ["DL", "deep neural learning"]),
            ("Natural Language Processing", ["NLP", "language processing"]),
            ("Computer Vision", ["CV", "visual processing"]),
            ("Vector Databases", ["Vector DB", "vector storage engine"]),
            ("Embeddings", ["dense vectors", "text embeddings"]),
            ("Neural Networks", ["artificial neural networks", "ANN"]),
            ("Transformer Models", ["transformers", "attention mechanisms"]),
        ],
        "distributed_systems": [
            ("Consensus Protocols", ["Raft", "Paxos", "distributed consensus"]),
            ("Fault Tolerance", ["resilience", "high availability"]),
            ("Replication Engine", ["state machine replication", "log replication"]),
            ("Distributed Locking", ["zookeeper locks", "etcd locking"]),
            ("Partitioning", ["sharding", "data partitioning"]),
            ("Event Eventual Consistency", ["eventual consistency", "CRDTs"]),
        ],
        "databases": [
            ("SQLite Storage", ["SQLite engine", "sqlite3 db"]),
            ("B-Tree Indices", ["B-Trees", "balanced tree index"]),
            ("ACID Transactions", ["atomicity consistency isolation durability"]),
            ("Graph Traversal", ["BFS graph search", "multi-hop graph traversal"]),
            ("Hybrid Retrieval", ["lexical vector hybrid search", "sparse dense retrieval"]),
            ("Query Execution Plan", ["AST query execution", "query planner"]),
        ],
        "cybersecurity": [
            ("Zero Trust Security", ["zero trust architecture", "ZTA"]),
            ("Encryption Standards", ["AES-256", "RSA encryption"]),
            ("Identity Authentication", ["OAuth2", "SAML SSO"]),
            ("Threat Detection", ["anomaly detection", "intrusion prevention"]),
        ],
    }

    TEMPLATES = [
        "{entity1} is a fundamental branch of {entity2} that enables {concept} to process complex datasets efficiently.",
        "{entity1} relies on {entity2} to maintain {concept} across distributed compute clusters.",
        "In modern systems, {entity1} combines with {entity2} for enhanced {concept} and high-performance operations.",
        "{entity1} explicitly supports {entity2} in production environments, ensuring reliable {concept}.",
        "{entity1} contradicts traditional assumptions about {entity2} when optimizing for {concept}.",
        "{entity1} is a critical component of {entity2}, providing underlying support for {concept}.",
    ]

    def __init__(self, seed: int = 42) -> None:
        """Initialize synthetic generator with a fixed random seed for reproducibility."""
        self.seed = seed

    def generate_corpus(self, profile: str = "quick") -> GroundTruthCorpus:
        """Generate a synthetic corpus and independent ground-truth dataset based on profile.

        Profiles:
        - 'quick': 100 documents
        - 'standard': 1,000 documents
        - 'stress': 5,000 documents
        - 'extreme': 10,000 documents
        """
        random.seed(self.seed)

        target_counts = {
            "quick": 100,
            "standard": 1000,
            "stress": 5000,
            "extreme": 10000,
        }
        num_docs = target_counts.get(profile, 100)

        corpus_docs: List[GroundTruthDocument] = []
        corpus_entities: List[GroundTruthEntity] = []
        corpus_relationships: List[GroundTruthRelationship] = []

        entity_pool: Dict[str, GroundTruthEntity] = {}

        # Pre-build entity pool
        for domain, concept_list in self.CONCEPTS.items():
            for name, aliases in concept_list:
                gt_ent = GroundTruthEntity(
                    name=name,
                    canonical_name=name,
                    type="CONCEPT",
                    aliases=aliases,
                )
                entity_pool[name] = gt_ent
                corpus_entities.append(gt_ent)

        entity_names = list(entity_pool.keys())

        for idx in range(num_docs):
            doc_id = f"synth-doc-{idx + 1:05d}"
            domain = self.DOMAINS[idx % len(self.DOMAINS)]

            # Select 2 distinct entities
            e1_name = entity_names[idx % len(entity_names)]
            e2_name = entity_names[(idx + 3) % len(entity_names)]
            if e1_name == e2_name:
                e2_name = entity_names[(idx + 5) % len(entity_names)]

            e1 = entity_pool[e1_name]
            e2 = entity_pool[e2_name]

            # Decide whether to use alias for variation
            text_e1 = random.choice([e1.name] + e1.aliases) if e1.aliases and idx % 3 == 0 else e1.name
            text_e2 = random.choice([e2.name] + e2.aliases) if e2.aliases and idx % 4 == 0 else e2.name

            tmpl = self.TEMPLATES[idx % len(self.TEMPLATES)]
            text = tmpl.format(
                entity1=text_e1,
                entity2=text_e2,
                concept=domain.replace("_", " "),
            )

            # Determine relationship type and kind
            is_contradict = "contradicts" in text.lower()
            rel_type = "CONTRADICTS" if is_contradict else ("SUPPORTS" if "supports" in text.lower() else "CO_OCCURS_WITH")
            rel_kind = RelationshipKind.CONTRADICTS if is_contradict else (RelationshipKind.SUPPORTS if "supports" in text.lower() else RelationshipKind.CO_OCCURS_WITH)

            gt_rel = GroundTruthRelationship(
                source_entity=e1.canonical_name,
                target_entity=e2.canonical_name,
                relation_type=rel_type,
                kind=rel_kind,
                confidence=0.9 if is_contradict else 0.75,
                evidence_text=text,
                doc_id=doc_id,
            )

            doc = GroundTruthDocument(
                id=doc_id,
                text=text,
                domain=domain,
                metadata={"domain": domain, "synthetic": True, "index": idx},
                expected_entities=[e1, e2],
                expected_relationships=[gt_rel],
                is_duplicate=(idx % 50 == 0 and idx > 0),
                has_contradiction=is_contradict,
            )

            corpus_docs.append(doc)
            corpus_relationships.append(gt_rel)

        return GroundTruthCorpus(
            documents=corpus_docs,
            entities=corpus_entities,
            relationships=corpus_relationships,
            profile=profile,
            seed=self.seed,
        )
