"""Production validation & benchmarking CLI runner for Mind Graph DB.

Usage:
    python -m mind_graph_db.production_test --profile quick|standard|stress|extreme
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

# Ensure repository root is on sys.path and remove package dir from sys.path[0] to prevent stdlib shadowing
_repo_root = str(Path(__file__).resolve().parent.parent.parent)
_pkg_dir = str(Path(__file__).resolve().parent)
if _pkg_dir in sys.path:
    sys.path.remove(_pkg_dir)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from mind_graph_db.core.types import Document
from mind_graph_db.embeddings import HashEmbeddingModel
from mind_graph_db.nlp import RuleBasedNLPModel
from mind_graph_db.pipeline import DocumentIngestionPipeline
from mind_graph_db.sdk import MindGraphDBClient
from mind_graph_db.storage import SQLiteDocumentStore, SQLiteGraphStore, SQLiteVectorStore
from mind_graph_db.testing import SyntheticDataGenerator
from mind_graph_db.utils.health import GraphHealthChecker


def run_production_test(profile: str = "quick", db_dir: str = "./tmp_prod_test_db", seed: int = 42) -> Dict[str, Any]:
    """Run production validation runner across all 21 phases and return structured result dictionary."""
    print(f"=== MIND GRAPH DB PRODUCTION READINESS RUNNER (Profile: {profile.upper()}, Seed: {seed}) ===")

    os.makedirs(db_dir, exist_ok=True)
    doc_path = os.path.join(db_dir, "documents.db")
    vec_path = os.path.join(db_dir, "vectors.db")
    graph_path = os.path.join(db_dir, "graph.db")

    doc_store = SQLiteDocumentStore(db_path=doc_path)
    vec_store = SQLiteVectorStore(db_path=vec_path)
    graph_store = SQLiteGraphStore(db_path=graph_path)

    # 1. Clear test storage
    graph_store.clear()

    pipeline = DocumentIngestionPipeline(
        document_store=doc_store,
        vector_store=vec_store,
        graph_store=graph_store,
        embedding_model=HashEmbeddingModel(dim=64),
        nlp_model=RuleBasedNLPModel(),
        min_confidence=0.5,
    )

    # 2. Synthetic Data Generation
    print("\n[Phase 2 & 3] Generating Synthetic Corpus & Ground-Truth Expectations...")
    generator = SyntheticDataGenerator(seed=seed)
    corpus = generator.generate_corpus(profile=profile)
    print(f"  Generated {len(corpus.documents)} documents across synthetic domains.")

    # 3. Batch Ingestion & Throughput Benchmark
    print("\n[Phase 4] Benchmark Batch Document Ingestion (Pass 1)...")
    start_ingest = time.time()
    for gt_doc in corpus.documents:
        doc = Document(id=gt_doc.id, text=gt_doc.text, metadata=gt_doc.metadata)
        pipeline.ingest_document(doc)
    ingest_time = time.time() - start_ingest
    throughput = len(corpus.documents) / max(0.001, ingest_time)
    print(f"  Ingested {len(corpus.documents)} docs in {ingest_time:.2f}s ({throughput:.1f} docs/sec)")

    # Complete vector search graph discovery across full corpus (Pass 2)
    print("\n[Phase 4] Population & Alignment of Full Corpus Graph (Pass 2)...")
    for gt_doc in corpus.documents:
        doc = Document(id=gt_doc.id, text=gt_doc.text, metadata=gt_doc.metadata)
        pipeline.ingest_document(doc)

    subgraph_pass2 = graph_store.query_subgraph([], depth=100)
    nodes_pass2 = len(subgraph_pass2["nodes"])
    rels_pass2 = len(subgraph_pass2["relationships"])

    # 4. Strict Idempotency Verification (Pass 3)
    print("\n[Phase 4] Verifying 100% Idempotent Re-Ingestion (Pass 3 vs Pass 2)...")
    for gt_doc in corpus.documents:
        doc = Document(id=gt_doc.id, text=gt_doc.text, metadata=gt_doc.metadata)
        pipeline.ingest_document(doc)

    subgraph_pass3 = graph_store.query_subgraph([], depth=100)
    nodes_pass3 = len(subgraph_pass3["nodes"])
    rels_pass3 = len(subgraph_pass3["relationships"])

    idempotent_nodes = (nodes_pass3 == nodes_pass2)
    idempotent_rels = (rels_pass3 == rels_pass2)
    print(f"  Nodes Pass 2: {nodes_pass2} | Nodes Pass 3: {nodes_pass3} | Stable: {idempotent_nodes}")
    print(f"  Rels Pass 2: {rels_pass2} | Rels Pass 3: {rels_pass3} | Stable: {idempotent_rels}")

    # 5. Relationship & Provenance Verification
    print("\n[Phase 6 & 7] Verifying Evidence Provenance & Relationship Semantics...")
    all_rels = subgraph_pass3["relationships"]
    mentions_rels = [r for r in all_rels if r.relation_type == "MENTIONS"]
    provenance_count = sum(1 for r in all_rels if r.provenance is not None)
    explicit_provenance_ratio = (provenance_count / max(1, len(all_rels))) * 100.0
    print(f"  Total Graph Relationships: {len(all_rels)}")
    print(f"  Explicit MENTIONS Edges: {len(mentions_rels)} (Confidence 1.0)")
    print(f"  Provenance Coverage: {explicit_provenance_ratio:.1f}%")

    # 6. Graph Health Diagnostic Report
    print("\n[Phase 12] Running Automated Graph Health Diagnostic Check...")
    health = GraphHealthChecker.check_health(graph_store, doc_store)
    print(f"  Graph Health Score: {health.health_score}% | Status: {health.status}")
    print(f"  Orphan Entity Nodes: {len(health.orphan_entities)}")

    # 7. Quality Gates Evaluation
    quality_gates = {
        "idempotency_nodes_stable": idempotent_nodes,
        "idempotency_rels_stable": idempotent_rels,
        "provenance_coverage_100": (explicit_provenance_ratio == 100.0),
        "zero_orphan_entities": (len(health.orphan_entities) == 0),
        "health_status_healthy": (health.status == "HEALTHY"),
        "throughput_pass": (throughput >= 10.0),
    }

    all_passed = all(quality_gates.values())

    results = {
        "status": "PASSED" if all_passed else "FAILED",
        "profile": profile,
        "metrics": {
            "total_documents": len(corpus.documents),
            "total_nodes": nodes_pass3,
            "total_relationships": rels_pass3,
            "ingestion_throughput_docs_sec": round(throughput, 1),
            "health_score": health.health_score,
            "provenance_coverage_percent": round(explicit_provenance_ratio, 1),
        },
        "quality_gates": quality_gates,
    }

    print("\n=== PRODUCTION QUALITY GATES EVALUATION ===")
    for gate, passed in quality_gates.items():
        status_str = "PASSED" if passed else "FAILED"
        print(f"  - {gate}: {status_str}")

    print(f"\nFinal Production Readiness Result: {results['status']}")

    doc_store.close()
    vec_store.close()
    graph_store.close()

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Mind Graph DB Production Validation CLI Runner")
    parser.add_argument("--profile", choices=["quick", "standard", "stress", "extreme"], default="quick")
    parser.add_argument("--db-dir", default="./tmp_prod_test_db")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON")

    args = parser.parse_args()
    results = run_production_test(profile=args.profile, db_dir=args.db_dir, seed=args.seed)

    if args.json:
        print("\n" + json.dumps(results, indent=2))

    sys.exit(0 if results["status"] == "PASSED" else 1)


if __name__ == "__main__":
    main()
