# Mind Graph DB - Production Validation & Benchmarking Guide

## Overview

The production test runner command evaluates Mind Graph DB across 21 phases against synthetic ground-truth corpora.

## Command Usage

```bash
python -m mind_graph_db.production_test --profile quick|standard|stress|extreme
```

### Profiles

- **`quick`**: 100 documents (~500 entities, ~1,000 relationships). Ideal for rapid smoke testing and local validation.
- **`standard`**: 1,000 documents (~5,000 entities, ~10,000 relationships). Comprehensive integration testing.
- **`stress`**: 5,000 documents (~25,000 entities, ~50,000 relationships). High-load performance testing.
- **`extreme`**: 10,000 documents (~50,000 entities, ~100,000 relationships). Scale limits evaluation.

---

## Production Quality Gates

The production test runner enforces 6 explicit quality gates:

1. **`idempotency_nodes_stable`**: Re-ingesting documents yields 0 node count growth.
2. **`idempotency_rels_stable`**: Re-ingesting documents yields 0 relationship edge count growth.
3. **`provenance_coverage_100`**: 100% of graph relationships possess valid `ProvenanceRecord` objects.
4. **`zero_orphan_entities`**: Deleting or updating documents leaves 0 orphan entity nodes.
5. **`health_status_healthy`**: Automated `GraphHealthChecker` reports `HEALTHY` status.
6. **`throughput_pass`**: Document ingestion throughput exceeds 10 docs/sec baseline threshold.
