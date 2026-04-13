# Audit Trail & Lineage MWE

**Structured event logging with lineage chains and tamper detection — no external dependencies.**

## Overview

This MWE demonstrates audit trail infrastructure for ML governance:

1. **AuditEvent dataclass** — structured log entries with typed events and parent references
2. **AuditLogger** — append-only JSON Lines log with SHA-256 hash chains
3. **Lifecycle simulation** — data ingestion → training → deployment → predictions → alert
4. **Lineage traversal** — reconstruct the chain from prediction back to data source
5. **Tamper detection** — verify log integrity by re-computing hash chains
6. **Raw format** — inspect the JSON Lines format that stores audit trails

## Prerequisites

- Python 3.8+
- No external dependencies (stdlib only)

## Setup (3 steps)

```bash
# 1. Navigate to the MWE directory
cd module8/mwe/audit-trail-lineage

# 2. No dependencies to install

# 3. Run the demo
python main.py
```

## Expected Output

| Demo | What It Shows | Key Lesson |
|------|--------------|------------|
| 1. Lifecycle Logging | 8 events across the ML lifecycle | Every significant action gets an audit record |
| 2. Read Log | Tabular view of all events | JSON Lines format enables streaming reads |
| 3. Lineage Traversal | Prediction → deployment → training → data | Every prediction traces to its data source |
| 4. Integrity Check | Hash chain validates ✓ | Append-only logs with hash chains are tamper-evident |
| 5. Tamper Detection | Modified entry detected ✗ | Re-computed hashes catch content modifications |
| 6. Raw Format | JSON structure with hash fields | Each entry is self-contained with chain references |

## Key Concepts Demonstrated

| Concept | Implementation |
|---------|----------------|
| Event types | `EventType` enum (trained, deployed, prediction, alert, etc.) |
| Structured events | `AuditEvent` dataclass with actor, resource, details |
| Lineage chains | `parent_event_id` links events into a provenance graph |
| Tamper detection | SHA-256 hash chain: `chain_hash = hash(previous_hash + event_hash)` |
| Append-only log | JSON Lines format — one event per line, sequential |
| Integrity verification | Re-compute hashes from event content to detect modifications |

## Extension Challenges

1. Add `KNOWLEDGE_BASE_UPDATED` and `PROMPT_TEMPLATE_CHANGED` events for a RAG system
2. Implement a `query_by_type()` method to filter events by `EventType`
3. Add a `PredictionLogger` wrapper that auto-hashes input features
4. Build a lineage visualization that outputs a text-based tree diagram

## Connection to Final Project

This MWE directly supports **Component 3: Model Card & Governance Packet**:

- The `AuditLogger` provides the audit trail infrastructure
- The lineage traversal shows how to trace predictions to data sources
- The tamper detection demonstrates log integrity for compliance
- The event types cover the ML lifecycle events your project needs

## Related Reading

- Audit Trails & Lineage (`mod8notes/instructional-materials/audit-trails-lineage.md`)
- Model Cards (`mod8notes/instructional-materials/model-cards.md`)
