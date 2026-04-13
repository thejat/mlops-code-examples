"""
Audit Trail & Lineage MWE - Structured Event Logging with Lineage Chains

Demonstrates audit trail infrastructure for ML governance:

  1. AuditEvent dataclass — structured log entries with typed events
  2. AuditLogger — append-only log with SHA-256 tamper detection
  3. Lifecycle simulation — data ingestion → training → deployment → predictions
  4. Lineage traversal — reconstruct the chain from prediction to data source
  5. Tamper detection — verify log integrity via hash chain

No external dependencies required (stdlib only).

Run: python main.py
"""

from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, Dict, Any, List
import json
import hashlib
import uuid
import os
import tempfile


# =============================================================================
# Event Types and Data Structures
# =============================================================================

class EventType(Enum):
    """Types of auditable events in an ML system."""
    DATA_INGESTED = "data_ingested"
    MODEL_TRAINED = "model_trained"
    MODEL_DEPLOYED = "model_deployed"
    MODEL_RETIRED = "model_retired"
    CONFIG_CHANGED = "config_changed"
    PREDICTION_MADE = "prediction_made"
    ALERT_TRIGGERED = "alert_triggered"
    MANUAL_OVERRIDE = "manual_override"
    KNOWLEDGE_BASE_UPDATED = "knowledge_base_updated"
    PROMPT_TEMPLATE_CHANGED = "prompt_template_changed"


@dataclass
class AuditEvent:
    """Structured audit log entry."""

    event_id: str
    event_type: EventType
    timestamp: datetime
    actor: str
    resource_type: str
    resource_id: str
    action: str
    details: Dict[str, Any]
    parent_event_id: Optional[str] = None

    def to_json(self) -> str:
        """Serialize to JSON for storage."""
        data = asdict(self)
        data["event_type"] = self.event_type.value
        data["timestamp"] = self.timestamp.isoformat()
        return json.dumps(data)

    def compute_hash(self) -> str:
        """Compute SHA-256 hash for tamper detection."""
        content = (
            f"{self.event_id}"
            f"{self.event_type.value}"
            f"{self.timestamp.isoformat()}"
            f"{self.actor}"
            f"{self.action}"
            f"{json.dumps(self.details, sort_keys=True)}"
        )
        return hashlib.sha256(content.encode()).hexdigest()[:16]


# =============================================================================
# Audit Logger
# =============================================================================

class AuditLogger:
    """Append-only audit trail logger with hash-based integrity."""

    def __init__(self, log_path: str):
        self.log_path = log_path
        self._event_counter = 0
        self._previous_hash = "0" * 16  # Genesis hash

    def log_event(self, event: AuditEvent) -> str:
        """
        Append event to the audit log.

        Returns the event hash for reference.
        """
        event_hash = event.compute_hash()

        # Chain hash includes previous hash for tamper detection
        chain_hash = hashlib.sha256(
            f"{self._previous_hash}{event_hash}".encode()
        ).hexdigest()[:16]

        log_entry = {
            "sequence": self._event_counter,
            "event": json.loads(event.to_json()),
            "event_hash": event_hash,
            "chain_hash": chain_hash,
            "previous_hash": self._previous_hash,
        }

        with open(self.log_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        self._previous_hash = chain_hash
        self._event_counter += 1
        return event_hash

    def read_log(self) -> List[Dict]:
        """Read all entries from the audit log."""
        entries = []
        if not os.path.exists(self.log_path):
            return entries
        with open(self.log_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        return entries

    def verify_integrity(self) -> Dict[str, Any]:
        """
        Verify the integrity of the audit log by re-computing hash chains.

        Returns a report of verification results.
        """
        entries = self.read_log()
        if not entries:
            return {"valid": True, "entries_checked": 0, "errors": []}

        errors = []
        previous_hash = "0" * 16  # Genesis hash

        for i, entry in enumerate(entries):
            # Verify sequence
            if entry["sequence"] != i:
                errors.append(
                    f"Sequence mismatch at position {i}: "
                    f"expected {i}, got {entry['sequence']}"
                )

            # Re-compute event_hash from event content to detect tampering
            evt = entry["event"]
            recomputed_hash = hashlib.sha256(
                (
                    f"{evt['event_id']}"
                    f"{evt['event_type']}"
                    f"{evt['timestamp']}"
                    f"{evt['actor']}"
                    f"{evt['action']}"
                    f"{json.dumps(evt['details'], sort_keys=True)}"
                ).encode()
            ).hexdigest()[:16]

            if recomputed_hash != entry["event_hash"]:
                errors.append(
                    f"Event hash mismatch at sequence {i}: "
                    f"event content was modified "
                    f"(expected {entry['event_hash']}, "
                    f"recomputed {recomputed_hash})"
                )

            # Verify chain hash
            expected_chain = hashlib.sha256(
                f"{previous_hash}{entry['event_hash']}".encode()
            ).hexdigest()[:16]

            if entry["chain_hash"] != expected_chain:
                errors.append(
                    f"Chain hash mismatch at sequence {i}: "
                    f"expected {expected_chain}, got {entry['chain_hash']}"
                )

            # Verify previous hash link
            if entry["previous_hash"] != previous_hash:
                errors.append(
                    f"Previous hash mismatch at sequence {i}: "
                    f"expected {previous_hash}, got {entry['previous_hash']}"
                )

            previous_hash = entry["chain_hash"]

        return {
            "valid": len(errors) == 0,
            "entries_checked": len(entries),
            "errors": errors,
        }


# =============================================================================
# Convenience Functions
# =============================================================================

def make_event_id(prefix: str) -> str:
    """Generate a short unique event ID."""
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


# =============================================================================
# Demonstrations
# =============================================================================

def demo_lifecycle_logging(logger: AuditLogger) -> List[str]:
    """Demo 1: Log a complete ML model lifecycle."""
    print("=" * 60)
    print("Demo 1: ML Model Lifecycle Audit Trail")
    print("=" * 60)

    base_time = datetime(2024, 1, 10, 9, 0, 0)
    event_ids = []

    # Step 1: Data ingestion
    data_event = AuditEvent(
        event_id=make_event_id("data"),
        event_type=EventType.DATA_INGESTED,
        timestamp=base_time,
        actor="data-pipeline",
        resource_type="dataset",
        resource_id="customer_transactions_v3",
        action="ingest_training_data",
        details={
            "source": "s3://data-lake/transactions/",
            "rows": 450000,
            "columns": 47,
            "hash": "a1b2c3d4e5f6",
            "date_range": "2020-01-01 to 2023-12-31",
        },
    )
    logger.log_event(data_event)
    event_ids.append(data_event.event_id)
    print(f"\n  [1] {data_event.event_type.value}: {data_event.resource_id}")

    # Step 2: Model training
    train_event = AuditEvent(
        event_id=make_event_id("train"),
        event_type=EventType.MODEL_TRAINED,
        timestamp=base_time + timedelta(hours=2),
        actor="ml-engineer@company.com",
        resource_type="model",
        resource_id="churn_predictor_v2.1.0",
        action="train_model",
        details={
            "algorithm": "RandomForestClassifier",
            "hyperparameters": {
                "n_estimators": 200,
                "max_depth": 15,
            },
            "metrics": {
                "auc_roc": 0.847,
                "f1": 0.720,
            },
            "training_dataset": data_event.event_id,
        },
        parent_event_id=data_event.event_id,
    )
    logger.log_event(train_event)
    event_ids.append(train_event.event_id)
    print(f"  [2] {train_event.event_type.value}: {train_event.resource_id}")

    # Step 3: Model deployment
    deploy_event = AuditEvent(
        event_id=make_event_id("deploy"),
        event_type=EventType.MODEL_DEPLOYED,
        timestamp=base_time + timedelta(hours=4),
        actor="ml-engineer@company.com",
        resource_type="model",
        resource_id="churn_predictor_v2.1.0",
        action="deploy_to_production",
        details={
            "environment": "production",
            "endpoint": "https://api.company.com/predict/churn",
            "replicas": 3,
            "model_artifact": "gs://models/churn_v2.1.0/model.pkl",
        },
        parent_event_id=train_event.event_id,
    )
    logger.log_event(deploy_event)
    event_ids.append(deploy_event.event_id)
    print(f"  [3] {deploy_event.event_type.value}: {deploy_event.resource_id}")

    # Step 4: Predictions
    for i in range(3):
        pred_event = AuditEvent(
            event_id=make_event_id("pred"),
            event_type=EventType.PREDICTION_MADE,
            timestamp=base_time + timedelta(hours=5, minutes=i * 10),
            actor=f"user_{1000 + i}",
            resource_type="prediction",
            resource_id=f"req_{uuid.uuid4().hex[:8]}",
            action="inference",
            details={
                "model_version": "v2.1.0",
                "input_hash": hashlib.sha256(
                    f"input_{i}".encode()
                ).hexdigest()[:16],
                "prediction": "churn" if i == 1 else "retain",
                "confidence": [0.87, 0.92, 0.65][i],
                "latency_ms": [45, 52, 48][i],
            },
            parent_event_id=deploy_event.event_id,
        )
        logger.log_event(pred_event)
        event_ids.append(pred_event.event_id)
        print(
            f"  [{4 + i}] {pred_event.event_type.value}: "
            f"confidence={pred_event.details['confidence']}"
        )

    # Step 5: Config change
    config_event = AuditEvent(
        event_id=make_event_id("config"),
        event_type=EventType.CONFIG_CHANGED,
        timestamp=base_time + timedelta(days=1),
        actor="ops-team@company.com",
        resource_type="config",
        resource_id="churn_predictor_config",
        action="update_threshold",
        details={
            "parameter": "decision_threshold",
            "old_value": 0.5,
            "new_value": 0.45,
            "reason": "Increase recall for enterprise segment",
        },
        parent_event_id=deploy_event.event_id,
    )
    logger.log_event(config_event)
    event_ids.append(config_event.event_id)
    print(f"  [7] {config_event.event_type.value}: threshold 0.5 → 0.45")

    # Step 6: Alert triggered
    alert_event = AuditEvent(
        event_id=make_event_id("alert"),
        event_type=EventType.ALERT_TRIGGERED,
        timestamp=base_time + timedelta(days=3),
        actor="monitoring-system",
        resource_type="alert",
        resource_id="drift_alert_age",
        action="drift_threshold_exceeded",
        details={
            "feature": "age",
            "psi_score": 0.23,
            "threshold": 0.20,
            "severity": "warning",
        },
        parent_event_id=deploy_event.event_id,
    )
    logger.log_event(alert_event)
    event_ids.append(alert_event.event_id)
    print(f"  [8] {alert_event.event_type.value}: PSI=0.23 > threshold")

    print(f"\n  Total events logged: {len(event_ids)}")
    return event_ids


def demo_read_log(logger: AuditLogger):
    """Demo 2: Read and display audit log entries."""
    print("\n" + "=" * 60)
    print("Demo 2: Reading the Audit Log")
    print("=" * 60)

    entries = logger.read_log()

    print(f"\n  {'Seq':>3}  {'Event Type':<25} {'Actor':<25} {'Resource'}")
    print(f"  {'—' * 3}  {'—' * 25} {'—' * 25} {'—' * 25}")

    for entry in entries:
        event = entry["event"]
        print(
            f"  {entry['sequence']:>3}  "
            f"{event['event_type']:<25} "
            f"{event['actor']:<25} "
            f"{event['resource_id'][:25]}"
        )

    print(f"\n  Log file size: {os.path.getsize(logger.log_path)} bytes")
    print(f"  Format: JSON Lines (one JSON object per line)")


def demo_lineage_traversal(logger: AuditLogger):
    """Demo 3: Traverse lineage chain from prediction back to data source."""
    print("\n" + "=" * 60)
    print("Demo 3: Lineage Chain Traversal")
    print("=" * 60)
    print("\n  Tracing from prediction back to data source:\n")

    entries = logger.read_log()

    # Build event lookup by ID
    event_by_id = {}
    for entry in entries:
        event = entry["event"]
        event_by_id[event["event_id"]] = event

    # Find the first prediction event
    pred_event = None
    for entry in entries:
        if entry["event"]["event_type"] == "prediction_made":
            pred_event = entry["event"]
            break

    if not pred_event:
        print("  No prediction events found.")
        return

    # Walk the lineage chain
    chain = []
    current = pred_event
    visited = set()

    while current and current["event_id"] not in visited:
        visited.add(current["event_id"])
        chain.append(current)
        parent_id = current.get("parent_event_id")
        if parent_id and parent_id in event_by_id:
            current = event_by_id[parent_id]
        else:
            current = None

    # Display chain (reverse to show source → prediction)
    chain.reverse()
    for i, event in enumerate(chain):
        indent = "  " * i
        connector = "→ " if i > 0 else "  "
        print(
            f"  {connector}{indent}"
            f"[{event['event_type']}] {event['resource_id']}"
        )
        if event["event_type"] == "data_ingested":
            print(f"    {indent}  rows: {event['details'].get('rows', '?')}")
        elif event["event_type"] == "model_trained":
            metrics = event["details"].get("metrics", {})
            print(f"    {indent}  AUC: {metrics.get('auc_roc', '?')}")
        elif event["event_type"] == "model_deployed":
            print(
                f"    {indent}  "
                f"endpoint: {event['details'].get('endpoint', '?')}"
            )
        elif event["event_type"] == "prediction_made":
            print(
                f"    {indent}  "
                f"prediction: {event['details'].get('prediction', '?')}, "
                f"confidence: {event['details'].get('confidence', '?')}"
            )

    print(f"\n  Lineage depth: {len(chain)} events")
    print("  Key insight: Every prediction traces back to its training data")


def demo_verify_integrity(logger: AuditLogger):
    """Demo 4: Verify audit log integrity."""
    print("\n" + "=" * 60)
    print("Demo 4: Log Integrity Verification")
    print("=" * 60)

    result = logger.verify_integrity()

    print(f"\n  Entries checked: {result['entries_checked']}")
    print(f"  Integrity valid: {'✓ YES' if result['valid'] else '✗ NO'}")

    if result["errors"]:
        print(f"  Errors found: {len(result['errors'])}")
        for error in result["errors"]:
            print(f"    - {error}")


def demo_tamper_detection(logger: AuditLogger):
    """Demo 5: Demonstrate tamper detection by modifying a log entry."""
    print("\n" + "=" * 60)
    print("Demo 5: Tamper Detection")
    print("=" * 60)

    print("\n  Creating a copy of the log and modifying one entry...")

    # Create a tampered copy
    tampered_path = logger.log_path + ".tampered"
    entries = logger.read_log()

    with open(tampered_path, "w") as f:
        for i, entry in enumerate(entries):
            if i == 2:  # Tamper with the 3rd entry
                entry["event"]["actor"] = "unauthorized_user"
                entry["event"]["details"]["environment"] = "staging"
                print(f"  Tampered entry {i}: changed actor and environment")
            f.write(json.dumps(entry) + "\n")

    # Verify tampered log
    tampered_logger = AuditLogger(tampered_path)
    tampered_logger._event_counter = len(entries)

    result = tampered_logger.verify_integrity()

    print(f"\n  Verification of tampered log:")
    print(f"  Integrity valid: {'✓ YES' if result['valid'] else '✗ NO'}")

    if result["errors"]:
        print(f"  Errors detected: {len(result['errors'])}")
        for error in result["errors"][:3]:
            print(f"    - {error}")

    # Clean up
    os.remove(tampered_path)

    print("\n  The hash chain caught the modification because the")
    print("  tampered entry's event_hash no longer matches the chain.")


def demo_log_format(logger: AuditLogger):
    """Demo 6: Show the raw JSON Lines format."""
    print("\n" + "=" * 60)
    print("Demo 6: Raw Log Format (JSON Lines)")
    print("=" * 60)

    entries = logger.read_log()
    if not entries:
        print("  No entries to display.")
        return

    # Show first entry
    print("\n  First entry (formatted):\n")
    first = entries[0]
    formatted = json.dumps(first, indent=4)
    for line in formatted.split("\n"):
        print(f"  {line}")

    print(f"\n  Each line is a self-contained JSON object with:")
    print(f"    • sequence — monotonic counter for ordering")
    print(f"    • event — the full audit event record")
    print(f"    • event_hash — SHA-256 of event content (16 chars)")
    print(f"    • chain_hash — SHA-256(previous_hash + event_hash)")
    print(f"    • previous_hash — chain_hash of preceding entry")


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("Audit Trail & Lineage MWE — ML Governance Infrastructure")
    print("=" * 60)

    # Use a temp file for the audit log
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False
    ) as f:
        log_path = f.name

    try:
        logger = AuditLogger(log_path)

        demo_lifecycle_logging(logger)
        demo_read_log(logger)
        demo_lineage_traversal(logger)
        demo_verify_integrity(logger)
        demo_tamper_detection(logger)
        demo_log_format(logger)
    finally:
        # Clean up temp file
        if os.path.exists(log_path):
            os.remove(log_path)

    print("\n" + "=" * 60)
    print("MWE Complete — Audit trail and lineage demonstrated!")
    print("=" * 60)


if __name__ == "__main__":
    main()
