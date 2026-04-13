"""
Metrics Instrumentation MWE - Prometheus Metric Types for ML Services

Demonstrates how to instrument an ML service with Prometheus metrics:
  1. Counter — track total requests and errors with labels
  2. Histogram — measure latency distributions with custom buckets
  3. Gauge — track point-in-time values like drift scores
  4. Simulated traffic — realistic request patterns with varying outcomes
  5. Registry introspection — read metrics back without a Prometheus server

No running Prometheus or Grafana instance required. The script uses the
prometheus_client library's in-memory registry to demonstrate metric
creation, updates, and exposition format output.

Run: python main.py
"""

import random
import time
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    CollectorRegistry,
    generate_latest,
)


# =============================================================================
# Metric Definitions (using a custom registry to avoid default collectors)
# =============================================================================

registry = CollectorRegistry()

# Counter: monotonically increasing count of events
REQUEST_COUNT = Counter(
    "ml_inference_requests_total",
    "Total number of inference requests",
    ["model_version", "status"],
    registry=registry,
)

PREDICTION_COUNT = Counter(
    "ml_predictions_total",
    "Total predictions by predicted class",
    ["model_version", "predicted_class"],
    registry=registry,
)

# Histogram: distribution of observed values
REQUEST_LATENCY = Histogram(
    "ml_inference_latency_seconds",
    "Inference request latency in seconds",
    ["model_version"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
    registry=registry,
)

PREDICTION_CONFIDENCE = Histogram(
    "ml_prediction_confidence",
    "Distribution of prediction confidence scores",
    ["model_version"],
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    registry=registry,
)

# Gauge: point-in-time value that can go up or down
FEATURE_DRIFT = Gauge(
    "ml_feature_drift_psi",
    "Current PSI drift score for input features",
    ["feature_name"],
    registry=registry,
)

MODEL_ACCURACY = Gauge(
    "ml_model_accuracy",
    "Current model accuracy estimate",
    ["model_version"],
    registry=registry,
)


# =============================================================================
# Simulated ML Service
# =============================================================================

def simulate_inference(model_version: str) -> dict:
    """
    Simulate a single ML inference request.

    Returns a dict with prediction, confidence, latency, and error status.
    Roughly 5% of requests produce errors.
    """
    # Simulate error with 5% probability
    if random.random() < 0.05:
        latency = random.uniform(0.5, 2.0)  # Errors tend to be slower
        return {
            "status": "error",
            "latency": latency,
            "prediction": None,
            "confidence": None,
        }

    # Simulate normal inference
    latency = random.gauss(0.12, 0.05)  # Mean 120ms, std 50ms
    latency = max(0.01, latency)  # Floor at 10ms

    # Simulate prediction
    predicted_class = random.choices(
        ["positive", "negative"],
        weights=[0.3, 0.7],  # 30% positive rate
    )[0]

    # Confidence tends to be high but with some low-confidence predictions
    if random.random() < 0.15:
        confidence = random.uniform(0.3, 0.6)  # Low confidence
    else:
        confidence = random.uniform(0.7, 0.99)  # High confidence

    return {
        "status": "success",
        "latency": latency,
        "prediction": predicted_class,
        "confidence": confidence,
    }


def record_metrics(result: dict, model_version: str):
    """Record a single inference result into Prometheus metrics."""
    # Always record request count and latency
    REQUEST_COUNT.labels(
        model_version=model_version,
        status=result["status"],
    ).inc()

    REQUEST_LATENCY.labels(
        model_version=model_version,
    ).observe(result["latency"])

    # Record prediction-specific metrics on success
    if result["status"] == "success":
        PREDICTION_COUNT.labels(
            model_version=model_version,
            predicted_class=result["prediction"],
        ).inc()
        PREDICTION_CONFIDENCE.labels(
            model_version=model_version,
        ).observe(result["confidence"])


# =============================================================================
# Demonstrations
# =============================================================================

def demo_simulate_traffic():
    """Demo 1: Simulate ML inference traffic and record metrics."""
    print("=" * 60)
    print("Demo 1: Simulating ML Inference Traffic")
    print("=" * 60)

    model_version = "v2.1.0"
    n_requests = 200

    print(f"\n  Simulating {n_requests} inference requests for model {model_version}...")

    successes = 0
    errors = 0

    for _ in range(n_requests):
        result = simulate_inference(model_version)
        record_metrics(result, model_version)

        if result["status"] == "success":
            successes += 1
        else:
            errors += 1

    print(f"  Completed: {successes} successes, {errors} errors")
    print(f"  Error rate: {errors / n_requests:.1%}")


def demo_drift_gauges():
    """Demo 2: Set drift gauge values for multiple features."""
    print("\n" + "=" * 60)
    print("Demo 2: Setting Drift Gauges")
    print("=" * 60)

    # Simulate PSI drift scores for different features
    drift_scores = {
        "age": 0.18,       # Moderate drift
        "income": 0.05,    # No drift
        "credit_score": 0.02,  # No drift
        "transaction_count": 0.31,  # Significant drift
    }

    print("\n  Setting feature drift PSI gauges:\n")
    print(f"  {'Feature':<20} {'PSI':>8} {'Status'}")
    print(f"  {'—' * 20} {'—' * 8} {'—' * 15}")

    for feature, psi in drift_scores.items():
        FEATURE_DRIFT.labels(feature_name=feature).set(psi)
        if psi < 0.1:
            status = "no drift"
        elif psi < 0.2:
            status = "investigate"
        else:
            status = "ACTION NEEDED"
        print(f"  {feature:<20} {psi:>8.2f} {status}")

    # Set model accuracy gauge
    MODEL_ACCURACY.labels(model_version="v2.1.0").set(0.847)
    print(f"\n  Model accuracy gauge set to 0.847")


def _parse_metric_value(exposition: str, metric_name: str) -> float:
    """Extract a single metric value from Prometheus exposition text."""
    for line in exposition.split("\n"):
        if line.startswith(metric_name):
            parts = line.rsplit(" ", 1)
            if len(parts) == 2:
                return float(parts[1])
    return 0.0


def demo_read_metrics():
    """Demo 3: Read metrics from the in-memory registry."""
    print("\n" + "=" * 60)
    print("Demo 3: Reading Metrics from Registry")
    print("=" * 60)

    print("\n  Querying in-memory metric values:\n")

    # Generate exposition text and parse it (works across prometheus_client versions)
    exposition = generate_latest(registry).decode("utf-8")

    # Read counter values
    success_count = _parse_metric_value(
        exposition,
        'ml_inference_requests_total{model_version="v2.1.0",status="success"}',
    )
    error_count = _parse_metric_value(
        exposition,
        'ml_inference_requests_total{model_version="v2.1.0",status="error"}',
    )

    print(f"  Counters:")
    print(f"    ml_inference_requests_total{{status='success'}}: {success_count:.0f}")
    print(f"    ml_inference_requests_total{{status='error'}}:   {error_count:.0f}")

    pos_count = _parse_metric_value(
        exposition,
        'ml_predictions_total{model_version="v2.1.0",predicted_class="positive"}',
    )
    neg_count = _parse_metric_value(
        exposition,
        'ml_predictions_total{model_version="v2.1.0",predicted_class="negative"}',
    )
    print(f"    ml_predictions_total{{class='positive'}}: {pos_count:.0f}")
    print(f"    ml_predictions_total{{class='negative'}}: {neg_count:.0f}")

    # Read histogram summary values (_sum and _count)
    latency_sum = _parse_metric_value(
        exposition,
        'ml_inference_latency_seconds_sum{model_version="v2.1.0"}',
    )
    latency_count = _parse_metric_value(
        exposition,
        'ml_inference_latency_seconds_count{model_version="v2.1.0"}',
    )

    if latency_count > 0:
        avg_latency = latency_sum / latency_count
        print(f"\n  Histograms:")
        print(f"    Average latency: {avg_latency * 1000:.1f} ms")
        print(f"    Total observations: {latency_count:.0f}")

    conf_sum = _parse_metric_value(
        exposition,
        'ml_prediction_confidence_sum{model_version="v2.1.0"}',
    )
    conf_count = _parse_metric_value(
        exposition,
        'ml_prediction_confidence_count{model_version="v2.1.0"}',
    )

    if conf_count > 0:
        avg_conf = conf_sum / conf_count
        print(f"    Average confidence: {avg_conf:.3f}")

    # Read gauge values
    drift_age = _parse_metric_value(
        exposition,
        'ml_feature_drift_psi{feature_name="age"}',
    )
    drift_txn = _parse_metric_value(
        exposition,
        'ml_feature_drift_psi{feature_name="transaction_count"}',
    )
    accuracy = _parse_metric_value(
        exposition,
        'ml_model_accuracy{model_version="v2.1.0"}',
    )

    print(f"\n  Gauges:")
    print(f"    ml_feature_drift_psi{{feature='age'}}: {drift_age:.2f}")
    print(f"    ml_feature_drift_psi{{feature='transaction_count'}}: {drift_txn:.2f}")
    print(f"    ml_model_accuracy{{model='v2.1.0'}}: {accuracy:.3f}")


def demo_exposition_format():
    """Demo 4: Show Prometheus exposition format output."""
    print("\n" + "=" * 60)
    print("Demo 4: Prometheus Exposition Format (what /metrics returns)")
    print("=" * 60)

    output = generate_latest(registry).decode("utf-8")

    # Show first 60 lines to keep output manageable
    lines = output.strip().split("\n")
    max_lines = 60

    print(f"\n  Showing first {min(max_lines, len(lines))} of {len(lines)} lines:\n")

    for line in lines[:max_lines]:
        print(f"  {line}")

    if len(lines) > max_lines:
        print(f"\n  ... ({len(lines) - max_lines} more lines)")

    print(f"\n  Total exposition output: {len(output)} bytes, {len(lines)} lines")
    print(f"  This is exactly what Prometheus would scrape from /metrics")


def demo_metric_types_summary():
    """Demo 5: Summary of metric types and when to use each."""
    print("\n" + "=" * 60)
    print("Demo 5: Metric Type Selection Guide")
    print("=" * 60)

    print("""
  Metric Type    When to Use                     ML Examples
  ———————————    ——————————————————————————       ————————————————————————
  Counter        Monotonically increasing         Request count, error count,
                 cumulative totals                prediction count by class

  Histogram      Distribution of values;          Inference latency, confidence
                 need percentiles (p50/p95/p99)   scores, token counts

  Gauge          Point-in-time snapshot;           Drift PSI, model accuracy,
                 value can go up or down           GPU utilization, queue depth

  Summary        Similar to Histogram but          (Less common in ML; prefer
                 calculates quantiles client-side  Histogram for Prometheus)

  Key decisions:
    • Use Counter for "how many?" → rate() in PromQL gives per-second rates
    • Use Histogram for "how long/big?" → histogram_quantile() gives percentiles
    • Use Gauge for "what is it now?" → direct value, no rate() needed
    • Add labels for segmentation, but avoid high-cardinality labels (no user IDs)
""")


# =============================================================================
# Main
# =============================================================================

def main():
    random.seed(42)

    print("=" * 60)
    print("Metrics Instrumentation MWE — Prometheus for ML Services")
    print("=" * 60)

    demo_simulate_traffic()
    demo_drift_gauges()
    demo_read_metrics()
    demo_exposition_format()
    demo_metric_types_summary()

    print("=" * 60)
    print("MWE Complete — Prometheus metric instrumentation demonstrated!")
    print("=" * 60)


if __name__ == "__main__":
    main()
