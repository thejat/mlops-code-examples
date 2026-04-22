"""
Prometheus + Grafana Stack MWE — ML Inference Service with Metrics

A FastAPI application that simulates an ML inference service and exposes
Prometheus metrics on /metrics. Designed to be scraped by Prometheus and
visualized in Grafana via docker-compose.

Endpoints:
    GET  /health           — Liveness check
    POST /predict           — Simulate an ML prediction
    GET  /metrics           — Prometheus exposition format
    POST /simulate-traffic  — Generate bulk requests for demo purposes

Run standalone:  uvicorn app:app --host 0.0.0.0 --port 8000
Run with stack:  docker compose up
"""

import random
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    CollectorRegistry,
    generate_latest,
    make_asgi_app,
    CONTENT_TYPE_LATEST,
)
from starlette.responses import Response


# =============================================================================
# Prometheus Metrics (custom registry to avoid default system collectors)
# =============================================================================

registry = CollectorRegistry()

REQUEST_COUNT = Counter(
    "ml_inference_requests_total",
    "Total inference requests",
    ["model_version", "status"],
    registry=registry,
)

PREDICTION_COUNT = Counter(
    "ml_predictions_total",
    "Total predictions by class",
    ["model_version", "predicted_class"],
    registry=registry,
)

REQUEST_LATENCY = Histogram(
    "ml_inference_latency_seconds",
    "Inference latency in seconds",
    ["model_version"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
    registry=registry,
)

PREDICTION_CONFIDENCE = Histogram(
    "ml_prediction_confidence",
    "Prediction confidence score distribution",
    ["model_version"],
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    registry=registry,
)

FEATURE_DRIFT = Gauge(
    "ml_feature_drift_psi",
    "PSI drift score per feature",
    ["feature_name"],
    registry=registry,
)

MODEL_ACCURACY = Gauge(
    "ml_model_accuracy",
    "Current model accuracy estimate",
    ["model_version"],
    registry=registry,
)

ACTIVE_REQUESTS = Gauge(
    "ml_active_requests",
    "Number of in-flight inference requests",
    registry=registry,
)


# =============================================================================
# Configuration
# =============================================================================

MODEL_VERSION = "v2.1.0"

DRIFT_SCORES = {
    "age": 0.18,
    "income": 0.05,
    "credit_score": 0.02,
    "transaction_count": 0.31,
}


# =============================================================================
# Simulated ML Inference
# =============================================================================

def simulate_inference() -> dict:
    """Simulate a single ML inference with realistic latency and error rates."""
    # 5% error rate
    if random.random() < 0.05:
        latency = random.uniform(0.5, 2.0)
        time.sleep(latency * 0.01)  # Scale down sleep for responsiveness
        return {
            "status": "error",
            "latency": latency,
            "prediction": None,
            "confidence": None,
            "error": "Model inference timeout",
        }

    # Normal inference
    latency = max(0.01, random.gauss(0.12, 0.05))
    time.sleep(latency * 0.1)  # Scale down sleep for responsiveness

    predicted_class = random.choices(
        ["positive", "negative"],
        weights=[0.3, 0.7],
    )[0]

    if random.random() < 0.15:
        confidence = random.uniform(0.3, 0.6)
    else:
        confidence = random.uniform(0.7, 0.99)

    return {
        "status": "success",
        "latency": latency,
        "prediction": predicted_class,
        "confidence": round(confidence, 4),
    }


def record_metrics(result: dict):
    """Record inference result into Prometheus metrics."""
    REQUEST_COUNT.labels(
        model_version=MODEL_VERSION,
        status=result["status"],
    ).inc()

    REQUEST_LATENCY.labels(
        model_version=MODEL_VERSION,
    ).observe(result["latency"])

    if result["status"] == "success":
        PREDICTION_COUNT.labels(
            model_version=MODEL_VERSION,
            predicted_class=result["prediction"],
        ).inc()
        PREDICTION_CONFIDENCE.labels(
            model_version=MODEL_VERSION,
        ).observe(result["confidence"])


# =============================================================================
# App Lifecycle
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Set initial gauge values on startup."""
    # Set initial drift scores
    for feature, psi in DRIFT_SCORES.items():
        FEATURE_DRIFT.labels(feature_name=feature).set(psi)

    # Set initial accuracy
    MODEL_ACCURACY.labels(model_version=MODEL_VERSION).set(0.847)

    print(f"ML service started — model {MODEL_VERSION}")
    print(f"Metrics available at http://localhost:8000/metrics")
    yield
    print("ML service shutting down")


# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title="ML Inference Service",
    description="Demo ML service with Prometheus metrics for the monitoring stack MWE",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    """Liveness/readiness probe."""
    return {"status": "healthy", "model_version": MODEL_VERSION}


@app.post("/predict")
async def predict(request: Request):
    """
    Simulate an ML prediction.

    Accepts any JSON body (treated as feature input) and returns
    a prediction with confidence score. Metrics are recorded automatically.
    """
    ACTIVE_REQUESTS.inc()
    try:
        result = simulate_inference()
        record_metrics(result)

        if result["status"] == "error":
            return JSONResponse(
                status_code=500,
                content={
                    "status": "error",
                    "error": result["error"],
                    "model_version": MODEL_VERSION,
                },
            )

        return {
            "status": "success",
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "model_version": MODEL_VERSION,
        }
    finally:
        ACTIVE_REQUESTS.dec()


@app.post("/simulate-traffic")
async def simulate_traffic(n: int = 100):
    """
    Generate n simulated inference requests.

    Useful for quickly populating Prometheus with data to visualize in Grafana.
    """
    successes = 0
    errors = 0

    for _ in range(min(n, 1000)):  # Cap at 1000
        result = simulate_inference()
        record_metrics(result)
        if result["status"] == "success":
            successes += 1
        else:
            errors += 1

        # Periodically jitter drift gauges to make dashboards dynamic
        if random.random() < 0.05:
            for feature in DRIFT_SCORES:
                base = DRIFT_SCORES[feature]
                jitter = random.uniform(-0.03, 0.03)
                FEATURE_DRIFT.labels(feature_name=feature).set(
                    max(0.0, base + jitter)
                )

    return {
        "generated": successes + errors,
        "successes": successes,
        "errors": errors,
        "error_rate": f"{errors / (successes + errors):.1%}",
    }


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    data = generate_latest(registry)
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)
