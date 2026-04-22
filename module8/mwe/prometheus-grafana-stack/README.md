# Prometheus + Grafana Monitoring Stack MWE

**Spin up a complete ML monitoring stack with one command: service → Prometheus → Grafana.**

## Overview

This MWE runs three containers via Docker Compose:

1. **ML Service** (FastAPI) — exposes mock ML inference metrics on `/metrics`
2. **Prometheus** — scrapes the service every 5 seconds
3. **Grafana** — visualizes metrics with a pre-built dashboard

```
ML Service :8000/metrics  →  Prometheus :9090  →  Grafana :3000
```

## Prerequisites

- Docker and Docker Compose (v2)
- Ports 3000, 8000, 9090 available

## Setup (2 steps)

```bash
# 1. Navigate to the MWE directory
cd module8/mwe/prometheus-grafana-stack

# 2. Build and start the stack
docker compose up --build
```

Wait ~15 seconds for all services to initialize, then open:

| Service    | URL                          | Credentials       |
|------------|------------------------------|--------------------|
| ML Service | http://localhost:8000/health  | —                  |
| Prometheus | http://localhost:9090         | —                  |
| Grafana    | http://localhost:3000         | admin / admin      |

## Generate Demo Data

The ML service starts with zero traffic. Generate some data to see the dashboard light up:

```bash
# Generate 500 simulated inference requests
curl -X POST "http://localhost:8000/simulate-traffic?n=500"

# Or send individual predictions
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{}'
```

Run the simulate-traffic call a few times with 30-second gaps so Prometheus captures rate changes.

## Using Grafana

1. Open http://localhost:3000 and log in with `admin` / `admin` (skip password change)
2. Go to **Dashboards** in the left sidebar
3. Open **ML Model Monitoring** — the pre-provisioned dashboard

The dashboard shows 9 panels:

| Panel                        | PromQL                                                                 |
|------------------------------|------------------------------------------------------------------------|
| Request Rate                 | `rate(ml_inference_requests_total[1m])`                                |
| Total Requests               | `sum(ml_inference_requests_total)`                                     |
| Error Rate                   | error count / total count ratio                                        |
| Model Accuracy               | `ml_model_accuracy`                                                    |
| Active Requests              | `ml_active_requests`                                                   |
| Latency p50/p95/p99          | `histogram_quantile(0.95, ...ml_inference_latency_seconds_bucket...)`  |
| Feature Drift (PSI)          | `ml_feature_drift_psi`                                                 |
| Prediction Confidence        | `histogram_quantile(0.50, ...ml_prediction_confidence_bucket...)`      |
| Predictions by Class         | `rate(ml_predictions_total[1m])`                                       |

## Exploring Prometheus Directly

Open http://localhost:9090 and try these PromQL queries:

```promql
# Raw counter
ml_inference_requests_total

# Request rate per second
rate(ml_inference_requests_total[1m])

# p95 latency
histogram_quantile(0.95, sum(rate(ml_inference_latency_seconds_bucket[5m])) by (le))

# Current drift scores
ml_feature_drift_psi

# Model accuracy gauge
ml_model_accuracy
```

## Service Endpoints

| Method | Endpoint            | Description                                 |
|--------|---------------------|---------------------------------------------|
| GET    | `/health`           | Liveness check                              |
| POST   | `/predict`          | Simulate a single ML prediction             |
| POST   | `/simulate-traffic` | Generate bulk requests (`?n=500`)           |
| GET    | `/metrics`          | Prometheus exposition format                |

## Teardown

```bash
docker compose down
```

## File Structure

```
prometheus-grafana-stack/
├── app.py                                    # FastAPI ML service
├── Dockerfile                                # Container for the ML service
├── docker-compose.yml                        # Orchestrates all 3 services
├── requirements.txt                          # Python dependencies
├── prometheus/
│   └── prometheus.yml                        # Scrape config
└── grafana/
    ├── dashboards/
    │   └── ml-monitoring.json                # Pre-built dashboard
    └── provisioning/
        ├── dashboards/
        │   └── dashboard-provider.yml        # Auto-load dashboards
        └── datasources/
            └── datasource.yml                # Prometheus data source
```

## Key Concepts Demonstrated

| Concept                  | How It Works Here                                      |
|--------------------------|--------------------------------------------------------|
| Metrics exposition       | FastAPI `/metrics` endpoint serves Prometheus format   |
| Scrape-based collection  | Prometheus pulls from targets every 5s                 |
| Service discovery        | Docker Compose DNS (`app:8000`) as scrape target       |
| Provisioned datasource   | Grafana auto-configures Prometheus on startup           |
| Provisioned dashboard    | Dashboard JSON loaded from file, no manual setup       |
| Counter → rate()         | Raw counters transformed to per-second rates           |
| Histogram → percentiles  | `histogram_quantile()` extracts p50/p95/p99            |
| Gauge for drift          | PSI scores displayed directly without rate()           |

## Connection to Final Project

This MWE directly supports **Component 1: Production Monitoring Dashboard**:

- The full stack (service + Prometheus + Grafana) is the same architecture used in production
- The pre-built dashboard demonstrates the panels expected in the final project
- The PromQL queries show how to derive rates, percentiles, and error ratios
- The provisioning configs show how to make dashboards reproducible

## Extension Challenges

1. Add an Alertmanager service that fires when `ml_feature_drift_psi > 0.2`
2. Add a second ML service (v3.0.0) and compare metrics across model versions
3. Add `ml_inference_batch_size` histogram and visualize throughput vs latency
4. Replace the simulated inference with a real scikit-learn model
