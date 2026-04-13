# Metrics Instrumentation MWE

**Demonstrate Prometheus metric types for ML services — no running Prometheus server required.**

## Overview

This MWE shows how to instrument an ML inference service with Prometheus metrics using the `prometheus_client` library's in-memory registry:

1. **Counter** — track total requests and errors with labels
2. **Histogram** — measure latency and confidence distributions with custom buckets
3. **Gauge** — track point-in-time values like drift PSI and model accuracy
4. **Simulated traffic** — 200 requests with realistic error rates and latency patterns
5. **Exposition format** — the exact output Prometheus scrapes from `/metrics`

## Prerequisites

- Python 3.8+
- Linux, macOS, or Windows

## Setup (3 steps)

```bash
# 1. Navigate to the MWE directory
cd module8/mwe/metrics-instrumentation

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the demo
python main.py
```

## Expected Output

| Demo | What It Shows | Key Lesson |
|------|--------------|------------|
| 1. Simulate Traffic | 200 requests → counters and histograms | Counters track cumulative totals |
| 2. Drift Gauges | PSI values for 4 features | Gauges capture point-in-time values |
| 3. Read Metrics | Query in-memory registry | Metrics are queryable without Prometheus |
| 4. Exposition Format | Raw `/metrics` output | This is what Prometheus scrapes |
| 5. Type Guide | When to use each metric type | Counter vs Histogram vs Gauge decisions |

## Key Concepts Demonstrated

| Concept | Implementation |
|---------|----------------|
| Counter with labels | `REQUEST_COUNT.labels(status='success').inc()` |
| Histogram with custom buckets | `REQUEST_LATENCY.observe(latency)` with ML-tuned buckets |
| Gauge for drift scores | `FEATURE_DRIFT.labels(feature_name='age').set(0.18)` |
| Custom registry | `CollectorRegistry()` to avoid default system collectors |
| Exposition format | `generate_latest(registry)` produces Prometheus text format |

## Extension Challenges

1. Add LLM-specific metrics (TTFT histogram, token counter, cache hit counter)
2. Mount metrics on a FastAPI `/metrics` endpoint using `make_asgi_app()`
3. Add a `Summary` metric type and compare its behavior with `Histogram`
4. Implement the `instrument_inference` decorator from the lecture notes

## Connection to Final Project

This MWE directly supports **Component 1: Production Monitoring Dashboard**:

- The metric definitions show the instrumentation patterns you need
- The exposition format output is what Prometheus scrapes and Grafana queries
- The drift gauges connect monitoring to drift detection (Component 4)

## Related Reading

- Metrics Instrumentation (`mod8notes/instructional-materials/metrics-instrumentation.md`)
- Production Monitoring Fundamentals (`mod8notes/instructional-materials/production-monitoring-fundamentals.md`)
- Dashboard Design (`mod8notes/instructional-materials/dashboard-design.md`)
