# Dynamic Request Batching MWE

Batch multiple inference requests together to maximize GPU utilization by amortizing model weight loading costs across requests.

## Prerequisites

- Python 3.8+
- Any OS (Linux, macOS, Windows)

## Quick Start

```bash
# 1. Clone the repository
git clone <repo-url>
cd src/module6/learning-activities/mwe/request-batching

# 2. No dependencies to install (uses only standard library)
# Optional: create a virtual environment
python -m venv venv && source venv/bin/activate

# 3. Run the demo
python main.py
```

## Expected Output

The demo simulates 10 clients sending requests with staggered arrival times. Requests are grouped into batches of up to 4, demonstrating how batching reduces the number of "model loads":

```
=== Dynamic Request Batching Demo ===

[Batch 1] Processing 4 requests together
  Client 0 received: Response to 'Query from client 0' (batch 1)
  Client 1 received: Response to 'Query from client 1' (batch 1)
  Client 2 received: Response to 'Query from client 2' (batch 1)
  Client 3 received: Response to 'Query from client 3' (batch 1)
[Batch 2] Processing 4 requests together
  Client 4 received: Response to 'Query from client 4' (batch 2)
  ...

=== Summary ===
Total requests: 10
Batches processed: 3
Average batch size: 3.3
```

## Key Concepts

| Parameter | Description | Default |
|-----------|-------------|---------|
| `max_batch_size` | Maximum requests per batch | 4 |
| `max_wait_ms` | Timeout before processing partial batch | 100ms |

**Why batching matters:** LLM inference is memory-bandwidth bound. Each token requires loading model weights. Batching amortizes this cost-8 requests batched = 8x efficiency gain.

## Extension Challenge

1. **Add latency tracking:** Measure and print per-request latency (time from submission to result)
2. **Implement priority queues:** Allow high-priority requests to jump ahead
3. **Integrate with FastAPI:** Expose the batcher as an HTTP endpoint using `asyncio` background tasks

## Related Materials

- Dynamic Request Batching
- Asyncio Concurrency Patterns