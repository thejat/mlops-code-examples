# LLM Benchmark MWE

A minimal working example for benchmarking LLM inference APIs with batching and caching measurement capabilities.

## Purpose

This benchmark runner helps you:

1. **Measure LLM-specific metrics:** TTFT (Time to First Token), TPOT (Time Per Output Token), throughput
2. **Test batching effectiveness:** Run concurrent requests to exercise your batching implementation
3. **Validate caching:** Use repeated prompts to measure cache hit rates
4. **Generate reproducible results:** CLI-driven with configurable parameters

## Prerequisites

- Python 3.10+
- An LLM inference API running locally (e.g., your Milestone 5 FastAPI server)

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Benchmark

```bash
# Run 50 requests with default settings
python main.py --endpoint http://localhost:8000/generate
```

### High Concurrency Test

```bash
# Test batching with 20 concurrent requests
python main.py --endpoint http://localhost:8000/generate \
    --concurrency 20 \
    --requests 200
```

### Cache Effectiveness Test

```bash
# Use 50% repeated prompts to test caching
python main.py --endpoint http://localhost:8000/generate \
    --repeat-ratio 0.5 \
    --requests 100
```

### Save Results

```bash
# Export results to JSON for analysis
python main.py --endpoint http://localhost:8000/generate \
    --output results.json
```

## CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--endpoint` | `http://localhost:8000/generate` | LLM inference API URL |
| `--requests` | `50` | Total number of requests |
| `--concurrency` | `5` | Maximum concurrent requests |
| `--max-tokens` | `100` | Tokens to generate per request |
| `--repeat-ratio` | `0.3` | Ratio of repeated prompts (for cache testing) |
| `--warmup` | `5` | Warmup requests to discard |
| `--output` | None | JSON output file path |

## Expected API Format

Your inference endpoint should accept POST requests with this JSON body:

```json
{
    "prompt": "Your prompt here",
    "max_tokens": 100,
    "temperature": 0.0
}
```

And return:

```json
{
    "text": "Generated response text",
    "cached": false
}
```

## Metrics Explained

| Metric | Definition | Why It Matters |
|--------|------------|----------------|
| **TTFT P50/P95/P99** | Time to First Token percentiles | User-perceived responsiveness |
| **TPOT** | Time Per Output Token | Generation speed after first token |
| **Throughput** | Total tokens/second | System capacity |
| **Cache Hit Rate** | % of requests served from cache | Cost efficiency |
| **Avg Latency** | Mean end-to-end response time | Overall performance |

## Sample Output

See `expected_output/sample_output.txt` for example benchmark results.

## Troubleshooting

### Connection refused

Ensure your LLM server is running:

```bash
uvicorn src.server:app --port 8000
```

### Timeouts

Increase timeout or reduce concurrency:

```bash
python main.py --concurrency 2 --requests 20
```

### All requests fail

Check your endpoint URL and API format match the expected schema.

## Related Files

- LLM Benchmarking - Concepts
- Dynamic Request Batching - Batching implementation
- Inference Caching - Caching implementation