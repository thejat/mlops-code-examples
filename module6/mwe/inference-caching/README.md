# Inference Caching MWE

Demonstrates LLM response caching with deterministic cache keys, TTL-based eviction, cache bypass for non-deterministic requests, and hashed keys to avoid storing raw prompt text.

## Prerequisites

- Python 3.8+
- Any OS (Linux, macOS, Windows)

## Quick Start

```bash
# 1. Clone the repository
git clone <repo-url>
cd module6/mwe/inference-caching

# 2. No dependencies to install (uses only standard library)
# Optional: create a virtual environment
python -m venv venv && source venv/bin/activate

# 3. Run the demo
python main.py
```

## Expected Output

The demo runs four scenarios showing cache behavior:

```
============================================================
INFERENCE CACHING DEMO
============================================================

--- Demo 1: Cache Hits and Misses ---

  [                MISS]   301.2ms | What is machine learning?
  [                MISS]   300.8ms | Explain neural networks.
  [                 HIT]     0.1ms | What is machine learning?
  [                 HIT]     0.0ms | What is machine learning?
  [                 HIT]     0.0ms | Explain neural networks.
  [                MISS]   300.5ms | What is deep learning?

  Cache hit rate: 50%
  Entries in cache: 3

--- Demo 2: Cache Bypass (temperature > 0) ---

  [BYPASS (temperature > 0)]   300.9ms | temp=0.7 | Write a poem about AI.
  [BYPASS (temperature > 0)]   301.1ms | temp=0.7 | Write a poem about AI.
  [                MISS]   300.6ms | temp=0.0 | Write a poem about AI.
  [                 HIT]     0.0ms | temp=0.0 | Write a poem about AI.

--- Demo 3: TTL Expiration ---

  Immediately after caching:  [HIT]
  Waiting 2.5 seconds for TTL expiry...
  After TTL expiry:           [MISS (expired)]

--- Demo 4: Hashed Keys (no plaintext stored) ---

  Prompt:     'What is the meaning of life?'
  Cache key:  'llm:a1b2c3...'
  ...
```

## Key Concepts

| Concept | Description |
|---------|-------------|
| **Cache key hashing** | SHA-256 of normalized prompt + params — avoids storing raw text |
| **TTL eviction** | Entries expire after a configurable time (default: 2s in demo) |
| **Cache bypass** | Requests with `temperature > 0` skip cache entirely |
| **Hit rate** | `hits / (hits + misses)` — bypasses excluded from rate |

### Privacy Note

Hashing cache keys with SHA-256 avoids accidental plaintext exposure of user prompts in cache stores. However, unsalted deterministic hashing is **not** resistant to dictionary/brute-force recovery for predictable inputs. For sensitive data, use encryption, access control, or keyed hashing (HMAC). This demo focuses on the caching pattern, not on providing cryptographic privacy guarantees.

## Extension Challenges

1. **Semantic caching:** Use embedding similarity instead of exact-match keys
2. **Redis backend:** Replace the in-memory dict with Redis for persistence
3. **LRU eviction:** Add a max-size limit with least-recently-used eviction
4. **Cache warming:** Pre-populate cache with common queries at startup

## Related Materials

- Inference Caching Strategies (module6 slides, lines 587–732)
- Cache Key Design and Privacy
