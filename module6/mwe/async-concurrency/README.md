# Async Concurrency Patterns MWE

Demonstrates Python asyncio patterns essential for LLM serving: sequential vs concurrent execution, the blocking sleep mistake, and semaphore-based concurrency limiting.

## Prerequisites

- Python 3.8+
- Any OS (Linux, macOS, Windows)

## Quick Start

```bash
# 1. Clone the repository
git clone <repo-url>
cd module6/mwe/async-concurrency

# 2. No dependencies to install (uses only standard library)

# 3. Run the demo
python main.py
```

## What It Demonstrates

| Demo | Pattern | Time (3 tasks × 1s) | Key Lesson |
|------|---------|---------------------|------------|
| Sequential | `await` one at a time | ~3.0s | Baseline — no concurrency |
| Concurrent | `asyncio.gather()` | ~1.0s | 3x speedup via concurrency |
| Blocking (bug) | `time.sleep()` in async | ~1.5s | Blocks event loop — no speedup |
| Semaphore | `asyncio.Semaphore(2)` | ~1.5s | Controlled max concurrency |

## Key Concepts

**Why async matters for LLM serving:**
- Network I/O (HTTP requests from clients)
- Database I/O (Redis cache lookups)
- Multiple concurrent user connections
- All benefit from non-blocking async patterns

**Common mistake to avoid:**
```python
# ❌ WRONG: Blocks the entire event loop
async def bad():
    time.sleep(5)  # Nothing else can run!

# ✅ CORRECT: Yields control while sleeping
async def good():
    await asyncio.sleep(5)  # Other coroutines run
```

## Extension Challenges

1. **Add asyncio.Lock:** Protect a shared counter from race conditions
2. **Background task:** Create a periodic health-check task that runs alongside request handling
3. **asyncio.Queue:** Implement a producer-consumer pattern for request batching

## Related Materials

- Python Async Fundamentals (module6 slides, lines 1093–1207)
- Asyncio Concurrency Patterns
- Connection Limiting with Semaphores
