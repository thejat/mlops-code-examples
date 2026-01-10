# Ray Map-Reduce - Minimum Working Example

**Pattern:** Parallelize Python functions across workers using Ray's `@ray.remote` decorator and the map-reduce pattern.

---

## Prerequisites

- Python 3.9+ (tested on 3.9, 3.10, 3.11)
- Linux or macOS (Windows requires WSL2)
- ~500MB disk space for Ray installation

---

## Setup (3 Steps)

### 1. Clone and Navigate

```bash
git clone <repository-url>
cd src/module5/learning-activities/mwe/ray-map-reduce
```

### 2. Install Dependencies

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install Ray
pip install -r requirements.txt
```

### 3. Run

```bash
python main.py
```

---

## Expected Output

```
============================================================
RAY MAP-REDUCE: Distributed Word Count Example
============================================================

[1/4] Initializing Ray...
  Ray initialized with 8 CPUs

[2/4] Submitting 5 batches for parallel processing...

[3/4] Waiting for workers to complete...
  [Worker] Processing batch 0...
  [Worker] Processing batch 1...
  [Worker] Processing batch 2...
  [Worker] Processing batch 3...
  [Worker] Processing batch 4...
  [Worker] Batch 2 complete: 7 words, 7 unique
  [Worker] Batch 0 complete: 9 words, 8 unique
  [Worker] Batch 1 complete: 7 words, 7 unique
  [Worker] Batch 3 complete: 10 words, 8 unique
  [Worker] Batch 4 complete: 7 words, 7 unique
  All 5 batches processed in 0.58s

[4/4] Reducing results...

============================================================
RESULTS: Top 10 Most Frequent Words
============================================================
  the               6 ██████
  fox               2 ██
  dog               2 ██
  and               2 ██
  distributed       2 ██
  for               2 ██
  quick             1 █
  brown             1 █
  jumps             1 █
  over              1 █

Total unique words: 30
Total word occurrences: 40

Ray shutdown complete.
============================================================
```

---

## Key Concepts Demonstrated

| Concept | Where in Code |
|---------|---------------|
| Remote Function | `@ray.remote` decorator on `map_word_count()` |
| Parallel Submission | `[func.remote(x) for x in data]` pattern |
| Non-blocking Calls | `.remote()` returns Future immediately |
| Blocking Collection | `ray.get(futures)` waits for all results |
| Map Phase | Each worker counts words in its batch |
| Reduce Phase | `reduce_counts()` merges all partial results |

---

## Extension Challenge

**Add Data Distribution with `ray.put()`:** Modify the example to use a shared lookup table for word normalization (e.g., stemming).

```python
# Store normalization map in object store once
stem_map = {"jumps": "jump", "computing": "compute", ...}
stem_ref = ray.put(stem_map)

# Pass reference to all workers (zero-copy on same node)
@ray.remote
def map_word_count(text_batch: str, batch_id: int, stem_ref) -> dict:
    stem_map = ray.get(stem_ref)  # Get from object store
    # Apply stemming before counting...
```

This demonstrates Ray's efficient shared memory for large objects.

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: ray` | Activate your virtual environment |
| `ray.init()` hangs | Check for existing Ray processes: `ray stop` |
| Low CPU count shown | Run on a machine with more cores for better parallelism |
| Worker errors | Check that all function dependencies are serializable |

---

## Resources

- [Ray Core Walkthrough](https://docs.ray.io/en/latest/ray-core/walkthrough.html)
- [Ray Tasks Documentation](https://docs.ray.io/en/latest/ray-core/tasks.html)
- [Ray Patterns Guide](https://docs.ray.io/en/latest/ray-core/patterns/index.html)