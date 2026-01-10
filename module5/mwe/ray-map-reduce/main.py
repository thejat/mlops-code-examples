#!/usr/bin/env python3
"""
Ray Map-Reduce Minimum Working Example

Demonstrates: Ray remote tasks, parallel execution, and map-reduce pattern.
Simulates distributed word counting across multiple text batches.

Run with: python main.py
"""

import ray
import time
from collections import Counter


# Define a remote function (runs in parallel across workers)
@ray.remote
def map_word_count(text_batch: str, batch_id: int) -> dict:
    """Map phase: Count words in a single batch."""
    print(f"  [Worker] Processing batch {batch_id}...")
    
    # Simulate some processing time
    time.sleep(0.5)
    
    # Count words in this batch
    words = text_batch.lower().split()
    word_counts = Counter(words)
    
    print(f"  [Worker] Batch {batch_id} complete: {len(words)} words, {len(word_counts)} unique")
    return dict(word_counts)


def reduce_counts(count_dicts: list[dict]) -> dict:
    """Reduce phase: Merge all word counts into one."""
    total_counts = Counter()
    for counts in count_dicts:
        total_counts.update(counts)
    return dict(total_counts)


def main():
    print("=" * 60)
    print("RAY MAP-REDUCE: Distributed Word Count Example")
    print("=" * 60)
    
    # Initialize Ray (starts local cluster if not connected)
    print("\n[1/4] Initializing Ray...")
    ray.init(ignore_reinit_error=True, logging_level="ERROR")
    print(f"  Ray initialized with {ray.available_resources().get('CPU', 0):.0f} CPUs")
    
    # Sample data: multiple text batches to process in parallel
    text_batches = [
        "the quick brown fox jumps over the lazy dog",
        "ray makes distributed computing easy and fast",
        "machine learning pipelines benefit from parallel processing",
        "the fox and the dog are friends in the forest",
        "distributed systems scale horizontally for big data",
    ]
    
    print(f"\n[2/4] Submitting {len(text_batches)} batches for parallel processing...")
    start_time = time.time()
    
    # MAP: Submit all tasks in parallel (non-blocking)
    futures = [
        map_word_count.remote(batch, i) 
        for i, batch in enumerate(text_batches)
    ]
    
    # Wait for all tasks to complete and collect results
    print("\n[3/4] Waiting for workers to complete...")
    batch_results = ray.get(futures)  # Blocks until all complete
    
    map_time = time.time() - start_time
    print(f"  All {len(batch_results)} batches processed in {map_time:.2f}s")
    
    # REDUCE: Aggregate all word counts
    print("\n[4/4] Reducing results...")
    final_counts = reduce_counts(batch_results)
    
    # Display results
    print("\n" + "=" * 60)
    print("RESULTS: Top 10 Most Frequent Words")
    print("=" * 60)
    
    sorted_words = sorted(final_counts.items(), key=lambda x: x[1], reverse=True)
    for word, count in sorted_words[:10]:
        bar = "█" * count
        print(f"  {word:15} {count:3} {bar}")
    
    print(f"\nTotal unique words: {len(final_counts)}")
    print(f"Total word occurrences: {sum(final_counts.values())}")
    
    # Cleanup
    ray.shutdown()
    print("\nRay shutdown complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()