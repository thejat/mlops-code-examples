#!/usr/bin/env python3
"""
Async Concurrency Patterns MWE

Demonstrates Python asyncio patterns for LLM serving:
- Sequential vs concurrent execution with asyncio.gather()
- The blocking mistake (time.sleep vs asyncio.sleep)
- Semaphore-based concurrency limiting
- Timing comparison showing speedups

Usage:
    python main.py
"""

import asyncio
import time


async def simulate_inference(task_name: str, duration: float) -> str:
    """Simulate an async inference call (non-blocking)."""
    await asyncio.sleep(duration)
    return f"{task_name}: completed in {duration}s"


def simulate_inference_blocking(task_name: str, duration: float) -> str:
    """Simulate a BLOCKING inference call (wrong pattern)."""
    time.sleep(duration)
    return f"{task_name}: completed in {duration}s"


async def demo_sequential():
    """Demo 1: Sequential execution — tasks run one after another."""
    print("--- Demo 1: Sequential Execution ---\n")
    print("  Running 3 tasks sequentially (1s each)...")

    start = time.perf_counter()

    result1 = await simulate_inference("Task 1", 1.0)
    result2 = await simulate_inference("Task 2", 1.0)
    result3 = await simulate_inference("Task 3", 1.0)

    elapsed = time.perf_counter() - start

    for r in [result1, result2, result3]:
        print(f"    {r}")
    print(f"\n  Total time: {elapsed:.2f}s (expected ~3.0s)")
    print(f"  Explanation: Each await completes before the next starts.\n")

    return elapsed


async def demo_concurrent():
    """Demo 2: Concurrent execution — tasks run simultaneously."""
    print("--- Demo 2: Concurrent Execution (asyncio.gather) ---\n")
    print("  Running 3 tasks concurrently (1s each)...")

    start = time.perf_counter()

    results = await asyncio.gather(
        simulate_inference("Task 1", 1.0),
        simulate_inference("Task 2", 1.0),
        simulate_inference("Task 3", 1.0),
    )

    elapsed = time.perf_counter() - start

    for r in results:
        print(f"    {r}")
    print(f"\n  Total time: {elapsed:.2f}s (expected ~1.0s)")
    print(f"  Explanation: All tasks run during the same await period.\n")

    return elapsed


async def demo_blocking_mistake():
    """Demo 3: The blocking mistake — time.sleep blocks the event loop."""
    print("--- Demo 3: Common Mistake (blocking sleep) ---\n")
    print("  Running 3 tasks with time.sleep() inside asyncio.gather()...")
    print("  (This should be concurrent but ISN'T because sleep blocks!)\n")

    async def blocking_task(name: str, duration: float) -> str:
        # WRONG: This blocks the entire event loop!
        time.sleep(duration)
        return f"{name}: completed in {duration}s"

    start = time.perf_counter()

    results = await asyncio.gather(
        blocking_task("Task 1", 0.5),
        blocking_task("Task 2", 0.5),
        blocking_task("Task 3", 0.5),
    )

    elapsed = time.perf_counter() - start

    for r in results:
        print(f"    {r}")
    print(f"\n  Total time: {elapsed:.2f}s (expected ~0.5s, got ~1.5s!)")
    print(f"  Explanation: time.sleep() blocks the event loop — nothing")
    print(f"  else can run. Use asyncio.sleep() for async code.\n")
    print(f"  ❌ WRONG: time.sleep(5)     # Blocks everything")
    print(f"  ✅ RIGHT: await asyncio.sleep(5)  # Yields control\n")

    return elapsed


async def demo_semaphore():
    """Demo 4: Semaphore limits concurrency (e.g., max GPU connections)."""
    print("--- Demo 4: Semaphore-Limited Concurrency ---\n")

    max_concurrent = 2
    semaphore = asyncio.Semaphore(max_concurrent)
    total_tasks = 6

    print(f"  Running {total_tasks} tasks with max {max_concurrent} concurrent...")
    print(f"  (Simulates limiting concurrent model inference calls)\n")

    async def limited_task(task_id: int) -> str:
        async with semaphore:
            start = time.perf_counter()
            await asyncio.sleep(0.5)  # simulate inference
            elapsed = time.perf_counter() - start
            result = f"Task {task_id}: ran for {elapsed:.2f}s"
            print(f"    {result}")
            return result

    start = time.perf_counter()

    await asyncio.gather(*(limited_task(i) for i in range(total_tasks)))

    elapsed = time.perf_counter() - start

    # With 6 tasks, 2 concurrent, 0.5s each → 3 batches → ~1.5s
    print(f"\n  Total time: {elapsed:.2f}s")
    print(f"  Expected: ~{total_tasks / max_concurrent * 0.5:.1f}s "
          f"({total_tasks} tasks / {max_concurrent} concurrent × 0.5s each)")
    print(f"  Without semaphore: ~0.5s (all 6 concurrent)")
    print(f"\n  Use case: Limit concurrent connections to Redis, GPU, or")
    print(f"  external APIs to avoid resource exhaustion.\n")

    return elapsed


async def main():
    print("=" * 60)
    print("ASYNC CONCURRENCY PATTERNS DEMO")
    print("=" * 60)
    print()

    t_seq = await demo_sequential()
    t_conc = await demo_concurrent()
    t_block = await demo_blocking_mistake()
    t_sem = await demo_semaphore()

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Sequential:     {t_seq:.2f}s  (baseline)")
    print(f"  Concurrent:     {t_conc:.2f}s  ({t_seq / t_conc:.1f}x speedup)")
    print(f"  Blocking (bug): {t_block:.2f}s  (no speedup — event loop blocked)")
    print(f"  Semaphore (2):  {t_sem:.2f}s  (controlled concurrency)")
    print()
    print("  Key takeaway: Use await + asyncio.gather() for concurrent I/O.")
    print("  Never use time.sleep() in async code — it blocks everything.")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
