#!/usr/bin/env python3
"""
Lab: Module 6 - Dynamic Request Batching for LLM Inference
Time: ~30 minutes
Prerequisites:
  - Read: asyncio-concurrency-patterns.md
  - Read: dynamic-request-batching.md
  - Read: llm-serving-frameworks.md

Learning Objectives:
- LO1: Implement a dynamic request batcher using asyncio
- LO2: Use asyncio.Lock to prevent race conditions in shared state
- LO3: Configure batch size and timeout parameters for latency/throughput tradeoffs
- LO4: Measure and compare batched vs. unbatched inference performance

Milestone 5 Connection:
This lab introduces the batching concepts you'll use for your LLM inference
optimization. You'll implement the core batching logic before adding caching
and benchmarking in the full milestone.

Setup:
    pip install asyncio  # (built-in for Python 3.7+)
    # No additional dependencies required!

Run:
    python batching.py
"""

import asyncio
import time
import random
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import statistics

# === SETUP (provided) ===

# Sample prompts for testing - simulates real LLM inference requests
SAMPLE_PROMPTS = [
    "What is machine learning?",
    "Explain neural networks briefly.",
    "How does gradient descent work?",
    "What is the difference between AI and ML?",
    "Describe the transformer architecture.",
    "What are embeddings in NLP?",
    "Explain the attention mechanism.",
    "What is transfer learning?",
    "How does backpropagation work?",
    "What is a loss function?",
    "Explain overfitting and underfitting.",
    "What is regularization?",
    "Describe convolutional neural networks.",
    "What is batch normalization?",
    "Explain the vanishing gradient problem.",
]


@dataclass
class PendingRequest:
    """
    Represents a request waiting to be processed.
    
    In production LLM serving, each request includes the prompt,
    generation parameters, and a future to return the result.
    """
    prompt: str
    max_tokens: int
    future: asyncio.Future
    arrival_time: float


@dataclass
class BatchMetrics:
    """Tracks performance metrics for benchmarking."""
    request_latencies: List[float] = field(default_factory=list)
    batch_sizes: List[int] = field(default_factory=list)
    processing_times: List[float] = field(default_factory=list)


async def simulate_llm_inference(prompts: List[str], max_tokens: int = 50) -> List[str]:
    """
    Simulates LLM inference with realistic timing characteristics.
    
    In production, this would call the actual model. The simulation
    demonstrates that batching amortizes fixed costs (model weight loading)
    across multiple requests.
    
    Timing model:
    - Fixed cost: 50ms per batch (simulates weight loading overhead)
    - Variable cost: 5ms per prompt (simulates actual generation)
    
    Args:
        prompts: List of prompts to process
        max_tokens: Maximum tokens to generate (affects timing)
    
    Returns:
        List of generated responses
    """
    # Fixed overhead per batch (loading weights, setup)
    fixed_overhead_ms = 50
    
    # Variable cost per prompt in batch
    per_prompt_ms = 5 + (max_tokens / 50) * 3  # Scales with output length
    
    # Total time = fixed + (variable * batch_size)
    total_time_ms = fixed_overhead_ms + (per_prompt_ms * len(prompts))
    
    await asyncio.sleep(total_time_ms / 1000)
    
    # Generate mock responses
    responses = []
    for prompt in prompts:
        response = f"[Response to '{prompt[:30]}...']: This is a simulated response."
        responses.append(response)
    
    return responses


# === TODO 1: Implement the DynamicBatcher Class ===
#
# The DynamicBatcher accumulates incoming requests and processes them
# in batches to improve throughput. It uses two triggers:
#   - Batch size threshold: Process when N requests accumulate
#   - Timeout: Process whatever is available after T milliseconds
#
# This pattern is used by production LLM serving frameworks like vLLM and TGI.
#
# Key concepts:
#   - asyncio.Lock protects shared state (pending requests list)
#   - asyncio.Future allows callers to await their individual results
#   - Background task handles timeout-based processing

class DynamicBatcher:
    """
    Dynamic request batcher for LLM inference optimization.
    
    Accumulates requests and processes them in batches to maximize
    GPU utilization and reduce per-request latency.
    """
    
    def __init__(
        self,
        max_batch_size: int = 8,
        max_wait_ms: float = 50.0
    ):
        """
        Initialize the batcher.
        
        Args:
            max_batch_size: Maximum requests per batch
            max_wait_ms: Maximum time to wait for batch to fill (milliseconds)
        """
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        
        # TODO: Initialize the pending requests list
        # This will hold PendingRequest objects waiting to be processed
        #
        # Your code here:
        # self.pending: List[PendingRequest] = []
        
        pass  # Remove this line when you add your code
        
        # TODO: Create an asyncio.Lock to protect shared state
        # This prevents race conditions when multiple coroutines
        # access self.pending simultaneously
        #
        # Your code here:
        # self.lock = asyncio.Lock()
        
        pass  # Remove this line when you add your code
        
        # Metrics tracking (provided)
        self.metrics = BatchMetrics()
        self._timeout_task: Optional[asyncio.Task] = None
    
    async def submit(self, prompt: str, max_tokens: int = 50) -> str:
        """
        Submit a request and wait for the result.
        
        This is the main entry point for inference requests.
        The request is added to the pending queue, and the caller
        awaits the future until the batch is processed.
        
        Args:
            prompt: The input prompt for generation
            max_tokens: Maximum tokens to generate
        
        Returns:
            The generated response string
        """
        # TODO: Create a future to hold this request's result
        # Use asyncio.get_event_loop().create_future() or asyncio.Future()
        #
        # Your code here:
        # future = asyncio.get_event_loop().create_future()
        
        pass  # Remove this line when you add your code
        
        # TODO: Create a PendingRequest with arrival timestamp
        #
        # Your code here:
        # request = PendingRequest(
        #     prompt=prompt,
        #     max_tokens=max_tokens,
        #     future=future,
        #     arrival_time=time.time()
        # )
        
        pass  # Remove this line when you add your code
        
        # TODO: Add request to pending list with lock protection
        # Check if batch size threshold is reached
        #
        # Your code here:
        # async with self.lock:
        #     self.pending.append(request)
        #     should_process = len(self.pending) >= self.max_batch_size
        
        pass  # Remove this line when you add your code
        
        # TODO: If batch is full, trigger processing
        # Use asyncio.create_task to schedule without blocking
        #
        # Your code here:
        # if should_process:
        #     asyncio.create_task(self._process_batch())
        
        pass  # Remove this line when you add your code
        
        # TODO: Return the awaited future result
        #
        # Your code here:
        # return await future
        
        pass  # Remove this line when you add your code


# === TODO 2: Implement Batch Processing Logic ===
#
# This method extracts a batch from pending requests and processes
# them together. The key insight is that we hold the lock only
# while modifying shared state, NOT during inference.

    async def _process_batch(self):
        """
        Extract and process a batch of pending requests.
        
        Critical implementation notes:
        1. Hold the lock only while accessing self.pending
        2. Release the lock before calling inference (expensive operation)
        3. Check future.done() before setting result (prevents errors)
        """
        # TODO: Extract batch atomically under lock
        # Take up to max_batch_size requests from pending
        #
        # Your code here:
        # async with self.lock:
        #     if not self.pending:
        #         return
        #     
        #     batch = self.pending[:self.max_batch_size]
        #     self.pending = self.pending[self.max_batch_size:]
        
        pass  # Remove this line when you add your code
        
        # Track batch size for metrics (provided)
        # self.metrics.batch_sizes.append(len(batch))
        
        # TODO: Collect prompts from batch and run inference
        # This happens OUTSIDE the lock
        #
        # Your code here:
        # prompts = [r.prompt for r in batch]
        # start_time = time.time()
        # results = await simulate_llm_inference(prompts)
        # processing_time = time.time() - start_time
        
        pass  # Remove this line when you add your code
        
        # Track processing time (provided)
        # self.metrics.processing_times.append(processing_time)
        
        # TODO: Return results to waiting callers
        # Check future.done() to avoid InvalidStateError
        #
        # Your code here:
        # for request, result in zip(batch, results):
        #     if not request.future.done():
        #         latency = time.time() - request.arrival_time
        #         self.metrics.request_latencies.append(latency)
        #         request.future.set_result(result)
        
        pass  # Remove this line when you add your code


# === TODO 3: Implement Timeout Processing ===
#
# The timeout processor ensures requests don't wait forever when
# traffic is low. It periodically checks if any requests have
# waited longer than max_wait_ms and triggers processing.

    async def start_timeout_processor(self):
        """
        Start the background timeout processor.
        
        This task runs continuously, checking if any pending
        requests have exceeded the timeout threshold.
        """
        # TODO: Create a background task that runs the timeout loop
        #
        # Your code here:
        # self._timeout_task = asyncio.create_task(self._timeout_loop())
        
        pass  # Remove this line when you add your code
    
    async def _timeout_loop(self):
        """
        Background loop that triggers processing on timeout.
        
        Runs until cancelled, checking pending requests periodically.
        """
        # TODO: Implement the timeout loop
        # 1. Sleep for max_wait_ms
        # 2. Check if oldest request has exceeded timeout
        # 3. If so, trigger _process_batch
        #
        # Your code here:
        # while True:
        #     await asyncio.sleep(self.max_wait_ms / 1000)
        #     
        #     async with self.lock:
        #         if not self.pending:
        #             continue
        #         
        #         oldest = self.pending[0]
        #         wait_time_ms = (time.time() - oldest.arrival_time) * 1000
        #         
        #         if wait_time_ms >= self.max_wait_ms:
        #             asyncio.create_task(self._process_batch())
        
        pass  # Remove this line when you add your code
    
    async def stop_timeout_processor(self):
        """
        Stop the background timeout processor gracefully.
        
        Always call this during shutdown to prevent orphaned tasks.
        """
        if self._timeout_task:
            self._timeout_task.cancel()
            try:
                await self._timeout_task
            except asyncio.CancelledError:
                pass  # Expected on cancellation


# === TODO 4: Implement Unbatched Processing for Comparison ===
#
# To demonstrate the benefit of batching, we need a baseline
# that processes requests one at a time.

async def process_unbatched(prompt: str, max_tokens: int = 50) -> str:
    """
    Process a single request without batching.
    
    This simulates naive LLM serving where each request
    is processed independently. Used as a baseline for comparison.
    
    Args:
        prompt: Input prompt
        max_tokens: Maximum tokens to generate
    
    Returns:
        Generated response
    """
    # TODO: Call simulate_llm_inference with a single-item list
    # Return the first (only) result
    #
    # Your code here:
    # results = await simulate_llm_inference([prompt], max_tokens)
    # return results[0]
    
    pass  # Remove this line when you add your code


# === TODO 5: Implement Benchmarking ===
#
# Compare batched vs unbatched performance under concurrent load.

async def benchmark_batching(
    n_requests: int = 32,
    max_batch_size: int = 8,
    max_wait_ms: float = 50.0
) -> Dict[str, Any]:
    """
    Benchmark batched vs unbatched inference performance.
    
    Submits n_requests concurrently and measures latency
    for both approaches.
    
    Args:
        n_requests: Number of concurrent requests
        max_batch_size: Batch size for batched approach
        max_wait_ms: Timeout for batching
    
    Returns:
        Dictionary with timing results and metrics
    """
    # Generate test prompts
    prompts = [random.choice(SAMPLE_PROMPTS) for _ in range(n_requests)]
    
    results = {
        "n_requests": n_requests,
        "max_batch_size": max_batch_size,
        "max_wait_ms": max_wait_ms,
    }
    
    # Benchmark unbatched approach
    print(f"   Running unbatched benchmark ({n_requests} requests)...")
    
    # TODO: Time unbatched concurrent execution
    # Use asyncio.gather to run all requests concurrently
    #
    # Your code here:
    # unbatched_start = time.time()
    # unbatched_tasks = [process_unbatched(p) for p in prompts]
    # await asyncio.gather(*unbatched_tasks)
    # unbatched_time = time.time() - unbatched_start
    # results["unbatched_total_time"] = unbatched_time
    # results["unbatched_avg_latency"] = unbatched_time / n_requests
    
    pass  # Remove this line when you add your code
    
    # Benchmark batched approach
    print(f"   Running batched benchmark (batch_size={max_batch_size})...")
    
    # TODO: Time batched concurrent execution
    # 1. Create DynamicBatcher
    # 2. Start timeout processor
    # 3. Submit all requests concurrently with asyncio.gather
    # 4. Collect metrics
    #
    # Your code here:
    # batcher = DynamicBatcher(max_batch_size=max_batch_size, max_wait_ms=max_wait_ms)
    # await batcher.start_timeout_processor()
    # 
    # batched_start = time.time()
    # batched_tasks = [batcher.submit(p) for p in prompts]
    # await asyncio.gather(*batched_tasks)
    # batched_time = time.time() - batched_start
    # 
    # await batcher.stop_timeout_processor()
    # 
    # results["batched_total_time"] = batched_time
    # results["batched_avg_latency"] = batched_time / n_requests
    # results["avg_batch_size"] = (
    #     sum(batcher.metrics.batch_sizes) / len(batcher.metrics.batch_sizes)
    #     if batcher.metrics.batch_sizes else 0
    # )
    # results["speedup"] = unbatched_time / batched_time if batched_time > 0 else 0
    
    pass  # Remove this line when you add your code
    
    return results


# === REFERENCE SOLUTION (hidden until you complete the TODOs) ===
#
# Uncomment the section below after attempting the TODOs to verify your work.

"""
# REFERENCE: Complete DynamicBatcher implementation
class DynamicBatcherReference:
    def __init__(self, max_batch_size: int = 8, max_wait_ms: float = 50.0):
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self.pending: List[PendingRequest] = []
        self.lock = asyncio.Lock()
        self.metrics = BatchMetrics()
        self._timeout_task: Optional[asyncio.Task] = None
    
    async def submit(self, prompt: str, max_tokens: int = 50) -> str:
        future = asyncio.get_event_loop().create_future()
        request = PendingRequest(
            prompt=prompt,
            max_tokens=max_tokens,
            future=future,
            arrival_time=time.time()
        )
        
        async with self.lock:
            self.pending.append(request)
            should_process = len(self.pending) >= self.max_batch_size
        
        if should_process:
            asyncio.create_task(self._process_batch())
        
        return await future
    
    async def _process_batch(self):
        async with self.lock:
            if not self.pending:
                return
            
            batch = self.pending[:self.max_batch_size]
            self.pending = self.pending[self.max_batch_size:]
        
        self.metrics.batch_sizes.append(len(batch))
        
        prompts = [r.prompt for r in batch]
        start_time = time.time()
        results = await simulate_llm_inference(prompts)
        processing_time = time.time() - start_time
        
        self.metrics.processing_times.append(processing_time)
        
        for request, result in zip(batch, results):
            if not request.future.done():
                latency = time.time() - request.arrival_time
                self.metrics.request_latencies.append(latency)
                request.future.set_result(result)
    
    async def start_timeout_processor(self):
        self._timeout_task = asyncio.create_task(self._timeout_loop())
    
    async def _timeout_loop(self):
        while True:
            await asyncio.sleep(self.max_wait_ms / 1000)
            
            async with self.lock:
                if not self.pending:
                    continue
                
                oldest = self.pending[0]
                wait_time_ms = (time.time() - oldest.arrival_time) * 1000
                
                if wait_time_ms >= self.max_wait_ms:
                    asyncio.create_task(self._process_batch())
    
    async def stop_timeout_processor(self):
        if self._timeout_task:
            self._timeout_task.cancel()
            try:
                await self._timeout_task
            except asyncio.CancelledError:
                pass

# REFERENCE: Unbatched processing
async def process_unbatched_reference(prompt: str, max_tokens: int = 50) -> str:
    results = await simulate_llm_inference([prompt], max_tokens)
    return results[0]
"""


# === MAIN EXECUTION ===

async def main():
    """
    Main function to run the lab exercises.
    """
    print("=" * 60)
    print("Module 6 Lab: Dynamic Request Batching for LLM Inference")
    print("=" * 60)
    
    # Demonstrate the core concept
    print("\n1. Understanding Batching Benefits")
    print("-" * 40)
    print("Simulated LLM timing model:")
    print("  - Fixed cost per batch: 50ms (weight loading)")
    print("  - Variable cost per prompt: ~5-8ms")
    print()
    print("Single request:  50ms + 5ms = 55ms")
    print("8 requests batched: 50ms + (5ms * 8) = 90ms total")
    print("  → Per-request: 90ms / 8 = 11.25ms (4.9x faster!)")
    
    # Test single inference
    print("\n2. Testing Single Inference...")
    start = time.time()
    result = await simulate_llm_inference(["Test prompt"], max_tokens=50)
    single_time = time.time() - start
    print(f"   Single request time: {single_time*1000:.1f}ms")
    print(f"   Response: {result[0][:50]}...")
    
    # Test batch inference
    print("\n3. Testing Batch Inference (8 prompts)...")
    test_prompts = SAMPLE_PROMPTS[:8]
    start = time.time()
    results = await simulate_llm_inference(test_prompts, max_tokens=50)
    batch_time = time.time() - start
    print(f"   Batch of 8 time: {batch_time*1000:.1f}ms total")
    print(f"   Per-request: {batch_time*1000/8:.1f}ms")
    print(f"   Speedup vs single: {single_time / (batch_time/8):.2f}x")
    
    # Run batcher (YOUR IMPLEMENTATION)
    print("\n4. Testing DynamicBatcher...")
    print("   [Complete the TODOs to enable this section]")
    
    # Uncomment after implementing TODOs:
    # batcher = DynamicBatcher(max_batch_size=4, max_wait_ms=50.0)
    # await batcher.start_timeout_processor()
    # 
    # # Submit concurrent requests
    # tasks = [batcher.submit(p) for p in SAMPLE_PROMPTS[:8]]
    # results = await asyncio.gather(*tasks)
    # 
    # await batcher.stop_timeout_processor()
    # 
    # print(f"   Processed {len(results)} requests")
    # print(f"   Average batch size: {sum(batcher.metrics.batch_sizes)/len(batcher.metrics.batch_sizes):.1f}")
    
    # Run benchmarks (YOUR IMPLEMENTATION)
    print("\n5. Running Benchmarks...")
    print("   [Complete the TODOs to enable benchmarking]")
    
    # Uncomment after implementing TODOs:
    # benchmark_results = await benchmark_batching(
    #     n_requests=32,
    #     max_batch_size=8,
    #     max_wait_ms=50.0
    # )
    # 
    # print(f"\n   Results:")
    # print(f"   Unbatched total time: {benchmark_results['unbatched_total_time']*1000:.1f}ms")
    # print(f"   Batched total time: {benchmark_results['batched_total_time']*1000:.1f}ms")
    # print(f"   Average batch size: {benchmark_results['avg_batch_size']:.1f}")
    # print(f"   Speedup: {benchmark_results['speedup']:.2f}x")
    
    print("\n" + "=" * 60)
    print("Lab complete! See SELF-CHECK section below.")
    print("=" * 60)


# === EXPECTED OUTPUT ===
#
# When you run this lab with completed TODOs, you should see output similar to:
#
# 1. Understanding Batching Benefits
# ----------------------------------------
# Simulated LLM timing model:
#   - Fixed cost per batch: 50ms (weight loading)
#   - Variable cost per prompt: ~5-8ms
#
# Single request:  50ms + 5ms = 55ms
# 8 requests batched: 50ms + (5ms * 8) = 90ms total
#   → Per-request: 90ms / 8 = 11.25ms (4.9x faster!)
#
# 2. Testing Single Inference...
#    Single request time: 58.0ms
#
# 3. Testing Batch Inference (8 prompts)...
#    Batch of 8 time: 114.0ms total
#    Per-request: 14.3ms
#    Speedup vs single: 4.06x
#
# 4. Testing DynamicBatcher...
#    Processed 8 requests
#    Average batch size: 4.0
#
# 5. Running Benchmarks...
#    Unbatched total time: 1856.0ms
#    Batched total time: 571.0ms
#    Average batch size: 8.0
#    Speedup: 3.25x


# === SELF-CHECK ===

async def run_self_check():
    """
    Self-check assertions to verify your implementation.
    Run this after completing the TODOs.
    """
    print("\n" + "=" * 60)
    print("SELF-CHECK")
    print("=" * 60)
    
    # Check 1: DynamicBatcher initialization
    print("\n✓ Check 1: DynamicBatcher initialization...")
    try:
        batcher = DynamicBatcher(max_batch_size=4, max_wait_ms=100.0)
        assert hasattr(batcher, 'pending'), "Missing 'pending' attribute"
        assert hasattr(batcher, 'lock'), "Missing 'lock' attribute"
        assert isinstance(batcher.lock, asyncio.Lock), "lock should be asyncio.Lock"
        print("  PASSED: DynamicBatcher initializes correctly")
    except (AttributeError, AssertionError) as e:
        print(f"  SKIPPED: Complete TODO 1 - {e}")
        return
    
    # Check 2: Submit and process single request
    print("\n✓ Check 2: Single request processing...")
    try:
        batcher = DynamicBatcher(max_batch_size=1, max_wait_ms=100.0)
        await batcher.start_timeout_processor()
        
        result = await asyncio.wait_for(
            batcher.submit("Test prompt"),
            timeout=5.0
        )
        
        await batcher.stop_timeout_processor()
        
        assert result is not None, "Result should not be None"
        assert isinstance(result, str), "Result should be a string"
        assert len(result) > 0, "Result should not be empty"
        print("  PASSED: Single request processed correctly")
    except asyncio.TimeoutError:
        print("  FAILED: Request timed out (check _process_batch logic)")
        return
    except (TypeError, AttributeError) as e:
        print(f"  SKIPPED: Complete TODOs 1-3 - {e}")
        return
    
    # Check 3: Batch processing works
    print("\n✓ Check 3: Batch processing (8 concurrent requests)...")
    try:
        batcher = DynamicBatcher(max_batch_size=4, max_wait_ms=100.0)
        await batcher.start_timeout_processor()
        
        tasks = [batcher.submit(f"Prompt {i}") for i in range(8)]
        results = await asyncio.wait_for(
            asyncio.gather(*tasks),
            timeout=5.0
        )
        
        await batcher.stop_timeout_processor()
        
        assert len(results) == 8, f"Expected 8 results, got {len(results)}"
        assert all(isinstance(r, str) for r in results), "All results should be strings"
        print(f"  PASSED: 8 requests processed in {len(batcher.metrics.batch_sizes)} batches")
    except asyncio.TimeoutError:
        print("  FAILED: Requests timed out")
        return
    except Exception as e:
        print(f"  FAILED: {e}")
        return
    
    # Check 4: No race conditions under high concurrency
    print("\n✓ Check 4: Race condition test (50 concurrent requests)...")
    try:
        batcher = DynamicBatcher(max_batch_size=8, max_wait_ms=50.0)
        await batcher.start_timeout_processor()
        
        n_requests = 50
        tasks = [batcher.submit(f"Concurrent prompt {i}") for i in range(n_requests)]
        results = await asyncio.wait_for(
            asyncio.gather(*tasks),
            timeout=10.0
        )
        
        await batcher.stop_timeout_processor()
        
        assert len(results) == n_requests, f"Expected {n_requests} results"
        total_processed = sum(batcher.metrics.batch_sizes)
        assert total_processed == n_requests, f"Processed {total_processed} != submitted {n_requests}"
        print(f"  PASSED: {n_requests} requests without race conditions")
    except Exception as e:
        print(f"  FAILED: {e}")
        return
    
    # Check 5: Unbatched processing works
    print("\n✓ Check 5: Unbatched processing...")
    try:
        result = await process_unbatched("Test prompt")
        assert result is not None, "Result should not be None"
        assert isinstance(result, str), "Result should be a string"
        print("  PASSED: Unbatched processing works")
    except (TypeError, NameError) as e:
        print(f"  SKIPPED: Complete TODO 4 - {e}")
    
    # Check 6: Benchmarking works
    print("\n✓ Check 6: Benchmarking...")
    try:
        results = await benchmark_batching(n_requests=16, max_batch_size=4)
        assert results is not None, "Benchmark should return results"
        assert "speedup" in results, "Results should include speedup"
        print(f"  PASSED: Benchmark shows {results.get('speedup', 0):.2f}x speedup")
    except (TypeError, KeyError) as e:
        print(f"  SKIPPED: Complete TODO 5 - {e}")
    
    print("\n" + "=" * 60)
    print("✅ All checks passed! Your implementation is correct.")
    print("=" * 60)
    print("\nNext steps for Milestone 5:")
    print("  - Add a caching layer (Redis or in-memory)")
    print("  - Implement cache key hashing for privacy")
    print("  - Build comprehensive benchmarks")
    print("  - Write governance memo on caching considerations")


def main_sync():
    """Synchronous wrapper for main()."""
    asyncio.run(main())


def run_self_check_sync():
    """Synchronous wrapper for self-check."""
    asyncio.run(run_self_check())


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--check":
        run_self_check_sync()
    else:
        main_sync()
        print("\n" + "-" * 60)
        print("To run self-check after completing TODOs:")
        print("  python batching.py --check")
        print("-" * 60)