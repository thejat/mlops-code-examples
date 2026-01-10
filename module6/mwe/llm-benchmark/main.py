"""
LLM Inference Benchmark Runner

A minimal working example for benchmarking LLM inference APIs with
batching and caching. Measures TTFT, TPOT, throughput, and cache hit rate.

Usage:
    python main.py --endpoint http://localhost:8000/generate
    python main.py --endpoint http://localhost:8000/generate --concurrency 10
    python main.py --help
"""

import argparse
import asyncio
import time
import statistics
import json
from dataclasses import dataclass, asdict
from typing import List, Optional
import aiohttp


@dataclass
class RequestMetrics:
    """Metrics for a single request."""
    prompt: str
    prompt_tokens: int
    output_tokens: int
    ttft_ms: float  # Time to first token
    total_time_ms: float
    cache_hit: bool
    error: Optional[str] = None


@dataclass 
class BenchmarkResult:
    """Aggregated benchmark results."""
    total_requests: int
    successful_requests: int
    failed_requests: int
    ttft_p50_ms: float
    ttft_p95_ms: float
    ttft_p99_ms: float
    tpot_avg_ms: float
    throughput_tokens_per_sec: float
    cache_hit_rate: float
    avg_latency_ms: float
    total_time_sec: float


def estimate_tokens(text: str) -> int:
    """Rough token count estimate (4 chars per token)."""
    return max(1, len(text) // 4)


async def make_request(
    session: aiohttp.ClientSession,
    endpoint: str,
    prompt: str,
    max_tokens: int = 100,
    temperature: float = 0.0
) -> RequestMetrics:
    """Make a single inference request and measure timing."""
    start = time.perf_counter()
    ttft = None
    output_text = ""
    cache_hit = False
    error = None
    
    payload = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    
    try:
        async with session.post(endpoint, json=payload) as response:
            if response.status != 200:
                error = f"HTTP {response.status}"
            else:
                # Time to first byte as proxy for TTFT
                ttft = time.perf_counter()
                data = await response.json()
                output_text = data.get("text", "")
                cache_hit = data.get("cached", False)
    except aiohttp.ClientError as e:
        error = str(e)
    except Exception as e:
        error = f"Unexpected error: {e}"
    
    end = time.perf_counter()
    
    return RequestMetrics(
        prompt=prompt[:50],  # Truncate for display
        prompt_tokens=estimate_tokens(prompt),
        output_tokens=estimate_tokens(output_text),
        ttft_ms=((ttft - start) * 1000) if ttft else 0,
        total_time_ms=(end - start) * 1000,
        cache_hit=cache_hit,
        error=error
    )


async def run_concurrent_benchmark(
    endpoint: str,
    prompts: List[str],
    concurrency: int,
    max_tokens: int
) -> List[RequestMetrics]:
    """Run benchmark with concurrent requests."""
    semaphore = asyncio.Semaphore(concurrency)
    results: List[RequestMetrics] = []
    
    async def bounded_request(session: aiohttp.ClientSession, prompt: str):
        async with semaphore:
            return await make_request(session, endpoint, prompt, max_tokens)
    
    connector = aiohttp.TCPConnector(limit=concurrency * 2)
    timeout = aiohttp.ClientTimeout(total=60)
    
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        tasks = [bounded_request(session, p) for p in prompts]
        results = await asyncio.gather(*tasks)
    
    return results


def calculate_percentile(data: List[float], percentile: int) -> float:
    """Calculate percentile from sorted data."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    idx = int(len(sorted_data) * percentile / 100)
    idx = min(idx, len(sorted_data) - 1)
    return sorted_data[idx]


def aggregate_results(metrics: List[RequestMetrics], total_time: float) -> BenchmarkResult:
    """Aggregate individual request metrics into summary."""
    successful = [m for m in metrics if m.error is None]
    failed = [m for m in metrics if m.error is not None]
    
    if not successful:
        return BenchmarkResult(
            total_requests=len(metrics),
            successful_requests=0,
            failed_requests=len(failed),
            ttft_p50_ms=0,
            ttft_p95_ms=0,
            ttft_p99_ms=0,
            tpot_avg_ms=0,
            throughput_tokens_per_sec=0,
            cache_hit_rate=0,
            avg_latency_ms=0,
            total_time_sec=total_time
        )
    
    ttfts = [m.ttft_ms for m in successful if m.ttft_ms > 0]
    latencies = [m.total_time_ms for m in successful]
    
    # Calculate TPOT (Time Per Output Token)
    tpots = []
    for m in successful:
        if m.output_tokens > 1 and m.ttft_ms > 0:
            decode_time = m.total_time_ms - m.ttft_ms
            tpots.append(decode_time / (m.output_tokens - 1))
    
    total_output_tokens = sum(m.output_tokens for m in successful)
    cache_hits = sum(1 for m in successful if m.cache_hit)
    
    return BenchmarkResult(
        total_requests=len(metrics),
        successful_requests=len(successful),
        failed_requests=len(failed),
        ttft_p50_ms=calculate_percentile(ttfts, 50) if ttfts else 0,
        ttft_p95_ms=calculate_percentile(ttfts, 95) if ttfts else 0,
        ttft_p99_ms=calculate_percentile(ttfts, 99) if ttfts else 0,
        tpot_avg_ms=statistics.mean(tpots) if tpots else 0,
        throughput_tokens_per_sec=total_output_tokens / total_time if total_time > 0 else 0,
        cache_hit_rate=cache_hits / len(successful) if successful else 0,
        avg_latency_ms=statistics.mean(latencies) if latencies else 0,
        total_time_sec=total_time
    )


def get_test_prompts(num_prompts: int, repeat_ratio: float = 0.3) -> List[str]:
    """Generate test prompts with some repetition for cache testing."""
    base_prompts = [
        "Explain machine learning in simple terms.",
        "What is the difference between AI and ML?",
        "How do neural networks work?",
        "What is gradient descent?",
        "Explain the transformer architecture.",
        "What is attention in deep learning?",
        "How does backpropagation work?",
        "What is overfitting and how to prevent it?",
        "Explain the bias-variance tradeoff.",
        "What are activation functions?",
    ]
    
    prompts = []
    unique_count = int(num_prompts * (1 - repeat_ratio))
    repeat_count = num_prompts - unique_count
    
    # Add unique prompts (cycling through base prompts)
    for i in range(unique_count):
        idx = i % len(base_prompts)
        # Add variation to make unique
        prompts.append(f"{base_prompts[idx]} (variation {i})")
    
    # Add repeated prompts (for cache testing)
    for i in range(repeat_count):
        idx = i % min(3, len(base_prompts))  # Repeat first 3 prompts
        prompts.append(base_prompts[idx])
    
    return prompts


def print_results(result: BenchmarkResult) -> None:
    """Print formatted benchmark results."""
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    
    print(f"\n📊 Request Summary:")
    print(f"   Total Requests:      {result.total_requests}")
    print(f"   Successful:          {result.successful_requests}")
    print(f"   Failed:              {result.failed_requests}")
    
    print(f"\n⏱️  Latency Metrics:")
    print(f"   TTFT P50:            {result.ttft_p50_ms:.2f} ms")
    print(f"   TTFT P95:            {result.ttft_p95_ms:.2f} ms")
    print(f"   TTFT P99:            {result.ttft_p99_ms:.2f} ms")
    print(f"   TPOT (avg):          {result.tpot_avg_ms:.2f} ms")
    print(f"   Avg Latency:         {result.avg_latency_ms:.2f} ms")
    
    print(f"\n🚀 Throughput:")
    print(f"   Tokens/sec:          {result.throughput_tokens_per_sec:.2f}")
    print(f"   Total Time:          {result.total_time_sec:.2f} sec")
    
    print(f"\n💾 Cache Performance:")
    print(f"   Hit Rate:            {result.cache_hit_rate * 100:.1f}%")
    
    print("\n" + "=" * 60)


async def main():
    parser = argparse.ArgumentParser(
        description="Benchmark LLM inference API with batching and caching",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic benchmark with 50 requests
  python main.py --endpoint http://localhost:8000/generate

  # High concurrency test
  python main.py --endpoint http://localhost:8000/generate --concurrency 20 --requests 200

  # Cache testing (high repeat ratio)
  python main.py --endpoint http://localhost:8000/generate --repeat-ratio 0.5
        """
    )
    
    parser.add_argument(
        "--endpoint",
        type=str,
        default="http://localhost:8000/generate",
        help="LLM inference endpoint URL (default: http://localhost:8000/generate)"
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=50,
        help="Number of requests to send (default: 50)"
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Maximum concurrent requests (default: 5)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Maximum tokens to generate per request (default: 100)"
    )
    parser.add_argument(
        "--repeat-ratio",
        type=float,
        default=0.3,
        help="Ratio of repeated prompts for cache testing (default: 0.3)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for JSON results (optional)"
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Number of warmup requests to discard (default: 5)"
    )
    
    args = parser.parse_args()
    
    print(f"🔧 Benchmark Configuration:")
    print(f"   Endpoint:      {args.endpoint}")
    print(f"   Requests:      {args.requests}")
    print(f"   Concurrency:   {args.concurrency}")
    print(f"   Max Tokens:    {args.max_tokens}")
    print(f"   Repeat Ratio:  {args.repeat_ratio}")
    
    # Generate test prompts
    total_prompts = args.requests + args.warmup
    prompts = get_test_prompts(total_prompts, args.repeat_ratio)
    
    # Run warmup
    if args.warmup > 0:
        print(f"\n🔥 Running {args.warmup} warmup requests...")
        warmup_prompts = prompts[:args.warmup]
        await run_concurrent_benchmark(
            args.endpoint,
            warmup_prompts,
            min(args.concurrency, args.warmup),
            args.max_tokens
        )
        prompts = prompts[args.warmup:]
    
    # Run main benchmark
    print(f"\n🏃 Running {args.requests} benchmark requests...")
    start_time = time.perf_counter()
    
    metrics = await run_concurrent_benchmark(
        args.endpoint,
        prompts,
        args.concurrency,
        args.max_tokens
    )
    
    total_time = time.perf_counter() - start_time
    
    # Aggregate and display results
    result = aggregate_results(metrics, total_time)
    print_results(result)
    
    # Save to file if requested
    if args.output:
        with open(args.output, "w") as f:
            json.dump(asdict(result), f, indent=2)
        print(f"\n📁 Results saved to: {args.output}")
    
    return result


if __name__ == "__main__":
    asyncio.run(main())