#!/usr/bin/env python3
"""
Inference Caching MWE

Demonstrates LLM response caching with:
- Deterministic cache key generation using SHA-256
- TTL-based eviction (entries expire after N seconds)
- Cache hit/miss tracking and hit-rate calculation
- Hashed cache keys (avoids storing raw prompt text)
- Cache bypass for temperature > 0

Usage:
    python main.py
"""

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple


@dataclass
class CacheEntry:
    """A cached response with metadata."""
    response: str
    created_at: float
    access_count: int = 0


@dataclass
class CacheStats:
    """Track cache performance metrics."""
    hits: int = 0
    misses: int = 0
    bypasses: int = 0
    evictions: int = 0

    @property
    def total(self) -> int:
        return self.hits + self.misses + self.bypasses

    @property
    def hit_rate(self) -> float:
        lookups = self.hits + self.misses  # bypasses are not cache lookups
        return self.hits / lookups if lookups > 0 else 0.0


class InferenceCache:
    """In-memory LLM response cache with TTL eviction and hashed keys."""

    def __init__(self, ttl_seconds: float = 10.0):
        self.ttl_seconds = ttl_seconds
        self.store: Dict[str, CacheEntry] = {}
        self.stats = CacheStats()

    def make_cache_key(
        self,
        prompt: str,
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Create a deterministic, hashed cache key from request parameters.

        The prompt is normalized (stripped, lowercased) before hashing.
        The key is a SHA-256 hex digest — the raw prompt text is never stored.

        Note: unsalted SHA-256 is not resistant to dictionary attacks on
        predictable inputs. The goal here is to avoid accidental plaintext
        exposure in cache stores, not to provide cryptographic privacy.
        """
        key_data = {
            "prompt": prompt.strip().lower(),
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        content = json.dumps(key_data, sort_keys=True)
        return f"llm:{hashlib.sha256(content.encode()).hexdigest()}"

    def _is_expired(self, entry: CacheEntry) -> bool:
        return (time.time() - entry.created_at) > self.ttl_seconds

    def get(
        self,
        prompt: str,
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> Tuple[Optional[str], str]:
        """Look up a cached response.

        Returns (response_or_None, status_string).
        Bypasses cache entirely when temperature > 0.
        """
        # Bypass: non-deterministic requests should never use cache
        if temperature > 0:
            self.stats.bypasses += 1
            return None, "BYPASS (temperature > 0)"

        key = self.make_cache_key(prompt, model, temperature, max_tokens)

        if key in self.store:
            entry = self.store[key]
            if self._is_expired(entry):
                del self.store[key]
                self.stats.evictions += 1
                self.stats.misses += 1
                return None, "MISS (expired)"
            entry.access_count += 1
            self.stats.hits += 1
            return entry.response, "HIT"

        self.stats.misses += 1
        return None, "MISS"

    def put(
        self,
        prompt: str,
        model: str,
        temperature: float,
        max_tokens: int,
        response: str,
    ) -> None:
        """Store a response in the cache."""
        if temperature > 0:
            return  # never cache non-deterministic responses

        key = self.make_cache_key(prompt, model, temperature, max_tokens)
        self.store[key] = CacheEntry(response=response, created_at=time.time())


def simulate_llm_inference(prompt: str, latency_sec: float = 0.3) -> str:
    """Simulate an LLM generating a response (slow)."""
    time.sleep(latency_sec)
    return f"[Generated response for: '{prompt[:40]}...']"


def run_demo():
    """Run the caching demonstration."""
    print("=" * 60)
    print("INFERENCE CACHING DEMO")
    print("=" * 60)

    cache = InferenceCache(ttl_seconds=2.0)
    model = "llama-7b"
    max_tokens = 100
    simulated_latency = 0.3  # seconds

    # --- Demo 1: Cache hits and misses ---
    print("\n--- Demo 1: Cache Hits and Misses ---\n")

    queries = [
        ("What is machine learning?", 0.0),
        ("Explain neural networks.", 0.0),
        ("What is machine learning?", 0.0),  # repeat → HIT
        ("What is machine learning?", 0.0),  # repeat → HIT
        ("Explain neural networks.", 0.0),    # repeat → HIT
        ("What is deep learning?", 0.0),      # new → MISS
    ]

    for prompt, temp in queries:
        start = time.perf_counter()
        cached, status = cache.get(prompt, model, temp, max_tokens)

        if cached is not None:
            response = cached
        else:
            response = simulate_llm_inference(prompt, simulated_latency)
            cache.put(prompt, model, temp, max_tokens, response)

        elapsed_ms = (time.perf_counter() - start) * 1000
        print(f"  [{status:>20s}] {elapsed_ms:7.1f}ms | {prompt[:45]}")

    print(f"\n  Cache hit rate: {cache.stats.hit_rate:.0%}")
    print(f"  Entries in cache: {len(cache.store)}")

    # --- Demo 2: Cache bypass for temperature > 0 ---
    print("\n--- Demo 2: Cache Bypass (temperature > 0) ---\n")

    creative_queries = [
        ("Write a poem about AI.", 0.7),
        ("Write a poem about AI.", 0.7),  # same prompt, still bypassed
        ("Write a poem about AI.", 0.0),  # deterministic → cached
        ("Write a poem about AI.", 0.0),  # → HIT
    ]

    for prompt, temp in creative_queries:
        start = time.perf_counter()
        cached, status = cache.get(prompt, model, temp, max_tokens)

        if cached is not None:
            response = cached
        else:
            response = simulate_llm_inference(prompt, simulated_latency)
            cache.put(prompt, model, temp, max_tokens, response)

        elapsed_ms = (time.perf_counter() - start) * 1000
        print(f"  [{status:>20s}] {elapsed_ms:7.1f}ms | temp={temp} | {prompt[:35]}")

    # --- Demo 3: TTL-based expiration ---
    print("\n--- Demo 3: TTL Expiration ---\n")

    prompt_ttl = "What is reinforcement learning?"
    cache.put(prompt_ttl, model, 0.0, max_tokens,
              simulate_llm_inference(prompt_ttl, 0.0))

    cached, status = cache.get(prompt_ttl, model, 0.0, max_tokens)
    print(f"  Immediately after caching:  [{status}]")

    print("  Waiting 2.5 seconds for TTL expiry...")
    time.sleep(2.5)

    cached, status = cache.get(prompt_ttl, model, 0.0, max_tokens)
    print(f"  After TTL expiry:           [{status}]")

    # --- Demo 4: Hashed keys vs plaintext ---
    print("\n--- Demo 4: Hashed Keys (no plaintext stored) ---\n")

    sample_prompt = "What is the meaning of life?"
    key = cache.make_cache_key(sample_prompt, model, 0.0, max_tokens)

    print(f"  Prompt:     '{sample_prompt}'")
    print(f"  Cache key:  '{key}'")
    print(f"  Key length: {len(key)} chars")
    print()
    print("  ❌ Anti-pattern (plaintext key):")
    print(f"     key = '{sample_prompt}'  ← exposes user data!")
    print()
    print("  ✅ Hashed key (this demo):")
    print(f"     key = '{key[:50]}...'")
    print()
    print("  Note: Hashing avoids accidental plaintext exposure but is")
    print("  not a substitute for encryption or access control on")
    print("  sensitive data.")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    stats = cache.stats
    print(f"  Total requests:   {stats.total}")
    print(f"  Cache hits:       {stats.hits}")
    print(f"  Cache misses:     {stats.misses}")
    print(f"  Cache bypasses:   {stats.bypasses}")
    print(f"  TTL evictions:    {stats.evictions}")
    print(f"  Hit rate:         {stats.hit_rate:.0%}")
    print(f"  Entries in cache: {len(cache.store)}")
    print("=" * 60)


if __name__ == "__main__":
    run_demo()
