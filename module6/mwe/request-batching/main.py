#!/usr/bin/env python3
"""
Dynamic Request Batching MWE

Demonstrates how to batch multiple inference requests together
to improve throughput by amortizing model loading costs.
"""

import asyncio
import time
from dataclasses import dataclass
from typing import List


@dataclass
class PendingRequest:
    """A request waiting to be processed."""
    prompt: str
    future: asyncio.Future
    arrival_time: float


class DynamicBatcher:
    """Batches requests by size threshold or timeout."""

    def __init__(self, max_batch_size: int = 4, max_wait_ms: float = 100.0):
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self.pending: List[PendingRequest] = []
        self.lock = asyncio.Lock()
        self.batch_count = 0

    async def submit(self, prompt: str) -> str:
        """Submit a request and wait for the batched result."""
        future = asyncio.get_event_loop().create_future()
        request = PendingRequest(prompt=prompt, future=future, arrival_time=time.time())

        async with self.lock:
            self.pending.append(request)
            should_process = len(self.pending) >= self.max_batch_size

        if should_process:
            asyncio.create_task(self._process_batch())

        return await future

    async def _process_batch(self):
        """Extract and process a batch of requests."""
        async with self.lock:
            if not self.pending:
                return
            batch = self.pending[: self.max_batch_size]
            self.pending = self.pending[self.max_batch_size :]
            self.batch_count += 1

        # Simulate batched inference (one "model load" for all requests)
        prompts = [r.prompt for r in batch]
        print(f"[Batch {self.batch_count}] Processing {len(batch)} requests together")
        await asyncio.sleep(0.05)  # Simulated inference time

        # Return results to waiting callers
        for i, request in enumerate(batch):
            result = f"Response to '{prompts[i]}' (batch {self.batch_count})"
            request.future.set_result(result)

    async def timeout_processor(self):
        """Background task that triggers batch processing on timeout."""
        while True:
            await asyncio.sleep(self.max_wait_ms / 1000)
            async with self.lock:
                if self.pending:
                    oldest = self.pending[0]
                    wait_time = (time.time() - oldest.arrival_time) * 1000
                    if wait_time >= self.max_wait_ms:
                        asyncio.create_task(self._process_batch())


async def simulate_client(batcher: DynamicBatcher, client_id: int, delay: float):
    """Simulate a client sending a request after a delay."""
    await asyncio.sleep(delay)
    prompt = f"Query from client {client_id}"
    result = await batcher.submit(prompt)
    print(f"  Client {client_id} received: {result}")


async def main():
    print("=== Dynamic Request Batching Demo ===\n")
    batcher = DynamicBatcher(max_batch_size=4, max_wait_ms=100)

    # Start timeout processor
    timeout_task = asyncio.create_task(batcher.timeout_processor())

    # Simulate 10 clients arriving at staggered times
    clients = [
        simulate_client(batcher, i, delay=i * 0.02)  # 20ms apart
        for i in range(10)
    ]

    await asyncio.gather(*clients)
    timeout_task.cancel()

    print(f"\n=== Summary ===")
    print(f"Total requests: 10")
    print(f"Batches processed: {batcher.batch_count}")
    print(f"Average batch size: {10 / batcher.batch_count:.1f}")


if __name__ == "__main__":
    asyncio.run(main())