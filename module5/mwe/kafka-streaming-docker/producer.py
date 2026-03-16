#!/usr/bin/env python3
"""
Kafka Producer - Synthetic Transaction Event Generator.

Sends synthetic transaction events to a Kafka topic. Generates events
matching the schema from the synthetic-data-benchmark MWE.

Prerequisites:
    docker compose up -d   # Start Kafka broker + create topic

Usage:
    python producer.py                    # Send 100 events (default)
    python producer.py --n_events 500     # Send 500 events
    python producer.py --rate 50          # 50 events/sec
    python producer.py --burst            # Send all at once (backpressure demo)
"""

import argparse
import json
import time
import uuid
from collections import defaultdict

import numpy as np

from kafka import KafkaProducer


def create_producer(bootstrap_servers: str = "localhost:9092") -> KafkaProducer:
    """Create a Kafka producer with JSON serialization.

    Args:
        bootstrap_servers: Kafka broker address.

    Returns:
        Configured KafkaProducer instance.
    """
    return KafkaProducer(
        bootstrap_servers=[bootstrap_servers],
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        key_serializer=lambda k: k.encode("utf-8") if k else None,
        acks="all",  # Wait for all replicas (strongest durability)
    )


def generate_event(seed_state: np.random.RandomState) -> dict:
    """Generate a single synthetic transaction event.

    Args:
        seed_state: NumPy RandomState for reproducible generation.

    Returns:
        Dictionary with transaction event fields.
    """
    categories = ["food", "electronics", "clothing", "travel",
                   "entertainment", "utilities", "healthcare", "education"]
    weights = [0.25, 0.15, 0.15, 0.10, 0.10, 0.10, 0.08, 0.07]

    n_users = 50
    user_id = f"u{seed_state.randint(0, n_users):03d}"
    amount = round(float(seed_state.exponential(scale=50.0)), 2)
    category = seed_state.choice(categories, p=weights)

    return {
        "event_id": str(uuid.uuid4()),
        "user_id": user_id,
        "amount": amount,
        "category": category,
        "timestamp": time.time(),
    }


def main():
    parser = argparse.ArgumentParser(description="Kafka transaction event producer")
    parser.add_argument("--n_events", type=int, default=100,
                        help="Number of events to send (default: 100)")
    parser.add_argument("--rate", type=float, default=20.0,
                        help="Events per second (default: 20)")
    parser.add_argument("--burst", action="store_true",
                        help="Send all events as fast as possible (backpressure demo)")
    parser.add_argument("--bootstrap_servers", type=str, default="localhost:9092",
                        help="Kafka broker address (default: localhost:9092)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible events (default: 42)")
    args = parser.parse_args()

    print("=" * 60)
    print("KAFKA PRODUCER: Transaction Event Generator")
    print("=" * 60)

    # Connect to Kafka
    print("\n[1/3] Connecting to Kafka broker...")
    try:
        producer = create_producer(args.bootstrap_servers)
        print(f"      Connected to: {args.bootstrap_servers}")
    except Exception as e:
        print(f"      ❌ Connection failed: {e}")
        print("      Is Kafka running? Try: docker compose up -d")
        return

    # Generate and send events
    rng = np.random.RandomState(args.seed)
    partition_counts = defaultdict(int)
    topic = "transactions"
    delay = 0 if args.burst else (1.0 / args.rate)
    mode = "burst" if args.burst else f"{args.rate} events/sec"

    print(f"\n[2/3] Sending {args.n_events} events to '{topic}' topic ({mode})...")
    start_time = time.time()

    for i in range(args.n_events):
        event = generate_event(rng)

        # Use user_id as key → ensures same user always goes to same partition
        future = producer.send(topic, key=event["user_id"], value=event)

        try:
            record_metadata = future.get(timeout=10)
            partition_counts[record_metadata.partition] += 1
        except Exception as e:
            print(f"      ❌ Event {i + 1} failed: {e}")
            continue

        # Print first 3 and last event
        if i < 3 or i == args.n_events - 1:
            print(f"      → Event {i + 1}: user_id={event['user_id']}, "
                  f"amount=${event['amount']:.2f}, "
                  f"category={event['category']} "
                  f"→ partition {record_metadata.partition}")
        elif i == 3:
            print(f"      ... ({args.n_events - 4} more events) ...")

        if delay > 0:
            time.sleep(delay)

    producer.flush()
    elapsed = time.time() - start_time

    # Summary
    print(f"\n[3/3] All events sent. Flushing...")
    print(f"      {args.n_events} events produced in {elapsed:.2f}s")
    print(f"      Throughput: {args.n_events / elapsed:.1f} events/sec")
    print(f"      Partition distribution: {dict(sorted(partition_counts.items()))}")

    producer.close()
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
