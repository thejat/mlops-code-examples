#!/usr/bin/env python3
"""
Kafka Consumer - Windowed Aggregation with Deduplication.

Consumes transaction events from Kafka, maintains sliding window statistics
per user, and demonstrates at-least-once delivery with deduplication.

Prerequisites:
    docker compose up -d   # Start Kafka broker + create topic
    python producer.py     # Send events first

Usage:
    python consumer.py                          # Consume all available events
    python consumer.py --window_size 300        # 5-minute sliding window
    python consumer.py --timeout 10             # Wait 10s for new events
"""

import argparse
import json
import time
from collections import defaultdict

from kafka import KafkaConsumer, TopicPartition


# ---------------------------------------------------------------------------
# Sliding window implementation (from slides: lines 779-799)
# ---------------------------------------------------------------------------

class SlidingWindowAggregator:
    """Maintains per-user sliding window statistics.

    Implements the manual windowing pattern from the slides.
    Events older than window_size_sec are evicted on each update.
    """

    def __init__(self, window_size_sec: int = 300):
        self.window_size_sec = window_size_sec
        self.user_events: dict[str, list[tuple[float, float]]] = defaultdict(list)

    def add_event(self, user_id: str, timestamp: float, amount: float) -> None:
        """Add an event and evict expired entries."""
        self.user_events[user_id].append((timestamp, amount))
        self._cleanup(user_id, timestamp)

    def _cleanup(self, user_id: str, current_time: float) -> None:
        """Remove events outside the window (from slides: cleanup_old_events)."""
        cutoff = current_time - self.window_size_sec
        self.user_events[user_id] = [
            (ts, amt) for ts, amt in self.user_events[user_id]
            if ts >= cutoff
        ]

    def get_stats(self, user_id: str) -> dict:
        """Compute window statistics for a user (from slides: compute_window_stats)."""
        events = self.user_events[user_id]
        if not events:
            return {"count": 0, "sum": 0.0, "avg": 0.0}
        total = sum(amt for _, amt in events)
        return {
            "count": len(events),
            "sum": round(total, 2),
            "avg": round(total / len(events), 2),
        }

    def get_all_stats(self) -> dict[str, dict]:
        """Get stats for all users with events in the window."""
        return {uid: self.get_stats(uid) for uid in sorted(self.user_events)
                if self.user_events[uid]}


# ---------------------------------------------------------------------------
# Deduplication (from slides: lines 814-826)
#
# NOTE: This is an IN-MEMORY deduplication demo for effectively-once
# behavior WITHIN A SINGLE CONSUMER RUN. It does NOT:
#   - Survive consumer restarts (IDs lost on process exit)
#   - Bound memory usage (set grows without limit)
#   - Provide true exactly-once guarantees
#
# Production exactly-once requires external state stores (Redis, DB)
# or Kafka's transactional API.
# ---------------------------------------------------------------------------

class DeduplicationTracker:
    """Track processed event IDs to skip duplicates within one run."""

    def __init__(self):
        self.processed_ids: set[str] = set()
        self.duplicate_count: int = 0

    def is_duplicate(self, event_id: str) -> bool:
        """Check if event was already processed in this run."""
        if event_id in self.processed_ids:
            self.duplicate_count += 1
            return True
        self.processed_ids.add(event_id)
        return False


def main():
    parser = argparse.ArgumentParser(description="Kafka transaction event consumer")
    parser.add_argument("--window_size", type=int, default=300,
                        help="Sliding window size in seconds (default: 300)")
    parser.add_argument("--timeout", type=float, default=5.0,
                        help="Seconds to wait for new messages before exiting (default: 5)")
    parser.add_argument("--bootstrap_servers", type=str, default="localhost:9092",
                        help="Kafka broker address (default: localhost:9092)")
    parser.add_argument("--group_id", type=str, default="feature-pipeline",
                        help="Consumer group ID (default: feature-pipeline)")
    parser.add_argument("--stats_interval", type=int, default=25,
                        help="Print window stats every N events (default: 25)")
    args = parser.parse_args()

    print("=" * 60)
    print("KAFKA CONSUMER: Windowed Aggregation + Deduplication")
    print("=" * 60)

    # Connect to Kafka
    print(f"\n[1/4] Connecting to Kafka broker (group: {args.group_id})...")
    try:
        consumer = KafkaConsumer(
            "transactions",
            bootstrap_servers=[args.bootstrap_servers],
            group_id=args.group_id,
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
            # At-least-once: disable auto commit, commit manually after processing
            enable_auto_commit=False,
            auto_offset_reset="earliest",
            consumer_timeout_ms=int(args.timeout * 1000),
        )
        print(f"      Connected to: {args.bootstrap_servers}")
        print(f"      Topic: transactions")
        print(f"      Auto-commit: disabled (at-least-once delivery)")
    except Exception as e:
        print(f"      ❌ Connection failed: {e}")
        print("      Is Kafka running? Try: docker compose up -d")
        return

    # Initialize components
    window = SlidingWindowAggregator(window_size_sec=args.window_size)
    dedup = DeduplicationTracker()
    event_count = 0
    partition_counts = defaultdict(int)
    start_time = time.time()

    # Consume events
    print(f"\n[2/4] Consuming from 'transactions' topic...")
    print(f"      Window size: {args.window_size}s | Stats every {args.stats_interval} events")
    print()

    try:
        for message in consumer:
            event = message.value

            # Deduplication check
            event_id = event.get("event_id", "")
            if dedup.is_duplicate(event_id):
                continue

            # Process event: add to sliding window
            user_id = event["user_id"]
            amount = event["amount"]
            timestamp = event["timestamp"]

            window.add_event(user_id, timestamp, amount)
            partition_counts[message.partition] += 1
            event_count += 1

            # Manual commit after processing (at-least-once)
            consumer.commit()

            # Print window stats periodically
            if event_count % args.stats_interval == 0:
                elapsed = time.time() - start_time
                print(f"--- Window Stats after {event_count} events ({elapsed:.1f}s) ---")
                stats = window.get_all_stats()
                # Show top 5 users by transaction count
                top_users = sorted(stats.items(),
                                   key=lambda x: x[1]["count"], reverse=True)[:5]
                for uid, s in top_users:
                    print(f"  {uid}: count={s['count']}, "
                          f"sum=${s['sum']:.2f}, avg=${s['avg']:.2f}")
                print()

    except KeyboardInterrupt:
        print("\n      Interrupted by user.")

    elapsed = time.time() - start_time

    # Final stats
    print(f"\n[3/4] Deduplication stats (in-memory, this run only):")
    print(f"      → {event_count + dedup.duplicate_count} events received, "
          f"{dedup.duplicate_count} duplicates skipped")
    print(f"      → {event_count} unique events processed")
    print(f"      ⚠  Note: IDs tracked in memory only; not persisted across restarts")

    # Consumer lag
    print(f"\n[4/4] Consumer lag check...")
    try:
        partitions = consumer.assignment()
        total_lag = 0
        for tp in partitions:
            end_offsets = consumer.end_offsets([tp])
            committed = consumer.committed(tp)
            if committed is not None and tp in end_offsets:
                lag = end_offsets[tp] - committed
                total_lag += lag
                print(f"      Partition {tp.partition}: "
                      f"committed={committed}, end={end_offsets[tp]}, lag={lag}")
        print(f"      Total lag: {total_lag} messages behind")
    except Exception as e:
        print(f"      Could not check lag: {e}")

    # Summary
    print(f"\n      Processing complete in {elapsed:.2f}s")
    print(f"      Events/sec: {event_count / elapsed:.1f}" if elapsed > 0 else "")
    print(f"      Partition distribution: {dict(sorted(partition_counts.items()))}")

    # Final window snapshot
    print(f"\n{'=' * 60}")
    print("FINAL WINDOW STATS (all users)")
    print("=" * 60)
    final_stats = window.get_all_stats()
    for uid, s in sorted(final_stats.items(), key=lambda x: x[1]["sum"], reverse=True)[:10]:
        print(f"  {uid}: count={s['count']}, sum=${s['sum']:.2f}, avg=${s['avg']:.2f}")
    print(f"\n  Total users in window: {len(final_stats)}")

    consumer.close()
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
