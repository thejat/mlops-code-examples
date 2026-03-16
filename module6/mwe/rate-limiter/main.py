#!/usr/bin/env python3
"""
Rate Limiter MWE

Demonstrates a sliding-window rate limiter for LLM serving governance:
- Per-user request tracking with sliding window
- Allow/deny decisions based on requests-per-minute
- Old requests aging out as time advances (the defining sliding-window property)
- Simulation of legitimate vs abusive traffic patterns
- Accepted/rejected counts and per-user statistics

Usage:
    python main.py
"""

import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional


class RateLimiter:
    """Sliding-window rate limiter (per-user, requests-per-minute).

    Accepts an optional clock function for testable time simulation.
    """

    def __init__(
        self,
        requests_per_minute: int = 60,
        clock=None,
    ):
        self.rpm = requests_per_minute
        self.requests: Dict[str, List[float]] = defaultdict(list)
        self._clock = clock or time.time

    def check(self, user_id: str) -> bool:
        """Check if a request from user_id is allowed.

        Returns True if allowed, False if rate-limited.
        Uses a sliding window of the last 60 seconds.
        """
        now = self._clock()
        minute_ago = now - 60

        # Remove timestamps older than the window
        self.requests[user_id] = [
            t for t in self.requests[user_id] if t > minute_ago
        ]

        # Check against limit
        if len(self.requests[user_id]) >= self.rpm:
            return False

        # Record this request
        self.requests[user_id].append(now)
        return True

    def window_count(self, user_id: str) -> int:
        """Return current count of requests in the window for user_id."""
        now = self._clock()
        minute_ago = now - 60
        return sum(1 for t in self.requests.get(user_id, []) if t > minute_ago)


@dataclass
class UserStats:
    """Track per-user rate limiting statistics."""
    accepted: int = 0
    rejected: int = 0

    @property
    def total(self) -> int:
        return self.accepted + self.rejected

    @property
    def reject_rate(self) -> float:
        return self.rejected / self.total if self.total > 0 else 0.0


class SimulatedClock:
    """A fake clock for deterministic sliding-window demonstration.

    Allows advancing time without actually sleeping, so the demo
    runs instantly while showing how old requests age out.
    """

    def __init__(self, start: float = 1000.0):
        self._now = start

    def __call__(self) -> float:
        return self._now

    def advance(self, seconds: float):
        """Advance the simulated clock by the given number of seconds."""
        self._now += seconds


def simulate_traffic(
    limiter: RateLimiter,
    user_id: str,
    num_requests: int,
) -> UserStats:
    """Simulate a user sending requests in a burst."""
    stats = UserStats()

    for _ in range(num_requests):
        if limiter.check(user_id):
            stats.accepted += 1
        else:
            stats.rejected += 1

    return stats


def main():
    print("=" * 60)
    print("RATE LIMITER DEMO")
    print("=" * 60)

    # Use simulated clock so the demo runs instantly
    clock = SimulatedClock(start=1000.0)

    rpm_limit = 20
    limiter = RateLimiter(requests_per_minute=rpm_limit, clock=clock)

    print(f"\n  Rate limit: {rpm_limit} requests per minute per user")
    print(f"  Using simulated clock for instant demonstration\n")

    # --- Demo 1: Normal user ---
    print("--- Demo 1: Normal User (10 requests) ---\n")

    stats_normal = simulate_traffic(limiter, "user_normal", num_requests=10)
    print(f"  User: user_normal")
    print(f"  Sent: {stats_normal.total} requests")
    print(f"  Accepted: {stats_normal.accepted}")
    print(f"  Rejected: {stats_normal.rejected}")
    print(f"  Reject rate: {stats_normal.reject_rate:.0%}")
    print(f"  Result: ✅ All requests accepted (under limit)\n")

    # --- Demo 2: Power user (at the limit) ---
    print("--- Demo 2: Power User (20 requests, exactly at limit) ---\n")

    stats_power = simulate_traffic(limiter, "user_power", num_requests=20)
    print(f"  User: user_power")
    print(f"  Sent: {stats_power.total} requests")
    print(f"  Accepted: {stats_power.accepted}")
    print(f"  Rejected: {stats_power.rejected}")
    print(f"  Reject rate: {stats_power.reject_rate:.0%}")
    print(f"  Result: ✅ Just barely under/at the limit\n")

    # --- Demo 3: Abusive user (well over the limit) ---
    print("--- Demo 3: Abusive User (50 requests, over limit) ---\n")

    stats_abuse = simulate_traffic(limiter, "user_abusive", num_requests=50)
    print(f"  User: user_abusive")
    print(f"  Sent: {stats_abuse.total} requests")
    print(f"  Accepted: {stats_abuse.accepted}")
    print(f"  Rejected: {stats_abuse.rejected}")
    print(f"  Reject rate: {stats_abuse.reject_rate:.0%}")
    print(f"  Result: 🚫 {stats_abuse.rejected} requests blocked!\n")

    # --- Demo 4: User isolation ---
    print("--- Demo 4: User Isolation ---\n")

    clock2 = SimulatedClock(start=2000.0)
    limiter2 = RateLimiter(requests_per_minute=10, clock=clock2)

    stats_a = simulate_traffic(limiter2, "user_A", num_requests=10)
    stats_b = simulate_traffic(limiter2, "user_B", num_requests=5)
    stats_a_extra = simulate_traffic(limiter2, "user_A", num_requests=5)

    print(f"  User A: {stats_a.accepted} accepted, then {stats_a_extra.rejected} rejected (at limit)")
    print(f"  User B: {stats_b.accepted} accepted (separate counter, under limit)")
    print(f"  Result: ✅ User A's traffic does NOT affect User B\n")

    # --- Demo 5: Sliding window — old requests age out ---
    print("--- Demo 5: Sliding Window (old requests age out) ---\n")
    print("  This is the KEY property of a sliding window rate limiter:")
    print("  as time advances, old requests leave the window, freeing quota.\n")

    clock3 = SimulatedClock(start=3000.0)
    limiter3 = RateLimiter(requests_per_minute=5, clock=clock3)

    # Phase 1: Fill up the quota
    print(f"  t=0s: Send 5 requests (fills quota of 5/min)...")
    phase1 = simulate_traffic(limiter3, "user_slide", num_requests=5)
    count_after_fill = limiter3.window_count("user_slide")
    print(f"    Accepted: {phase1.accepted}, Window count: {count_after_fill}/5")

    # Phase 2: Try one more — should be rejected
    phase2 = simulate_traffic(limiter3, "user_slide", num_requests=1)
    print(f"\n  t=0s: Try 1 more request...")
    print(f"    Accepted: {phase2.accepted}, Rejected: {phase2.rejected}")
    print(f"    Status: 🚫 Quota exhausted")

    # Phase 3: Advance 30 seconds — requests still in window
    clock3.advance(30)
    count_at_30s = limiter3.window_count("user_slide")
    phase3 = simulate_traffic(limiter3, "user_slide", num_requests=1)
    print(f"\n  t=30s: Advance 30 seconds, try 1 request...")
    print(f"    Window count before: {count_at_30s}/5")
    print(f"    Accepted: {phase3.accepted}, Rejected: {phase3.rejected}")
    print(f"    Status: 🚫 Original 5 requests still in window (only 30s old)")

    # Phase 4: Advance to 61 seconds — all original requests age out
    clock3.advance(31)  # total: 61 seconds from original burst
    count_at_61s = limiter3.window_count("user_slide")
    print(f"\n  t=61s: Advance to 61 seconds total...")
    print(f"    Window count: {count_at_61s}/5 (original requests aged out!)")

    phase4 = simulate_traffic(limiter3, "user_slide", num_requests=3)
    count_after_new = limiter3.window_count("user_slide")
    print(f"    Send 3 new requests → Accepted: {phase4.accepted}")
    print(f"    Window count now: {count_after_new}/5")
    print(f"    Status: ✅ Quota refreshed — sliding window works!")

    # Phase 5: Can send 2 more (5 - 3 = 2 remaining)
    phase5 = simulate_traffic(limiter3, "user_slide", num_requests=4)
    print(f"\n  t=61s: Try 4 more requests (only 2 quota remaining)...")
    print(f"    Accepted: {phase5.accepted}, Rejected: {phase5.rejected}")
    print(f"    Status: 2 accepted (quota), 2 rejected (over limit)")

    print()

    # --- Summary ---
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\n  Rate limit: {rpm_limit} requests/minute")
    print()

    all_users = [
        ("user_normal", stats_normal),
        ("user_power", stats_power),
        ("user_abusive", stats_abuse),
    ]

    header = f"  {'User':<16s} {'Sent':>6s} {'Accepted':>9s} {'Rejected':>9s} {'Reject%':>8s}"
    print(header)
    print("  " + "-" * 50)
    for name, stats in all_users:
        print(f"  {name:<16s} {stats.total:>6d} {stats.accepted:>9d} "
              f"{stats.rejected:>9d} {stats.reject_rate:>7.0%}")

    print()
    print("  Governance takeaways:")
    print("  - Rate limiting prevents abuse and ensures fair access")
    print("  - Per-user limits isolate users from each other")
    print("  - Sliding window continuously ages out old requests (Demo 5)")
    print("  - In production, use Redis-backed distributed rate limiting")
    print("=" * 60)


if __name__ == "__main__":
    main()
