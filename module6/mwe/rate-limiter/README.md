# Rate Limiter MWE

Demonstrates a sliding-window rate limiter for LLM serving governance. Simulates normal, power, and abusive users to show how per-user rate limits protect system resources.

## Prerequisites

- Python 3.8+
- Any OS (Linux, macOS, Windows)

## Quick Start

```bash
# 1. Clone the repository
git clone <repo-url>
cd module6/mwe/rate-limiter

# 2. No dependencies to install (uses only standard library)

# 3. Run the demo
python main.py
```

## What It Demonstrates

| Demo | User Type | Requests | Result |
|------|-----------|----------|--------|
| 1. Normal user | 10 requests | Under limit | All accepted |
| 2. Power user | 20 requests | At limit | All/most accepted |
| 3. Abusive user | 50 requests | Over limit | Excess rejected |
| 4. User isolation | Two users | Independent | Separate counters |
| 5. Sliding window | Time-advancing | Quota refresh | Old requests age out |

## Key Concepts

| Concept | Description |
|---------|-------------|
| **Sliding window** | Tracks timestamps in the last 60 seconds; old entries are cleaned on each check |
| **Per-user isolation** | Each user has an independent request counter |
| **Simulated clock** | Demo 5 uses a `SimulatedClock` to advance time without `sleep()`, showing requests aging out instantly |
| **Deterministic** | Same traffic pattern → same accept/reject decisions |
| **Stateful** | Requires storing timestamps (in-memory here, Redis in production) |

## Governance Context

Rate limiting is a critical governance control for LLM serving:

- **Abuse prevention:** Stop individual users from monopolizing GPU resources
- **Cost control:** Prevent unexpected cost spikes from runaway clients
- **Fair access:** Ensure all users get reasonable service quality
- **Compliance:** May be required by service-level agreements (SLAs)

In production, combine rate limiting with:
- Authentication (who is making requests)
- Quotas (monthly/daily usage caps)
- Monitoring (alert on repeated rate-limit hits)

## Extension Challenges

1. **Token-based limiting:** Rate limit by tokens consumed, not just request count
2. **Tiered limits:** Different RPM limits for free vs paid users
3. **Distributed rate limiting:** Use Redis to share state across multiple servers
4. **Backoff headers:** Return `Retry-After` headers when requests are rejected

## Related Materials

- LLM Serving Governance (module6 slides, lines 1539–1611)
- Rate Limiting for Abuse Prevention
