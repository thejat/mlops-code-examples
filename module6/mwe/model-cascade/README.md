# Model Cascade MWE

Demonstrates two orthogonal LLM serving patterns: complexity-based model routing (cascade) and load-based graceful degradation. Each is shown as an independent demo to keep the teaching objectives clear.

## Prerequisites

- Python 3.8+
- Any OS (Linux, macOS, Windows)

## Quick Start

```bash
# 1. Clone the repository
git clone <repo-url>
cd module6/mwe/model-cascade

# 2. No dependencies to install (uses only standard library)

# 3. Run the demo
python main.py
```

## What It Demonstrates

### Demo 1: Complexity-Based Routing (Model Cascade)

Routes prompts to the cheapest model capable of handling them:

| Complexity | Model | Cost/request | Typical Traffic |
|------------|-------|-------------|-----------------|
| Simple | 7B | $0.001 | ~80% |
| Medium | 13B | $0.005 | ~15% |
| Complex | 70B | $0.020 | ~5% |

### Demo 2: Load-Based Degradation (Graceful Degradation)

When system load exceeds thresholds, overrides model assignments:

| Load Level | Policy | Effect |
|------------|--------|--------|
| < 80% | Normal | Use assigned model |
| 80–95% | Degraded | Cap at 13B max |
| > 95% | Emergency | Force all to 7B |

### How They Compose

In production, these policies compose in sequence:
1. **Complexity routing** decides which model fits the task
2. **Load degradation** overrides when the system is stressed

```python
assigned_model = route_by_complexity(prompt)
actual_model = handle_overload(load_level, assigned_model)
```

## Key Implementation Detail

Model tier comparison uses an **explicit ordered mapping** rather than string comparison:

```python
MODEL_TIERS = {"7b": 0, "13b": 1, "70b": 2}
```

This avoids bugs from lexicographic string ordering (where `"7b" > "13b"` is `True`).

## Extension Challenges

1. **ML-based classifier:** Replace keyword heuristics with a lightweight complexity classifier
2. **A/B testing:** Compare cascade routing against always-70B on quality metrics
3. **Dynamic thresholds:** Adjust load thresholds based on time-of-day patterns
4. **Cost tracking:** Integrate with the inference-cost-calculator MWE for detailed cost analysis

## Related Materials

- Model Cascade Pattern (module6 slides, lines 2239–2253)
- Graceful Degradation Under Load (module6 slides, lines 2395–2456)
