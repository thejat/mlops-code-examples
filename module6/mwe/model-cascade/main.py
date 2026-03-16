#!/usr/bin/env python3
"""
Model Cascade MWE

Demonstrates two orthogonal LLM serving patterns:

Demo 1: Complexity-Based Routing (Model Cascade)
  - Route requests to 7B/13B/70B models based on prompt complexity
  - Track cost savings vs always using the largest model

Demo 2: Load-Based Degradation (Graceful Degradation)
  - Override model assignments when system load exceeds thresholds
  - Fall back to smaller models to maintain SLA

In production, these compose: complexity routing first, then
load-based override.

Usage:
    python main.py
"""

import time
from dataclasses import dataclass
from typing import List, Tuple


# --- Model tier definitions ---

# Explicit ordered mapping for model tiers (avoids lexicographic string comparison)
MODEL_TIERS = {"7b": 0, "13b": 1, "70b": 2}  # smaller number = cheaper/faster
TIER_TO_MODEL = {v: k for k, v in MODEL_TIERS.items()}

# Simulated model characteristics
MODEL_CONFIGS = {
    "7b": {
        "latency_ms": 50,
        "cost_per_request": 0.001,
        "quality": "basic",
    },
    "13b": {
        "latency_ms": 100,
        "cost_per_request": 0.005,
        "quality": "good",
    },
    "70b": {
        "latency_ms": 300,
        "cost_per_request": 0.02,
        "quality": "excellent",
    },
}


def simulate_inference(prompt: str, model: str) -> str:
    """Simulate model inference (returns a placeholder response)."""
    config = MODEL_CONFIGS[model]
    # In a real system, this would call the actual model
    return f"[{model} response to '{prompt[:30]}...']"


# ============================================================
# Demo 1: Complexity-Based Routing
# ============================================================

def classify_complexity(prompt: str) -> str:
    """Classify prompt complexity based on length and keywords.

    This is a simplified heuristic. In production, you might use
    a lightweight classifier or rule engine.
    """
    complex_keywords = [
        "analyze", "compare", "explain in detail", "derive",
        "prove", "step by step", "multi-step", "reasoning",
        "code review", "debug", "optimize",
    ]
    prompt_lower = prompt.lower()

    # Check for complex keywords
    has_complex_keyword = any(kw in prompt_lower for kw in complex_keywords)

    # Classify based on length + keywords
    if has_complex_keyword or len(prompt) > 200:
        return "complex"
    elif len(prompt) > 80:
        return "medium"
    else:
        return "simple"


def route_by_complexity(prompt: str) -> str:
    """Route request to appropriate model tier based on complexity."""
    complexity = classify_complexity(prompt)

    if complexity == "simple":
        return "7b"
    elif complexity == "medium":
        return "13b"
    else:
        return "70b"


def demo_complexity_routing():
    """Demo 1: Route requests by complexity, compare cost vs always-70B."""
    print("--- Demo 1: Complexity-Based Routing (Model Cascade) ---\n")

    sample_prompts = [
        "What is AI?",
        "Define machine learning.",
        "How does Python work?",
        "Explain the difference between supervised and unsupervised learning, "
        "including practical examples of each approach.",
        "Compare and contrast transformer architectures with recurrent neural "
        "networks. Analyze the trade-offs in terms of training efficiency, "
        "inference speed, and model quality for different NLP tasks.",
        "What is gradient descent?",
        "Step by step, derive the backpropagation algorithm for a two-layer "
        "neural network with ReLU activations.",
        "Summarize this article.",
        "Hello!",
        "Write a Python function to sort a list.",
    ]

    cascade_cost = 0.0
    always_70b_cost = 0.0

    header = f"  {'Prompt':<50s} {'Complexity':<10s} {'Model':<6s} {'Cost':>7s}"
    print(header)
    print("  " + "-" * 75)

    for prompt in sample_prompts:
        model = route_by_complexity(prompt)
        complexity = classify_complexity(prompt)
        cost = MODEL_CONFIGS[model]["cost_per_request"]
        cascade_cost += cost
        always_70b_cost += MODEL_CONFIGS["70b"]["cost_per_request"]

        print(f"  {prompt[:50]:<50s} {complexity:<10s} {model:<6s} ${cost:.3f}")

    print()
    print(f"  Cascade total cost:    ${cascade_cost:.3f}")
    print(f"  Always-70B total cost: ${always_70b_cost:.3f}")
    savings = (1 - cascade_cost / always_70b_cost) * 100
    print(f"  Savings:               {savings:.0f}%")
    print()
    print(f"  Key insight: Most traffic is simple queries that a 7B model")
    print(f"  handles well. Routing saves cost without sacrificing quality")
    print(f"  on complex requests.\n")

    return cascade_cost, always_70b_cost


# ============================================================
# Demo 2: Load-Based Degradation
# ============================================================

def cap_model(assigned_model: str, max_tier_name: str) -> str:
    """Downgrade to at most max_tier_name, preserving smaller models as-is.

    Uses explicit numeric tier ranks to avoid lexicographic string comparison.
    """
    assigned_rank = MODEL_TIERS[assigned_model]
    max_rank = MODEL_TIERS[max_tier_name]
    return TIER_TO_MODEL[min(assigned_rank, max_rank)]


def handle_overload(load_level: float, assigned_model: str) -> Tuple[str, str]:
    """Determine actual model to use under given load level.

    Returns (actual_model, policy_applied).
    """
    if load_level < 0.8:
        return assigned_model, "normal"
    elif load_level < 0.95:
        capped = cap_model(assigned_model, "13b")
        policy = "degraded (cap at 13b)" if capped != assigned_model else "normal"
        return capped, policy
    else:
        return "7b", "emergency (all → 7b)"


def demo_load_degradation():
    """Demo 2: Show how load levels affect model routing."""
    print("--- Demo 2: Load-Based Degradation (Graceful Degradation) ---\n")

    # Fixed set of requests with pre-assigned models (from complexity routing)
    requests = [
        ("Simple query", "7b"),
        ("Medium query", "13b"),
        ("Complex query", "70b"),
        ("Another complex", "70b"),
        ("Simple again", "7b"),
    ]

    load_levels = [
        (0.5, "Normal (50%)"),
        (0.9, "High (90%)"),
        (0.98, "Critical (98%)"),
    ]

    for load, load_desc in load_levels:
        print(f"  Load level: {load_desc}")
        print(f"  {'Request':<20s} {'Assigned':<10s} {'Actual':<10s} {'Policy':<25s}")
        print(f"  " + "-" * 65)

        total_cost = 0.0
        for prompt, assigned in requests:
            actual, policy = handle_overload(load, assigned)
            cost = MODEL_CONFIGS[actual]["cost_per_request"]
            total_cost += cost
            print(f"  {prompt:<20s} {assigned:<10s} {actual:<10s} {policy:<25s}")

        print(f"  {'':>42s} Total cost: ${total_cost:.3f}")
        print()

    print(f"  Key insight: Under heavy load, downgrading models maintains")
    print(f"  availability at the cost of response quality. This is")
    print(f"  preferable to rejecting requests or timing out.\n")


def main():
    print("=" * 60)
    print("MODEL CASCADE & GRACEFUL DEGRADATION DEMO")
    print("=" * 60)
    print()
    print("  This demo shows two orthogonal routing policies:")
    print("  1. Complexity-based routing (which model to use)")
    print("  2. Load-based degradation (when to downgrade)")
    print("  In production, they compose: route by complexity first,")
    print("  then apply load-based overrides if needed.")
    print()

    cascade_cost, always_70b_cost = demo_complexity_routing()
    demo_load_degradation()

    # --- Summary ---
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print()
    print("  Model tier characteristics:")
    header = f"  {'Model':<8s} {'Latency':>10s} {'Cost/req':>10s} {'Quality':<12s}"
    print(header)
    print("  " + "-" * 42)
    for model, config in MODEL_CONFIGS.items():
        print(f"  {model:<8s} {config['latency_ms']:>8d}ms "
              f"${config['cost_per_request']:>8.3f} {config['quality']:<12s}")

    print()
    print("  Policies are orthogonal:")
    print("  - Complexity routing: decides WHICH model fits the task")
    print("  - Load degradation:   overrides WHEN system is stressed")
    print("  - Combined: route_by_complexity() → handle_overload()")
    print("=" * 60)


if __name__ == "__main__":
    main()
