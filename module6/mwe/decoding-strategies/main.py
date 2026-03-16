#!/usr/bin/env python3
"""
Decoding Strategies MWE

Demonstrates different token sampling strategies for LLM inference:
- Greedy decoding (deterministic, fastest)
- Temperature sampling (adjustable creativity)
- Top-k sampling (diverse, limited candidates)
- Top-p / nucleus sampling (balanced)

Uses a simulated vocabulary with pre-defined logit distributions
to show how each strategy affects token selection.

Usage:
    python main.py
"""

import numpy as np
from typing import List, Tuple


# Simulated vocabulary
VOCAB = [
    "the", "a", "is", "was", "are",          # common words
    "machine", "learning", "model", "data",    # ML terms
    "beautiful", "elegant", "fantastic",        # creative words
    "quantum", "entropy", "derivative",         # technical words
    "!", ".", ",",                               # punctuation
]


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Compute softmax with temperature scaling.

    Temperature controls distribution sharpness:
    - temp → 0: approaches argmax (deterministic)
    - temp = 1: standard softmax
    - temp > 1: flatter distribution (more random)
    """
    if temperature <= 0:
        # Approach argmax behavior
        result = np.zeros_like(logits, dtype=float)
        result[np.argmax(logits)] = 1.0
        return result

    scaled = logits / temperature
    # Subtract max for numerical stability
    shifted = scaled - np.max(scaled)
    exp_vals = np.exp(shifted)
    return exp_vals / np.sum(exp_vals)


def greedy_decode(logits: np.ndarray) -> Tuple[int, str]:
    """Greedy decoding: always select the highest probability token.

    - Deterministic (same input → same output)
    - Fastest (no sampling overhead)
    - Best for factual Q&A where one answer is correct
    """
    idx = int(np.argmax(logits))
    return idx, "greedy"


def temperature_sample(
    logits: np.ndarray,
    temperature: float = 0.7,
    rng: np.random.Generator = None,
) -> Tuple[int, str]:
    """Temperature sampling: scale logits before sampling.

    - temperature < 1: sharper distribution, more focused
    - temperature = 1: standard sampling
    - temperature > 1: flatter distribution, more creative
    """
    if rng is None:
        rng = np.random.default_rng()

    probs = softmax(logits, temperature)
    idx = int(rng.choice(len(probs), p=probs))
    return idx, f"temp={temperature}"


def top_k_sample(
    logits: np.ndarray,
    k: int = 5,
    temperature: float = 1.0,
    rng: np.random.Generator = None,
) -> Tuple[int, str]:
    """Top-k sampling: only consider the k most likely tokens.

    - Limits candidates to top k tokens
    - Prevents sampling from very unlikely tokens
    - Good for brainstorming (controlled diversity)
    """
    if rng is None:
        rng = np.random.default_rng()

    # Find top-k indices
    top_k_indices = np.argsort(logits)[-k:]

    # Zero out non-top-k logits
    filtered_logits = np.full_like(logits, -np.inf)
    filtered_logits[top_k_indices] = logits[top_k_indices]

    probs = softmax(filtered_logits, temperature)
    idx = int(rng.choice(len(probs), p=probs))
    return idx, f"top_k={k}"


def top_p_sample(
    logits: np.ndarray,
    p: float = 0.9,
    temperature: float = 1.0,
    rng: np.random.Generator = None,
) -> Tuple[int, str]:
    """Top-p (nucleus) sampling: sample from smallest set with cumprob >= p.

    - Dynamically adjusts candidate set size
    - p=0.9: exclude bottom 10% probability mass
    - Balanced approach: diverse but not random
    """
    if rng is None:
        rng = np.random.default_rng()

    probs = softmax(logits, temperature)

    # Sort by probability descending
    sorted_indices = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_indices]

    # Find cutoff where cumulative probability exceeds p
    cumsum = np.cumsum(sorted_probs)
    cutoff_idx = int(np.searchsorted(cumsum, p)) + 1
    cutoff_idx = min(cutoff_idx, len(probs))

    # Keep only tokens in the nucleus
    nucleus_indices = sorted_indices[:cutoff_idx]
    filtered_probs = np.zeros_like(probs)
    filtered_probs[nucleus_indices] = probs[nucleus_indices]
    filtered_probs /= filtered_probs.sum()  # re-normalize

    idx = int(rng.choice(len(filtered_probs), p=filtered_probs))
    return idx, f"top_p={p}"


def show_probability_distribution(logits: np.ndarray, title: str):
    """Display top tokens and their probabilities."""
    probs = softmax(logits)
    sorted_indices = np.argsort(probs)[::-1]

    print(f"\n  {title}")
    print(f"  {'Token':<15s} {'Probability':>12s} {'Bar'}")
    print(f"  " + "-" * 45)

    for i in range(min(8, len(sorted_indices))):
        idx = sorted_indices[i]
        token = VOCAB[idx]
        prob = probs[idx]
        bar = "█" * int(prob * 40)
        print(f"  {token:<15s} {prob:>11.2%}  {bar}")


def run_sampling_comparison(logits: np.ndarray, num_trials: int = 20):
    """Run all strategies multiple times and compare output diversity."""
    print(f"\n  Sampling {num_trials} tokens with each strategy:\n")

    rng = np.random.default_rng(42)  # fixed seed for reproducibility

    strategies = [
        ("Greedy", lambda: greedy_decode(logits)),
        ("Temp=0.3", lambda: temperature_sample(logits, 0.3, rng)),
        ("Temp=0.7", lambda: temperature_sample(logits, 0.7, rng)),
        ("Temp=1.5", lambda: temperature_sample(logits, 1.5, rng)),
        ("Top-k=3", lambda: top_k_sample(logits, k=3, rng=rng)),
        ("Top-k=8", lambda: top_k_sample(logits, k=8, rng=rng)),
        ("Top-p=0.5", lambda: top_p_sample(logits, p=0.5, rng=rng)),
        ("Top-p=0.9", lambda: top_p_sample(logits, p=0.9, rng=rng)),
    ]

    header = f"  {'Strategy':<12s} {'Unique':>7s} {'Most Common':>15s} {'Sample Tokens'}"
    print(header)
    print("  " + "-" * 70)

    for name, strategy_fn in strategies:
        tokens = []
        for _ in range(num_trials):
            idx, _ = strategy_fn()
            tokens.append(VOCAB[idx])

        unique = len(set(tokens))
        from collections import Counter
        most_common = Counter(tokens).most_common(1)[0]
        sample = " ".join(tokens[:8])

        print(f"  {name:<12s} {unique:>5d}/{ num_trials} "
              f"{most_common[0]:>12s}({most_common[1]:>2d}) "
              f"{sample}")


def main():
    print("=" * 60)
    print("DECODING STRATEGIES DEMO")
    print("=" * 60)

    # Create a realistic logit distribution:
    # "the" and "is" have high probability (common words)
    # ML terms have medium probability
    # creative/technical words have low probability
    logits = np.array([
        5.0,   # the       ← high prob
        3.0,   # a
        4.5,   # is        ← high prob
        2.0,   # was
        1.5,   # are
        3.5,   # machine
        3.2,   # learning
        2.8,   # model
        2.5,   # data
        0.5,   # beautiful ← low prob
        0.3,   # elegant
        0.1,   # fantastic
        -0.5,  # quantum   ← very low prob
        -1.0,  # entropy
        -0.8,  # derivative
        1.0,   # !
        2.0,   # .
        1.5,   # ,
    ])

    # --- Show base distribution ---
    show_probability_distribution(logits, "Base Logit Distribution (temperature=1.0)")

    # --- Show temperature effect ---
    print("\n--- Temperature Effect on Distribution ---")

    for temp in [0.3, 1.0, 2.0]:
        probs = softmax(logits, temp)
        top_idx = np.argmax(probs)
        top_prob = probs[top_idx]
        entropy = -np.sum(probs * np.log(probs + 1e-10))

        print(f"\n  Temperature = {temp}")
        print(f"    Top token: '{VOCAB[top_idx]}' at {top_prob:.1%}")
        print(f"    Entropy: {entropy:.2f} (higher = more random)")

        # Show top 5
        sorted_idx = np.argsort(probs)[::-1]
        tokens_str = ", ".join(f"{VOCAB[i]}({probs[i]:.1%})" for i in sorted_idx[:5])
        print(f"    Top 5: {tokens_str}")

    # --- Run sampling comparison ---
    print("\n--- Sampling Comparison ---")
    run_sampling_comparison(logits, num_trials=20)

    # --- Summary table ---
    print("\n" + "=" * 60)
    print("STRATEGY COMPARISON")
    print("=" * 60)
    print()
    print(f"  {'Strategy':<14s} {'Speed':<10s} {'Deterministic':<15s} {'Best For'}")
    print(f"  " + "-" * 55)
    print(f"  {'Greedy':<14s} {'Fastest':<10s} {'Yes':<15s} {'Factual Q&A'}")
    print(f"  {'Temperature':<14s} {'Fast':<10s} {'No':<15s} {'Creative writing'}")
    print(f"  {'Top-k':<14s} {'Medium':<10s} {'No':<15s} {'Brainstorming'}")
    print(f"  {'Top-p':<14s} {'Medium':<10s} {'No':<15s} {'General chat'}")
    print()
    print("  Key insight: Greedy is fastest and deterministic.")
    print("  Sampling adds latency but improves diversity.")
    print("  For caching, only greedy (temp=0) gives deterministic results.")
    print("=" * 60)


if __name__ == "__main__":
    main()
