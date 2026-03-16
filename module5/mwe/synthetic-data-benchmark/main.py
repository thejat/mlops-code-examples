#!/usr/bin/env python3
"""
Synthetic Data Benchmark - Minimum Working Example

Demonstrates: Reproducible data generation, hash verification, timing wrappers,
multi-scale benchmarking, and scaling analysis.

Run with: python main.py
"""

import json
import time
from contextlib import contextmanager

import numpy as np
import pandas as pd

from generate_data import generate_transactions, compute_data_hash


# ---------------------------------------------------------------------------
# Benchmark utilities (from slides: timing wrapper pattern, lines 504-522)
# ---------------------------------------------------------------------------

@contextmanager
def benchmark(name: str, metrics: dict):
    """Context manager to time a code block and record the result.

    Args:
        name: Label for this benchmark step.
        metrics: Dictionary to store timing results in-place.

    Yields:
        Control to the wrapped block.
    """
    start_time = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start_time
    metrics[name] = {"wall_time_sec": round(elapsed, 4)}
    print(f"      {name}: {elapsed:.4f}s")


def feature_engineering_local(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the same feature transforms used in the slides.

    Mirrors the PySpark/Ray pipeline so students can later compare
    local pandas timing against distributed execution.
    """
    # Add derived columns
    df = df.copy()
    df["log_amount"] = np.log1p(df["amount"])
    df["hour_of_day"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek

    # Group-by aggregation (triggers data movement in distributed setting)
    user_stats = df.groupby("user_id").agg(
        total_amount=("amount", "sum"),
        mean_amount=("amount", "mean"),
        tx_count=("amount", "count"),
        unique_categories=("category", "nunique"),
        unique_merchants=("merchant_id", "nunique"),
    ).reset_index()

    return user_stats


def analyze_scaling(results: list[dict]) -> None:
    """Analyze how processing time scales with data size.

    Prints efficiency metrics comparing actual scaling against ideal
    linear scaling (from slides: scaling analysis, lines 550-561).
    """
    baseline = results[0]
    baseline_rows = baseline["n_rows"]
    baseline_time = baseline["total_time_sec"]

    print("\n" + "=" * 60)
    print("SCALING ANALYSIS")
    print("=" * 60)
    print(f"  {'Scale':>8}  {'Rows':>12}  {'Time (s)':>10}  {'vs 1x':>8}  {'Efficiency':>10}")
    print(f"  {'─' * 8}  {'─' * 12}  {'─' * 10}  {'─' * 8}  {'─' * 10}")

    for r in results:
        scale = r["n_rows"] / baseline_rows
        time_ratio = r["total_time_sec"] / baseline_time if baseline_time > 0 else 0
        # Ideal: linear scaling (Nx data = Nx time)
        # Efficiency: how close to ideal (100% = perfectly linear)
        efficiency = (scale / time_ratio * 100) if time_ratio > 0 else 0

        print(f"  {scale:>7.0f}x  {r['n_rows']:>12,}  {r['total_time_sec']:>10.4f}  "
              f"{time_ratio:>7.1f}x  {efficiency:>9.1f}%")

    print()
    print("  Key insight: Sub-linear scaling (efficiency > 100%) at small sizes")
    print("  reflects fixed overhead; super-linear slowdown at large sizes")
    print("  reflects memory pressure and cache effects.")


def main():
    print("=" * 60)
    print("SYNTHETIC DATA BENCHMARK")
    print("Reproducible Generation, Hashing, and Scaling Analysis")
    print("=" * 60)

    # ── Step 1: Generate data at multiple scales ──────────────────────
    scales = [1_000, 10_000, 100_000, 1_000_000]
    seed = 42

    print(f"\n[1/5] Generating synthetic transaction data (seed={seed})...")
    generated = {}
    for n in scales:
        start = time.perf_counter()
        df = generate_transactions(n_rows=n, seed=seed)
        elapsed = time.perf_counter() - start
        data_hash = compute_data_hash(df)
        generated[n] = {"df": df, "hash": data_hash, "gen_time": elapsed}
        print(f"      → {n:>10,} rows  hash={data_hash}  ({elapsed:.3f}s)")

    # ── Step 2: Verify reproducibility ────────────────────────────────
    print(f"\n[2/5] Verifying reproducibility...")
    verify_n = 10_000
    df_check = generate_transactions(n_rows=verify_n, seed=seed)
    hash_check = compute_data_hash(df_check)
    original_hash = generated[verify_n]["hash"]

    if hash_check == original_hash:
        print(f"      → Re-generated {verify_n:,} rows with seed={seed}")
        print(f"      → Hash match: ✅  ({hash_check})")
    else:
        print(f"      → Hash MISMATCH: ❌")
        print(f"        Original: {original_hash}")
        print(f"        Regenerated: {hash_check}")

    # Also verify different seed produces different hash
    df_different = generate_transactions(n_rows=verify_n, seed=99)
    hash_different = compute_data_hash(df_different)
    if hash_different != original_hash:
        print(f"      → Different seed (99) → different hash: ✅  ({hash_different})")
    else:
        print(f"      → Different seed produced same hash: ❌ (unexpected)")

    # ── Step 3: Benchmark pandas operations at each scale ─────────────
    print(f"\n[3/5] Benchmarking pandas feature engineering at each scale...")
    benchmark_results = []

    for n in scales:
        print(f"\n      Scale: {n:,} rows")
        df = generated[n]["df"]
        step_metrics = {}

        # Warm-up run (discarded, per fair comparison methodology)
        _ = feature_engineering_local(df)

        # Timed run
        with benchmark("transform", step_metrics):
            result_df = feature_engineering_local(df)

        total_time = step_metrics["transform"]["wall_time_sec"]

        benchmark_results.append({
            "n_rows": n,
            "total_time_sec": total_time,
            "output_rows": len(result_df),
            "metrics": step_metrics,
        })

    # ── Step 4: Scaling analysis ──────────────────────────────────────
    print(f"\n[4/5] Scaling analysis...")
    analyze_scaling(benchmark_results)

    # ── Step 5: Benchmark report ──────────────────────────────────────
    print("[5/5] Benchmark report")
    print("-" * 60)

    report = {
        "environment": "local_pandas",
        "seed": seed,
        "scales": [],
    }

    for r in benchmark_results:
        entry = {
            "n_rows": r["n_rows"],
            "total_time_sec": r["total_time_sec"],
            "output_rows": r["output_rows"],
            "hash": generated[r["n_rows"]]["hash"],
        }
        report["scales"].append(entry)

    print(json.dumps(report, indent=2))

    print("\n" + "=" * 60)
    print("NOTE: For Milestone 4, generate 10M+ rows:")
    print("  python generate_data.py --n_rows 10000000 --seed 42 \\")
    print("      --output data/transactions.parquet")
    print("=" * 60)


if __name__ == "__main__":
    main()
