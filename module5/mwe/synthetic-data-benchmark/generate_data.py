#!/usr/bin/env python3
"""
Synthetic Data Generator for Distributed Computing Benchmarks.

Generates reproducible synthetic transaction datasets with configurable size.
Matches Milestone 4 deliverable: generate_data.py

Usage:
    # As a module:
    from generate_data import generate_transactions, compute_data_hash
    df = generate_transactions(n_rows=10_000_000, seed=42)

    # As a CLI tool:
    python generate_data.py --n_rows 10000000 --seed 42 --output data/transactions.parquet
"""

import argparse
import hashlib
import time
from pathlib import Path

import numpy as np
import pandas as pd


def generate_transactions(n_rows: int, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic transaction data with realistic distributions.

    Args:
        n_rows: Number of transaction rows to generate.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with columns: user_id, amount, timestamp, category, merchant_id
    """
    np.random.seed(seed)

    # User pool: ~1% of rows as unique users (min 10, max 100_000)
    n_users = max(10, min(100_000, n_rows // 100))
    user_ids = [f"u{i:06d}" for i in range(n_users)]

    # Category distribution (weighted)
    categories = ["food", "electronics", "clothing", "travel", "entertainment",
                   "utilities", "healthcare", "education"]
    category_weights = [0.25, 0.15, 0.15, 0.10, 0.10, 0.10, 0.08, 0.07]

    # Merchant pool
    n_merchants = max(5, min(10_000, n_rows // 500))
    merchant_ids = [f"m{i:05d}" for i in range(n_merchants)]

    # Generate timestamp range: 90 days of data
    base_ts = pd.Timestamp("2025-01-01")
    ts_offsets = np.random.uniform(0, 90 * 24 * 3600, size=n_rows)

    data = {
        "user_id": np.random.choice(user_ids, size=n_rows),
        "amount": np.round(np.random.exponential(scale=50.0, size=n_rows), 2),
        "timestamp": [base_ts + pd.Timedelta(seconds=float(s)) for s in ts_offsets],
        "category": np.random.choice(categories, size=n_rows, p=category_weights),
        "merchant_id": np.random.choice(merchant_ids, size=n_rows),
    }

    return pd.DataFrame(data)


def compute_data_hash(df: pd.DataFrame) -> str:
    """
    Compute a deterministic hash of a DataFrame for verification.

    Uses pandas hash_pandas_object + SHA-256 to produce a reproducible
    fingerprint. Same data with same seed always produces same hash.

    Args:
        df: DataFrame to hash.

    Returns:
        First 16 characters of SHA-256 hex digest.
    """
    content = pd.util.hash_pandas_object(df).values
    return hashlib.sha256(content.tobytes()).hexdigest()[:16]


def main():
    """CLI entry point for standalone data generation."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic transaction data for benchmarking."
    )
    parser.add_argument(
        "--n_rows", type=int, default=100_000,
        help="Number of rows to generate (default: 100000)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output file path (.parquet or .csv). If omitted, prints summary only."
    )

    args = parser.parse_args()

    print(f"Generating {args.n_rows:,} rows with seed={args.seed}...")
    start = time.perf_counter()
    df = generate_transactions(n_rows=args.n_rows, seed=args.seed)
    elapsed = time.perf_counter() - start

    data_hash = compute_data_hash(df)
    print(f"  Generated in {elapsed:.2f}s")
    print(f"  Shape: {df.shape}")
    print(f"  Hash:  {data_hash}")
    print(f"  Unique users: {df['user_id'].nunique():,}")
    print(f"  Amount range: ${df['amount'].min():.2f} – ${df['amount'].max():.2f}")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if args.output.endswith(".csv"):
            df.to_csv(output_path, index=False)
        else:
            df.to_parquet(output_path, index=False)

        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"  Saved to: {output_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
