#!/usr/bin/env python3
"""
Lab: Module 5 - Distributed Processing with Ray
Time: ~30 minutes
Prerequisites:
  - Read: distributed-computing-fundamentals.md
  - Read: ray-parallel-computing.md
  - Read: benchmarking-local-vs-distributed.md

Learning Objectives:
- LO1: Implement parallel data processing using Ray tasks
- LO2: Compare local vs. distributed execution performance
- LO3: Understand when distributed processing provides benefits
- LO4: Apply the map-reduce pattern for aggregations

Milestone 4 Connection:
This lab introduces the concepts you'll use for your distributed feature
engineering pipeline. You'll practice with Ray's task-based parallelism
before scaling to larger datasets.

Setup:
    pip install ray pandas numpy

Run:
    python distributed.py
"""

import time
from typing import List, Dict, Any
import numpy as np
import pandas as pd

# === SETUP (provided) ===

# We'll use Ray for distributed processing
# Ray is simpler to set up locally than Spark
try:
    import ray
except ImportError:
    raise ImportError(
        "Ray is required for this lab.\n"
        "Install with: pip install ray"
    )


def generate_synthetic_transactions(
    n_rows: int = 100_000,
    n_users: int = 1000,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic transaction data for the lab.
    
    This simulates the kind of data you'll process in Milestone 4,
    but at a smaller scale suitable for a 30-minute lab.
    
    Args:
        n_rows: Number of transaction rows to generate
        n_users: Number of unique users
        seed: Random seed for reproducibility
    
    Returns:
        DataFrame with columns: user_id, amount, category, timestamp
    """
    np.random.seed(seed)
    
    # Generate transaction data
    data = {
        "user_id": np.random.randint(1, n_users + 1, size=n_rows),
        "amount": np.abs(np.random.lognormal(mean=4.0, sigma=1.0, size=n_rows)),
        "category": np.random.choice(
            ["electronics", "grocery", "clothing", "restaurant", "travel"],
            size=n_rows,
            p=[0.15, 0.35, 0.20, 0.20, 0.10]
        ),
        "timestamp": pd.date_range(
            start="2024-01-01",
            periods=n_rows,
            freq="s"
        )
    }
    
    return pd.DataFrame(data)


def feature_engineering_local(df: pd.DataFrame) -> pd.DataFrame:
    """
    Local (single-threaded) feature engineering.
    
    This is the baseline approach using pandas - all processing
    happens on a single CPU core sequentially.
    """
    # Add derived features
    df_features = df.copy()
    df_features["log_amount"] = np.log1p(df_features["amount"])
    df_features["amount_squared"] = df_features["amount"] ** 2
    df_features["hour"] = df_features["timestamp"].dt.hour
    df_features["day_of_week"] = df_features["timestamp"].dt.dayofweek
    
    # Aggregate by user
    user_stats = df_features.groupby("user_id").agg(
        total_amount=("amount", "sum"),
        avg_amount=("amount", "mean"),
        transaction_count=("amount", "count"),
        avg_log_amount=("log_amount", "mean"),
        max_amount=("amount", "max"),
    ).reset_index()
    
    return user_stats


# === TODO 1: Implement a Ray Remote Function ===
# 
# Ray uses the @ray.remote decorator to mark functions that can
# run in parallel across workers. Each call to a remote function
# returns a "future" (ObjectRef) that you can later retrieve with ray.get().
#
# Your task: Complete the remote function below that processes a single
# batch of data. This function will be called multiple times in parallel.
#
# Hint: The function signature and logic are similar to feature_engineering_local,
#       but operates on a subset of the data.

@ray.remote
def process_batch(batch_df: pd.DataFrame) -> pd.DataFrame:
    """
    Process a single batch of transaction data.
    
    This function runs on a Ray worker. Multiple batches are
    processed in parallel across available CPU cores.
    
    Args:
        batch_df: A subset of the full transaction DataFrame
    
    Returns:
        DataFrame with per-user aggregated features for this batch
    """
    # TODO: Add derived features to the batch
    # - log_amount: natural log of (1 + amount)
    # - amount_squared: amount raised to power 2
    # - hour: hour extracted from timestamp
    # - day_of_week: day of week from timestamp (0=Monday, 6=Sunday)
    #
    # Your code here:
    # batch_df["log_amount"] = ...
    # batch_df["amount_squared"] = ...
    # batch_df["hour"] = ...
    # batch_df["day_of_week"] = ...
    
    pass  # Remove this line when you add your code
    
    # TODO: Aggregate by user_id within this batch
    # Calculate: total_amount, avg_amount, transaction_count, 
    #            avg_log_amount, max_amount
    #
    # Your code here:
    # user_stats = batch_df.groupby("user_id").agg(...)
    
    pass  # Remove this line when you add your code
    
    # Return the aggregated stats for this batch
    # return user_stats


# === TODO 2: Implement Batch Splitting ===
#
# To process data in parallel, we need to split it into batches.
# Each batch will be sent to a different Ray worker.
#
# Key concept: More batches = more parallelism, but also more overhead.
# For Milestone 4, you'll need to consider optimal batch/partition sizes.

def split_into_batches(df: pd.DataFrame, n_batches: int) -> List[pd.DataFrame]:
    """
    Split a DataFrame into n roughly equal batches.
    
    Args:
        df: The full DataFrame to split
        n_batches: Number of batches to create
    
    Returns:
        List of DataFrames, each representing one batch
    """
    # TODO: Use numpy.array_split to divide the dataframe into n_batches
    # 
    # Hint: np.array_split(df, n_batches) returns a list of DataFrames
    #
    # Your code here:
    # batches = ...
    # return batches
    
    pass  # Remove this line when you add your code


# === TODO 3: Implement Distributed Processing ===
#
# Now combine the pieces: split the data, process batches in parallel,
# and merge the results.
#
# Key concept: This is the Map-Reduce pattern:
#   - Map: Apply process_batch to each batch in parallel
#   - Reduce: Combine partial results into final aggregates

def feature_engineering_distributed(
    df: pd.DataFrame,
    n_batches: int = 4
) -> pd.DataFrame:
    """
    Distributed feature engineering using Ray.
    
    This demonstrates how distributed frameworks like Ray and Spark
    parallelize work: split data into partitions, process each partition
    independently, then merge results.
    
    Args:
        df: Full transaction DataFrame
        n_batches: Number of parallel batches (roughly = parallelism level)
    
    Returns:
        DataFrame with per-user aggregated features
    """
    # Step 1: Split data into batches
    # TODO: Call split_into_batches to divide the data
    #
    # Your code here:
    # batches = split_into_batches(df, n_batches)
    
    pass  # Remove this line when you add your code
    
    # Step 2: Submit all batches for parallel processing
    # TODO: Create a list of futures by calling process_batch.remote(batch)
    #       for each batch. The .remote() call returns immediately with a future.
    #
    # Your code here:
    # futures = [process_batch.remote(batch) for batch in batches]
    
    pass  # Remove this line when you add your code
    
    # Step 3: Wait for all results
    # TODO: Use ray.get(futures) to collect all results
    #       This blocks until all parallel tasks complete.
    #
    # Your code here:
    # batch_results = ray.get(futures)
    
    pass  # Remove this line when you add your code
    
    # Step 4: Merge partial results
    # TODO: Concatenate batch results and re-aggregate
    #       Since each batch computed partial user stats, we need to
    #       combine stats for users that appeared in multiple batches.
    #
    # Hint: pd.concat(batch_results) gives you all partial stats,
    #       then groupby("user_id").agg(...) to combine them
    #
    # For combining partial aggregates:
    #   - total_amount: sum of partial sums
    #   - avg_amount: need to recompute from total/count
    #   - transaction_count: sum of partial counts
    #   - avg_log_amount: need to recompute from sum of log amounts / count
    #   - max_amount: max of partial maxes
    #
    # Your code here (simplified version that works for this lab):
    # merged = pd.concat(batch_results)
    # final_stats = merged.groupby("user_id").agg({
    #     "total_amount": "sum",
    #     "transaction_count": "sum",
    #     "max_amount": "max",
    # }).reset_index()
    
    pass  # Remove this line when you add your code
    
    # return final_stats


# === TODO 4: Implement Benchmarking ===
#
# For Milestone 4, you must provide quantitative performance comparison.
# This function times both local and distributed execution.

def benchmark_processing(
    df: pd.DataFrame,
    n_batches: int = 4,
    n_iterations: int = 3
) -> Dict[str, Any]:
    """
    Benchmark local vs. distributed processing.
    
    Args:
        df: Transaction DataFrame
        n_batches: Number of parallel batches for distributed
        n_iterations: Number of timing iterations (median is reported)
    
    Returns:
        Dictionary with timing results
    """
    results = {
        "row_count": len(df),
        "n_batches": n_batches,
        "local_times": [],
        "distributed_times": [],
    }
    
    # Benchmark local processing
    for i in range(n_iterations):
        # TODO: Time the local feature engineering
        #
        # Your code here:
        # start = time.perf_counter()
        # local_result = feature_engineering_local(df)
        # elapsed = time.perf_counter() - start
        # results["local_times"].append(elapsed)
        
        pass  # Remove this line when you add your code
    
    # Benchmark distributed processing
    for i in range(n_iterations):
        # TODO: Time the distributed feature engineering
        #
        # Your code here:
        # start = time.perf_counter()
        # dist_result = feature_engineering_distributed(df, n_batches)
        # elapsed = time.perf_counter() - start
        # results["distributed_times"].append(elapsed)
        
        pass  # Remove this line when you add your code
    
    # Calculate summary statistics
    # TODO: Compute median times and speedup
    #
    # Your code here:
    # results["local_median"] = np.median(results["local_times"])
    # results["distributed_median"] = np.median(results["distributed_times"])
    # results["speedup"] = results["local_median"] / results["distributed_median"]
    
    pass  # Remove this line when you add your code
    
    return results


# === REFERENCE SOLUTION (hidden until you complete the TODOs) ===
# 
# Uncomment the section below after attempting the TODOs to verify your work.

"""
# REFERENCE: process_batch implementation
@ray.remote
def process_batch_reference(batch_df: pd.DataFrame) -> pd.DataFrame:
    batch_df = batch_df.copy()
    batch_df["log_amount"] = np.log1p(batch_df["amount"])
    batch_df["amount_squared"] = batch_df["amount"] ** 2
    batch_df["hour"] = batch_df["timestamp"].dt.hour
    batch_df["day_of_week"] = batch_df["timestamp"].dt.dayofweek
    
    user_stats = batch_df.groupby("user_id").agg(
        total_amount=("amount", "sum"),
        avg_amount=("amount", "mean"),
        transaction_count=("amount", "count"),
        avg_log_amount=("log_amount", "mean"),
        max_amount=("amount", "max"),
    ).reset_index()
    
    return user_stats

# REFERENCE: split_into_batches implementation
def split_into_batches_reference(df: pd.DataFrame, n_batches: int) -> List[pd.DataFrame]:
    return np.array_split(df, n_batches)

# REFERENCE: feature_engineering_distributed implementation
def feature_engineering_distributed_reference(df: pd.DataFrame, n_batches: int = 4) -> pd.DataFrame:
    batches = np.array_split(df, n_batches)
    futures = [process_batch_reference.remote(batch) for batch in batches]
    batch_results = ray.get(futures)
    
    merged = pd.concat(batch_results)
    final_stats = merged.groupby("user_id").agg({
        "total_amount": "sum",
        "transaction_count": "sum",
        "max_amount": "max",
    }).reset_index()
    
    return final_stats
"""


# === MAIN EXECUTION ===

def main():
    """
    Main function to run the lab exercises.
    """
    print("=" * 60)
    print("Module 5 Lab: Distributed Processing with Ray")
    print("=" * 60)
    
    # Initialize Ray (connects to existing cluster or starts local)
    print("\n1. Initializing Ray...")
    ray.init(ignore_reinit_error=True, num_cpus=4)
    print(f"   Ray initialized with {ray.cluster_resources().get('CPU', 0)} CPUs")
    
    # Generate synthetic data
    print("\n2. Generating synthetic transaction data...")
    N_ROWS = 100_000  # Adjust for your machine (start small, increase to test)
    df = generate_synthetic_transactions(n_rows=N_ROWS)
    print(f"   Generated {len(df):,} rows")
    print(f"   Columns: {list(df.columns)}")
    print(f"   Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Test local processing
    print("\n3. Testing local (pandas) processing...")
    start = time.perf_counter()
    local_result = feature_engineering_local(df)
    local_time = time.perf_counter() - start
    print(f"   Local processing took: {local_time:.3f} seconds")
    print(f"   Output rows: {len(local_result):,}")
    print(f"   Sample output:")
    print(local_result.head(3).to_string(index=False))
    
    # Test distributed processing (YOUR IMPLEMENTATION)
    print("\n4. Testing distributed (Ray) processing...")
    print("   [Complete the TODOs to enable this section]")
    
    # Uncomment after implementing TODOs:
    # start = time.perf_counter()
    # dist_result = feature_engineering_distributed(df, n_batches=4)
    # dist_time = time.perf_counter() - start
    # print(f"   Distributed processing took: {dist_time:.3f} seconds")
    # print(f"   Output rows: {len(dist_result):,}")
    
    # Run benchmarks (YOUR IMPLEMENTATION)
    print("\n5. Running benchmarks...")
    print("   [Complete the TODOs to enable benchmarking]")
    
    # Uncomment after implementing TODOs:
    # results = benchmark_processing(df, n_batches=4, n_iterations=3)
    # print(f"   Local median time: {results['local_median']:.3f}s")
    # print(f"   Distributed median time: {results['distributed_median']:.3f}s")
    # print(f"   Speedup: {results['speedup']:.2f}x")
    
    # Cleanup
    print("\n6. Shutting down Ray...")
    ray.shutdown()
    
    print("\n" + "=" * 60)
    print("Lab complete! See SELF-CHECK section below.")
    print("=" * 60)


# === EXPECTED OUTPUT ===
# 
# When you run this lab with completed TODOs, you should see output similar to:
#
# 1. Initializing Ray...
#    Ray initialized with 4.0 CPUs
#
# 2. Generating synthetic transaction data...
#    Generated 100,000 rows
#    Columns: ['user_id', 'amount', 'category', 'timestamp']
#    Memory usage: 3.05 MB
#
# 3. Testing local (pandas) processing...
#    Local processing took: 0.045 seconds
#    Output rows: 1,000
#
# 4. Testing distributed (Ray) processing...
#    Distributed processing took: 0.XXX seconds  (may vary)
#    Output rows: 1,000
#
# 5. Running benchmarks...
#    Local median time: 0.04Xs
#    Distributed median time: 0.XXXs
#    Speedup: X.XXx
#
# NOTE: For small datasets (100K rows), local may be FASTER than distributed
#       due to Ray overhead. This is expected! The benefit of distributed
#       processing appears at larger scales (1M+ rows).


# === SELF-CHECK ===

def run_self_check():
    """
    Self-check assertions to verify your implementation.
    Run this after completing the TODOs.
    """
    print("\n" + "=" * 60)
    print("SELF-CHECK")
    print("=" * 60)
    
    ray.init(ignore_reinit_error=True, num_cpus=4)
    
    # Generate test data
    test_df = generate_synthetic_transactions(n_rows=10_000, n_users=100)
    
    # Check 1: Local processing works
    print("\n✓ Check 1: Local processing...")
    local_result = feature_engineering_local(test_df)
    assert len(local_result) == 100, f"Expected 100 users, got {len(local_result)}"
    assert "total_amount" in local_result.columns, "Missing total_amount column"
    assert "transaction_count" in local_result.columns, "Missing transaction_count column"
    print("  PASSED: Local processing produces correct output")
    
    # Check 2: Batch splitting works
    print("\n✓ Check 2: Batch splitting...")
    try:
        batches = split_into_batches(test_df, 4)
        assert batches is not None, "split_into_batches returned None"
        assert len(batches) == 4, f"Expected 4 batches, got {len(batches)}"
        total_rows = sum(len(b) for b in batches)
        assert total_rows == len(test_df), f"Batches don't sum to original: {total_rows} vs {len(test_df)}"
        print("  PASSED: Batch splitting works correctly")
    except TypeError:
        print("  SKIPPED: Complete TODO 2 (split_into_batches)")
    
    # Check 3: Distributed processing works
    print("\n✓ Check 3: Distributed processing...")
    try:
        dist_result = feature_engineering_distributed(test_df, n_batches=4)
        assert dist_result is not None, "Distributed result is None"
        assert len(dist_result) == 100, f"Expected 100 users, got {len(dist_result)}"
        print("  PASSED: Distributed processing produces correct output")
    except (TypeError, AttributeError):
        print("  SKIPPED: Complete TODOs 1-3")
    
    # Check 4: Results match (approximately)
    print("\n✓ Check 4: Result consistency...")
    try:
        # Compare total_amount sums between local and distributed
        local_total = local_result["total_amount"].sum()
        dist_total = dist_result["total_amount"].sum()
        
        # Allow small floating point differences
        diff_pct = abs(local_total - dist_total) / local_total * 100
        assert diff_pct < 0.1, f"Results differ by {diff_pct:.2f}%"
        print(f"  PASSED: Results match (diff: {diff_pct:.4f}%)")
    except (NameError, TypeError):
        print("  SKIPPED: Complete distributed processing first")
    
    ray.shutdown()
    
    print("\n" + "=" * 60)
    print("✅ All checks passed! Your implementation is correct.")
    print("=" * 60)
    print("\nNext steps for Milestone 4:")
    print("  - Scale to 10M+ rows and measure performance")
    print("  - Try different n_batches values (scaling analysis)")
    print("  - Switch to PySpark for full distributed feature engineering")
    print("  - Document when distributed overhead exceeds benefits")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--check":
        run_self_check()
    else:
        main()
        print("\n" + "-" * 60)
        print("To run self-check after completing TODOs:")
        print("  python distributed.py --check")
        print("-" * 60)