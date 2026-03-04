#!/usr/bin/env python3
"""
PySpark Word Count - Minimum Working Example

Demonstrates: Spark cluster connection, DataFrame API, distributed word counting.
Uses the same sample text as the Ray MWE for direct comparison.

Run with: docker exec -it spark-master /opt/spark/bin/spark-submit /opt/spark/apps/word_count.py
"""

import time
from pyspark.sql import SparkSession
from pyspark.sql.functions import explode, split, lower, col, count

# Same sample text batches as Ray MWE (module5/mwe/ray-map-reduce/main.py)
TEXT_BATCHES = [
    "the quick brown fox jumps over the lazy dog",
    "ray makes distributed computing easy and fast",
    "machine learning pipelines benefit from parallel processing",
    "the fox and the dog are friends in the forest",
    "distributed systems scale horizontally for big data",
]


def main():
    print("=" * 60)
    print("PYSPARK WORD COUNT: Distributed Word Count Example")
    print("=" * 60)

    # Connect to Spark cluster
    print("\n[1/5] Connecting to Spark cluster...")
    spark = SparkSession.builder \
        .appName("WordCount") \
        .master("spark://spark-master:7077") \
        .getOrCreate()

    sc = spark.sparkContext
    print(f"  Connected to: {sc.master}")
    print(f"  App ID: {sc.applicationId}")
    print(f"  Spark UI: http://localhost:4040 (accessible while job runs)")

    # Create DataFrame from text batches
    print(f"\n[2/5] Processing {len(TEXT_BATCHES)} text batches...")

    # Parallelize text batches across cluster
    rdd = sc.parallelize(TEXT_BATCHES)
    df = spark.createDataFrame(rdd.map(lambda x: (x,)), ["text"])

    # Word count using DataFrame API
    print("\n[3/5] Building word count query...")
    word_counts = df \
        .select(explode(split(lower(col("text")), " ")).alias("word")) \
        .groupBy("word") \
        .agg(count("*").alias("count")) \
        .orderBy(col("count").desc())

    # Show execution plan
    print("\n" + "=" * 60)
    print("EXECUTION PLAN (Physical)")
    print("=" * 60)
    word_counts.explain(mode="simple")
    
    print("\n" + "=" * 60)
    print("EXECUTION PLAN (Extended - Logical + Physical)")
    print("=" * 60)
    word_counts.explain(mode="extended")

    # Pause to allow viewing Spark UI
    print("\n" + "-" * 60)
    print("Spark Application UI available at: http://localhost:4040")
    print("  - Click 'SQL / DataFrame' tab to see query plans")
    print("  - Click 'Stages' tab to see execution stages")
    print("  - Click 'Executors' tab to see worker details")
    print("-" * 60)
    print("Waiting 10 seconds before collecting results...")
    print("(Press Ctrl+C to skip wait)")
    try:
        time.sleep(10)
    except KeyboardInterrupt:
        print("Skipping wait...")

    # Collect results
    print("\n[4/5] Collecting results from workers...")
    results = word_counts.collect()

    # Display results (same format as Ray MWE)
    print("\n" + "=" * 60)
    print("RESULTS: Top 10 Most Frequent Words")
    print("=" * 60)

    for row in results[:10]:
        bar = "█" * row["count"]
        print(f"  {row['word']:15} {row['count']:3} {bar}")

    total_unique = word_counts.count()
    total_occurrences = sum(r["count"] for r in results)

    print(f"\nTotal unique words: {total_unique}")
    print(f"Total word occurrences: {total_occurrences}")

    # Show final Spark UI hint
    print("\n[5/5] Job complete!")
    print("-" * 60)
    print("View completed job details at: http://localhost:8080")
    print("  - Click on 'app-...' under 'Completed Applications'")
    print("-" * 60)

    # Cleanup
    spark.stop()
    print("\nSpark session stopped.")
    print("=" * 60)


if __name__ == "__main__":
    main()