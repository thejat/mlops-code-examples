# Synthetic Data Benchmark - Minimum Working Example

**Pattern:** Generate reproducible synthetic datasets, verify with hashing, benchmark pandas operations at multiple scales, and analyze scaling behavior.

---

## Prerequisites

- Python 3.9+ (tested on 3.9, 3.10, 3.11)
- ~100MB disk space for dependencies
- No Docker required

---

## Setup (3 Steps)

### 1. Clone and Navigate

```bash
git clone <repository-url>
cd module5/mwe/synthetic-data-benchmark
```

### 2. Install Dependencies

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Run

```bash
python main.py
```

---

## Expected Output

```
============================================================
SYNTHETIC DATA BENCHMARK
Reproducible Generation, Hashing, and Scaling Analysis
============================================================

[1/5] Generating synthetic transaction data (seed=42)...
      →      1,000 rows  hash=ab03334ac2a0f702  (0.010s)
      →     10,000 rows  hash=0f63641162f3f7df  (0.095s)
      →    100,000 rows  hash=5eb60c9c1ec288f1  (0.337s)
      →  1,000,000 rows  hash=ecd55642f6493866  (3.476s)

[2/5] Verifying reproducibility...
      → Re-generated 10,000 rows with seed=42
      → Hash match: ✅  (0f63641162f3f7df)
      → Different seed (99) → different hash: ✅  (85d299dc4804e62e)

[3/5] Benchmarking pandas feature engineering at each scale...

      Scale: 1,000 rows
      transform: 0.0034s

      Scale: 10,000 rows
      transform: 0.0134s

      Scale: 100,000 rows
      transform: 0.0470s

      Scale: 1,000,000 rows
      transform: 0.2783s

[4/5] Scaling analysis...

============================================================
SCALING ANALYSIS
============================================================
     Scale          Rows    Time (s)     vs 1x  Efficiency
  ────────  ────────────  ──────────  ────────  ──────────
        1x         1,000      0.0034      1.0x      100.0%
       10x        10,000      0.0134      3.9x      253.7%
      100x       100,000      0.0470     13.8x      723.4%
     1000x     1,000,000      0.2783     81.9x     1221.7%
```

**Note:** Timing values vary by machine. Hash values are deterministic and should match exactly with the same seed.

---

## Key Concepts Demonstrated

| Concept | Where in Code |
|---------|---------------|
| Synthetic data generation | `generate_data.py` — `generate_transactions(n_rows, seed)` |
| Reproducible hashing | `generate_data.py` — `compute_data_hash()` with SHA-256 |
| Hash verification | `main.py` — same seed → same hash; different seed → different hash |
| Timing wrapper | `main.py` — `@contextmanager benchmark(name, metrics)` |
| Fair benchmarking | `main.py` — warm-up run discarded before timed run |
| Scaling analysis | `main.py` — `analyze_scaling()` with efficiency percentages |
| Benchmark report | `main.py` — JSON metrics dictionary |

---

## Milestone 4 Usage

The `generate_data.py` file matches the Milestone 4 deliverable. Use it standalone for 10M+ row generation:

```bash
# Generate 10M rows as Parquet (Milestone 4 requirement)
python generate_data.py --n_rows 10000000 --seed 42 --output data/transactions.parquet

# Generate as CSV instead
python generate_data.py --n_rows 10000000 --seed 42 --output data/transactions.csv

# Print summary only (no file output)
python generate_data.py --n_rows 10000000 --seed 42
```

### Using as a Module

```python
from generate_data import generate_transactions, compute_data_hash

# Generate data
df = generate_transactions(n_rows=10_000_000, seed=42)

# Verify reproducibility
df2 = generate_transactions(n_rows=10_000_000, seed=42)
assert compute_data_hash(df) == compute_data_hash(df2)
```

---

## Data Schema

The generated transactions DataFrame has these columns:

| Column | Type | Distribution | Description |
|--------|------|-------------|-------------|
| `user_id` | string | Uniform over pool | Format: `u000000` – `u099999` |
| `amount` | float | Exponential (scale=50) | Transaction amount in dollars |
| `timestamp` | datetime | Uniform over 90 days | Starting from 2025-01-01 |
| `category` | string | Weighted (8 categories) | food, electronics, clothing, etc. |
| `merchant_id` | string | Uniform over pool | Format: `m00000` – `m09999` |

---

## Files Included

| File | Purpose |
|------|---------|
| [`generate_data.py`](generate_data.py) | Standalone data generator (Milestone 4 deliverable) with CLI |
| [`main.py`](main.py) | End-to-end benchmark demo: generate → verify → benchmark → report |
| [`requirements.txt`](requirements.txt) | Python dependencies (numpy, pandas, pyarrow) |

---

## Project Structure

```
synthetic-data-benchmark/
├── generate_data.py             # Standalone data generator (CLI + module)
├── main.py                      # Benchmark demo script
├── requirements.txt             # Dependencies
├── expected_output/
│   └── sample_output.txt
└── README.md
```

---

## Extension Challenges

### 1. Add Data Skew

Modify `generate_transactions()` to introduce skew (a few users generate most transactions):

```python
# Zipf distribution for user selection (realistic skew)
user_indices = np.random.zipf(a=1.5, size=n_rows) % n_users
data["user_id"] = [user_ids[i] for i in user_indices]
```

### 2. Benchmark with Spark/Ray

Use the same `generate_data.py` to create data, then benchmark with PySpark or Ray:

```python
from generate_data import generate_transactions

df = generate_transactions(n_rows=10_000_000, seed=42)
df.to_parquet("data/transactions.parquet")

# Then load in PySpark:
# spark_df = spark.read.parquet("data/transactions.parquet")
```

### 3. Add Null Values and Outliers

Extend the generator for data quality testing:

```python
# Inject 2% nulls
null_mask = np.random.random(n_rows) < 0.02
data["amount"] = np.where(null_mask, np.nan, data["amount"])
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: generate_data` | Run from the `synthetic-data-benchmark/` directory |
| `ModuleNotFoundError: numpy` | Activate your virtual environment and install requirements |
| Parquet write fails | Ensure `pyarrow` is installed: `pip install pyarrow` |
| Hash mismatch on same seed | Check numpy/pandas versions match across environments |
| 10M rows is slow | Generation takes ~30-60s; consider SSD storage for parquet output |

---

## Resources

- [NumPy Random Generator](https://numpy.org/doc/stable/reference/random/index.html)
- [Pandas DataFrame](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html)
- [PyArrow Parquet](https://arrow.apache.org/docs/python/parquet.html)
