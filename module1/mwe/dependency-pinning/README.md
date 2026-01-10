# Dependency Pinning MWE

**Exact package versions ensure reproducible ML environments across machines and time.**

## Prerequisites

- Python 3.10+
- Linux, macOS, or Windows
- pip (included with Python)

## Quick Start (3 Steps)

```bash
# 1. Clone and navigate
git clone <repo-url>
cd src/module1/learning-activities/mwe/dependency-pinning

# 2. Create environment and install pinned dependencies
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. Run the demonstration
python main.py
```

## Expected Output

```
============================================================
DEPENDENCY PINNING - Reproducibility Demonstration
============================================================

 Environment Snapshot:
   Python: 3.10.x
   NumPy:  1.24.3
   Hash:   <16-char-hash>

🔢 Numerical Results (seed=42):
   Mean: 0.0044569909
   Std:  0.9920438018
   Sum:  4.4569908697

 Key Insight:
   If your teammate gets a DIFFERENT hash, your environments differ!
   This is why we pin dependencies with exact versions (==).

 Expected hash with pinned requirements: a1b2c3d4e5f6...
   (Compare this with expected_output/hash.txt)

============================================================
```

The **hash value** should match across all machines using the same pinned `requirements.txt`. If it differs, environments are not identical.

## What This Demonstrates

1. **Environment Fingerprinting** - Captures exact package versions
2. **Reproducibility Hash** - Verifies identical numerical behavior
3. **Version Sensitivity** - Shows how results depend on library versions

## Extension Challenge

 **For Advanced Learners:**

1. Modify `requirements.txt` to use `numpy>=1.20` instead of `numpy==1.24.3`
2. Run `pip install -r requirements.txt` on two different machines
3. Compare the hashes - do they match?
4. Document your findings in a short write-up

This exercise demonstrates why `>=` specifiers can break reproducibility in ML pipelines.

## Files

| File | Purpose |
|------|---------|
| `main.py` | Core demonstration (~90 lines) |
| `requirements.txt` | Pinned dependencies (1 package) |
| `expected_output/hash.txt` | Reference hash for verification |