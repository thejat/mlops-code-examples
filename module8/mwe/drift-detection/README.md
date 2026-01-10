# Drift Detection MWE

Detect data drift using Population Stability Index (PSI) to compare reference and production distributions.

## Prerequisites

- Python 3.8+
- Linux, macOS, or Windows

## Setup & Run

```bash
# 1. Clone and navigate
git clone <repo-url>
cd src/module8/learning-activities/mwe/drift-detection

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the demo
python main.py
```

## Expected Output

The script generates synthetic data with intentional drift in the "age" feature, then detects and reports it:

```
=== Drift Detection Demo ===

Generating synthetic data...
Comparing reference vs current distributions...

Feature         PSI Severity Drift?
----------------------------------------
age          0.1547 low      no
income       0.0065 none     no
score        0.0017 none     no

=== Summary ===
Features analyzed: 3
Features with drift: 0
Action: Continue monitoring
```

PSI values will vary slightly due to random sampling, but "age" should consistently show elevated PSI (~0.15-0.20) due to the simulated population shift.

## PSI Interpretation

| PSI Range | Interpretation | Action |
|-----------|---------------|--------|
| < 0.1 | No significant change | Continue monitoring |
| 0.1 - 0.2 | Moderate change | Investigate |
| > 0.2 | Significant drift | Retrain model |

## Extension Challenge

**Advanced:** Modify the script to:
1. Add KL divergence as a second drift metric
2. Implement categorical drift detection using chi-square test
3. Generate a visualization comparing the distributions (requires matplotlib)