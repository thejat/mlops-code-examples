# A/B Test Simulation MWE

Complete A/B testing workflow for ML model evaluation: sample size calculation, user assignment, outcome simulation, and statistical analysis.

## Prerequisites

- Python 3.8+
- Linux, macOS, or Windows

## Setup & Run

```bash
# 1. Clone and navigate
git clone <repo-url>
cd src/module8/learning-activities/mwe/ab-test-simulation

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the demo
python main.py
```

## Expected Output

The script simulates an A/B test comparing a new model version against the control:

```
=== A/B Test Simulation Demo ===

Step 1: Sample Size Calculation
  Required per variant: 31,234
  Simulating: 20,000 total users

Step 2: Simulating Experiment
  Control users: 9,962
  Treatment users: 10,038

Step 3: Statistical Analysis

============================================================
A/B TEST SIMULATION RESULTS
============================================================

Experiment: model_v2_conversion_test

--- Sample Sizes ---
  Control: 9,962
  Treatment: 10,038

--- Conversion Rates ---
  Control: 0.0996 (9.96%)
  Treatment: 0.1053 (10.53%)
  Relative Lift: +5.72%

--- Statistical Analysis ---
  P-value: 0.019853
  95% CI: [+0.0009, +0.0105]
  Significant (α=0.05): Yes

--- Recommendation ---
  SHIP_TREATMENT
  → Deploy treatment to all users
```

Note: Results will vary slightly due to random sampling, but with the default 5% treatment effect, the test should typically detect a significant positive lift.

## Key Concepts Demonstrated

| Concept | Implementation |
|---------|---------------|
| Sample size calculation | Power analysis for proportion tests |
| User assignment | Deterministic hashing for consistency |
| Randomization | 50/50 split via hash bucketing |
| Statistical testing | Two-proportion z-test |
| Confidence intervals | 95% CI for rate difference |
| Decision framework | Ship/Keep/Extend recommendation |

## Interpretation Guide

| P-value | Lift Direction | Recommendation |
|---------|---------------|----------------|
| < 0.05 | Positive | SHIP_TREATMENT |
| < 0.05 | Negative | KEEP_CONTROL |
| ≥ 0.05 | Any | EXTEND_EXPERIMENT |

## Extension Challenges

**Challenge 1:** Add guardrail metrics

Modify the simulation to track a secondary metric (e.g., latency) and implement a check that prevents shipping if guardrails fail.

**Challenge 2:** Multiple comparison correction

Simulate testing 3 primary metrics and apply Bonferroni correction to control the family-wise error rate.

**Challenge 3:** Power curve visualization

Generate a plot showing how statistical power varies with sample size for different effect sizes.

## Connection to Final Project

This MWE directly supports **Component 2: A/B Test Design & Simulation**:

- The `calculate_sample_size()` function demonstrates proper experimental design
- The `simulate_experiment()` function shows realistic outcome generation
- The `analyze_experiment()` function implements statistically valid evaluation
- The recommendation logic provides the foundation for decision memos

Use this code as a starting point, then extend it with guardrail metrics and a written recommendation memo for your final submission.