# Statistical Testing MWE

## Overview

Demonstrates the statistical testing methodology needed to make rigorous decisions
from A/B experiments on ML systems. Covers proportion tests, t-tests, multiple
comparison correction, confidence-interval-based decisions, guardrail checks, and
an integrated decision framework that combines all of these.

## Prerequisites

- Python 3.9+
- `numpy`, `scipy`

## Setup and Run

```bash
# 1. Navigate to this directory
cd module8/mwe/statistical-testing

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the demo
python main.py
```

## Expected Output

Six numbered demos are printed to the terminal:

1. **Two-Proportion Z-Test** — compares binary conversion rates between control
   and treatment using pooled standard error and a z-statistic.
2. **Welch's T-Test** — compares continuous metric means (e.g., retrieval
   similarity scores) with unequal-variance correction.
3. **Bonferroni Correction** — adjusts the significance threshold when testing
   multiple metrics simultaneously, preventing false discoveries.
4. **CI Decision Framework** — maps four realistic scenarios (strong positive,
   weak/underpowered, null, negative) to actionable decisions using confidence
   interval position and width.
5. **Guardrail Checks** — verifies that secondary metrics (latency, error rate,
   cache hit rate) have not degraded beyond acceptable thresholds.
6. **Integrated Decision Framework** — combines the primary metric test
   (with Bonferroni correction), secondary metric, and guardrail compliance
   into a single ship/extend/keep decision.

## Key Concepts Demonstrated

- **Two-proportion z-test**: pooled proportion, z-statistic, two-sided p-value
- **Welch's t-test**: Welch-Satterthwaite degrees of freedom, unequal variance
- **Bonferroni correction**: α/k threshold to control family-wise error rate
- **Confidence intervals**: CI-based decisions vs. p-value-only decisions
- **Guardrail metrics**: max degradation thresholds, higher-/lower-is-better
- **Decision framework**: combining statistical significance, practical
  significance (lift direction), and guardrail compliance

## Extension Challenges

1. Implement Benjamini-Hochberg FDR correction as an alternative to Bonferroni.
2. Add sequential testing (e.g., group-sequential boundaries) so you can stop
   the experiment early when the result is clear.
3. Compute the minimum detectable effect (MDE) for a given sample size and power.
4. Extend the guardrail framework to accept per-metric one-sided tests
   (e.g., latency must not increase, accuracy must not decrease).

## Related Reading

- `mod8notes/instructional-materials/statistical-testing-ml.md`
- `mod8notes/instructional-materials/ab-testing-fundamentals.md`
- `mod8notes/slides/module8-slides.md` (slides on Bonferroni, CI decisions)
