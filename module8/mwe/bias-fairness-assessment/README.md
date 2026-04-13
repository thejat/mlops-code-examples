# Bias & Fairness Assessment MWE

**Quantitative fairness metrics for ML classification systems — requires numpy only.**

## Overview

This MWE demonstrates fairness assessment on a synthetic loan approval dataset with intentional bias:

1. **Synthetic dataset** — 5,000 applicants with gender and age group attributes
2. **Group metrics** — per-group accuracy, TPR, FPR, and selection rates
3. **Demographic parity** — equal positive outcome rates across groups
4. **Equalized odds** — equal TPR and FPR across groups
5. **Disparate impact** — 80% rule analysis with violation detection
6. **Combined report** — summary with pass/fail checks and recommendations

## Prerequisites

- Python 3.8+
- numpy

## Setup (3 steps)

```bash
# 1. Navigate to the MWE directory
cd module8/mwe/bias-fairness-assessment

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the demo
python main.py
```

## Expected Output

| Demo | What It Shows | Key Lesson |
|------|--------------|------------|
| 1. Dataset | Approval rates by group | Bias creates visible disparities |
| 2. Group Metrics | TPR, FPR, accuracy per group | Disaggregated metrics reveal hidden gaps |
| 3. Demographic Parity | Selection rate ratios vs reference | non_binary group fails 80% threshold |
| 4. Equalized Odds | TPR/FPR gaps across groups | Model error rates differ by group |
| 5. Disparate Impact | 80% rule for age groups | 18-25 group marginally fails threshold |
| 6. Combined Report | 5/5 checks fail | Multiple fairness definitions give different views |

## Key Concepts Demonstrated

| Concept | Implementation |
|---------|----------------|
| Demographic parity | `demographic_parity_ratio()` — selection rate ratio vs reference |
| Equalized odds | `equalized_odds_difference()` — TPR and FPR gap |
| Disparate impact | `disparate_impact_analysis()` — 80% rule with violation list |
| Group metrics | `compute_group_metrics()` — per-group TP, FP, TN, FN |
| Intentional bias | Synthetic data with group-correlated approval shifts |
| Incompatible definitions | Same model fails different fairness criteria |

## Extension Challenges

1. Implement threshold adjustment per group to satisfy demographic parity
2. Add calibration analysis — do predicted probabilities match true rates per group?
3. Compare pre-processing (resampling) vs post-processing (threshold adjustment) mitigation
4. Generate a `FairnessReport.to_markdown()` output for model card inclusion

## Connection to Final Project

This MWE directly supports **Component 5: AI Risk Assessment**:

- The fairness metrics identify bias risks for your risk register
- The 80% rule provides a quantitative threshold for deployment gates
- The combined report format connects to model card documentation (Component 3)
- Disparate impact violations become risk register entries with review triggers

## Related Reading

- Bias & Fairness Assessment (`mod8notes/instructional-materials/bias-fairness-assessment.md`)
- Model Cards (`mod8notes/instructional-materials/model-cards.md`)
- AI Risk Management (`mod8notes/instructional-materials/ai-risk-management.md`)
