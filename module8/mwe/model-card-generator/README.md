# Model Card Generator MWE

**Programmatic model card creation with markdown and JSON export — no external dependencies.**

## Overview

This MWE demonstrates how to build model cards as structured data and export them in multiple formats:

1. **ModelCard dataclass** — structured representation of all model metadata
2. **Subgroup metrics** — disaggregated performance across customer segments
3. **Markdown export** — human-readable documentation artifact
4. **JSON export** — machine-readable format for governance pipelines
5. **Completeness validation** — check required sections before deployment
6. **Subgroup analysis** — highlight performance disparities across segments

## Prerequisites

- Python 3.8+
- No external dependencies (stdlib only)

## Setup (3 steps)

```bash
# 1. Navigate to the MWE directory
cd module8/mwe/model-card-generator

# 2. No dependencies to install

# 3. Run the demo
python main.py
```

## Expected Output

| Demo | What It Shows | Key Lesson |
|------|--------------|------------|
| 1. Build Card | Populate all fields programmatically | Model cards are structured data, not free text |
| 2. Markdown Export | 89-line markdown document | Human-readable governance artifact |
| 3. JSON Export | Machine-readable format | Enables automated governance pipelines |
| 4. Validation (complete) | 9/9 sections pass | Deployment readiness gate |
| 5. Validation (incomplete) | 7/9 sections fail | Catches missing documentation early |
| 6. Subgroup Analysis | Enterprise segment underperforms | Aggregate metrics hide disparities |

## Key Concepts Demonstrated

| Concept | Implementation |
|---------|----------------|
| Structured documentation | `ModelCard` dataclass with typed fields |
| Subgroup disaggregation | `subgroup_metrics` dict per segment |
| Markdown generation | `to_markdown()` with tables and sections |
| JSON serialization | `to_json()` for programmatic access |
| Completeness checks | `validate_completeness()` returns pass/fail per section |
| Performance gaps | Spread analysis flags segments below overall by >0.05 |

## Extension Challenges

1. Add a `to_html()` method that generates a styled HTML model card
2. Implement a `SystemCard` variant for LLM/RAG systems (different fields)
3. Add version comparison: diff two model card versions to highlight changes
4. Integrate with MLflow to auto-populate metrics from a training run

## Connection to Final Project

This MWE directly supports **Component 3: Model Card & Governance Packet**:

- The `ModelCard` dataclass provides the structure for your documentation
- The `to_markdown()` output is the artifact you submit
- The `validate_completeness()` method ensures nothing is missing
- Subgroup analysis connects to fairness assessment (Component 5)

## Related Reading

- Model Cards (`mod8notes/instructional-materials/model-cards.md`)
- Bias & Fairness Assessment (`mod8notes/instructional-materials/bias-fairness-assessment.md`)
