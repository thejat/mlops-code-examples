# Risk Scoring MWE

**NIST AI RMF-style risk matrix, risk register, and deployment gate — no external dependencies.**

## Overview

This MWE demonstrates AI risk management for ML and LLM/RAG systems:

1. **Risk matrix** — 5×5 likelihood × severity grid with level classification
2. **ML risk register** — 5 risks for traditional ML (bias, drift, misuse, accountability, vendor)
3. **LLM/RAG risk register** — 3 additional risks (prompt injection, KB staleness, data leaks)
4. **Residual risk** — inherent vs residual scores after applying controls
5. **Deployment gate** — go/no-go decision based on risk profile
6. **Treatment summary** — mitigate/accept/transfer/avoid distribution

## Prerequisites

- Python 3.8+
- No external dependencies (stdlib only)

## Setup (3 steps)

```bash
# 1. Navigate to the MWE directory
cd module8/mwe/risk-scoring

# 2. No dependencies to install

# 3. Run the demo
python main.py
```

## Expected Output

| Demo | What It Shows | Key Lesson |
|------|--------------|------------|
| 1. Risk Matrix | 5×5 scoring grid | Score = likelihood × severity |
| 2. ML Register | 5 traditional ML risks | Each risk needs owner + controls + trigger |
| 3. LLM Register | 3 generative AI risks | Same framework applies to LLM/RAG |
| 4. Residual Risk | Before vs after controls | Controls reduce risk 50-73% |
| 5. Deploy Gate | PROCEED WITH CONDITIONS | Missing residual data blocks full approval |
| 6. Treatment | 7 mitigate, 1 transfer | Most risks are mitigated, not avoided |

## Key Concepts Demonstrated

| Concept | Implementation |
|---------|----------------|
| Risk scoring | `compute_risk_score()` — likelihood × severity |
| Risk levels | LOW (1-4), MEDIUM (5-9), HIGH (10-15), CRITICAL (16-25) |
| Risk register | `RiskEntry` dataclass with ID, controls, owner, trigger |
| Treatment options | Mitigate, Accept, Transfer, Avoid |
| Residual risk | Re-scored after controls are applied |
| Deployment gate | Block / Proceed with conditions / Proceed |

## Extension Challenges

1. Add a `to_markdown()` method to export the risk register as a table
2. Implement a risk trend tracker that shows how scores change over time
3. Add system boundary diagram integration (map boundary crossings to risks)
4. Build an incident response plan template that references risk register entries

## Connection to Final Project

This MWE directly supports **Component 5: AI Risk Assessment**:

- The risk matrix provides the scoring framework for your assessment
- The risk register structure is exactly what you submit
- The deployment gate logic demonstrates the decision process
- LLM/RAG risks (R006-R008) can be adapted to your specific system

## Related Reading

- AI Risk Management (`mod8notes/instructional-materials/ai-risk-management.md`)
- System Boundary Diagrams (`mod8notes/instructional-materials/system-boundary-diagrams.md`)
- Bias & Fairness Assessment (`mod8notes/instructional-materials/bias-fairness-assessment.md`)
