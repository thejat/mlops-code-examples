# RAG Evaluation Metrics MWE

**Measure retrieval and generation quality for RAG pipelines — pure Python, no external dependencies.**

## Overview

This MWE implements the key metrics for evaluating RAG systems:

1. **Precision@k** — fraction of top-k retrieved documents that are relevant
2. **Recall@k** — fraction of relevant documents found in top-k
3. **MRR** — mean reciprocal rank of first relevant result
4. **Groundedness** — keyword overlap between response and retrieved context
5. **Multi-query report** — aggregate metrics with diagnostic guidance

## Prerequisites

- Python 3.10+
- No external packages required

## Setup (2 steps)

```bash
# 1. Navigate to the MWE directory
cd module7/mwe/rag-eval-metrics

# 2. Run the demo
python main.py
```

## Expected Output

The script runs 5 demos across mock retrieval data:

| Demo | Metric | Key Lesson |
|------|--------|------------|
| 1 & 2 | Precision@k, Recall@k | Increasing k improves recall but may lower precision |
| 3 | MRR | Measures how quickly the first relevant doc appears |
| 4 | Groundedness | Keyword overlap detects ungrounded claims |
| 5 | Full report | Aggregate metrics diagnose retrieval vs generation failures |

## Key Concepts Demonstrated

| Concept | Implementation |
|---------|----------------|
| Precision@k | `precision_at_k()` — set intersection of top-k and relevant |
| Recall@k | `recall_at_k()` — fraction of relevant docs found |
| Reciprocal Rank | `reciprocal_rank()` — 1/rank of first relevant result |
| MRR | `mean_reciprocal_rank()` — average RR across queries |
| Groundedness | `groundedness_score()` — content word overlap |
| Faithfulness | `faithfulness_check()` — sentence-level coverage check |

## Extension Prompt

**Challenge:** Modify the MWE to:
1. Add NDCG (Normalized Discounted Cumulative Gain) as a metric
2. Implement a confusion matrix for retrieval (relevant vs retrieved)
3. Add a "hallucination detector" that flags specific unsupported claims
4. Export the evaluation report as JSON for integration with CI/CD

## Related Reading

- RAG Evaluation Metrics (`notes7/instructional-materials/rag-evaluation-metrics.md`)
- Retrieval Query Design (`notes7/instructional-materials/retrieval-query-design.md`)
