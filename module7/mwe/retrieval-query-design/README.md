# Retrieval Query Design MWE

**Query design patterns for effective RAG retrieval — pure Python, no external dependencies.**

## Overview

This MWE demonstrates strategies that improve retrieval quality in RAG pipelines using simulated embeddings:

1. **Top-k sweep** — precision/recall trade-off at different k values
2. **Metadata filtering** — filter by category or date before similarity search
3. **Query expansion** — add synonyms and related terms to improve recall
4. **Score thresholding** — reject low-confidence results
5. **Hybrid search** — combine keyword + semantic scores with weighted fusion

## Prerequisites

- Python 3.10+
- No external packages required (uses simulated embeddings)

## Setup (2 steps)

```bash
# 1. Navigate to the MWE directory
cd module7/mwe/retrieval-query-design

# 2. Run the demo
python main.py
```

## Expected Output

| Demo | Pattern | Key Lesson |
|------|---------|------------|
| 1. Top-k sweep | k=1,2,3,5,8,12 | k=3 maximizes precision while achieving full recall |
| 2. Metadata filtering | Category and date filters | Filters reduce noise before scoring |
| 3. Query expansion | Synonym injection | Matches paraphrased content |
| 4. Score thresholding | Similarity cutoffs | Removes noise; threshold=0.5 balances precision/recall |
| 5. Hybrid search | 60% semantic + 40% keyword | Combines strengths of both methods |

## Key Concepts Demonstrated

| Concept | Implementation |
|---------|----------------|
| Cosine similarity | `cosine_similarity()` — manual dot product calculation |
| Simulated embeddings | `make_similar_embedding()` — controllable similarity |
| Keyword search | `keyword_search()` — term overlap scoring |
| Metadata pre-filtering | List comprehension on category/date before search |
| Query expansion | Synonym dictionary lookup |
| Weighted fusion | `alpha * semantic + (1-alpha) * keyword` |

## Extension Prompt

**Challenge:** Modify the MWE to:
1. Add Maximum Marginal Relevance (MMR) re-ranking for diversity
2. Implement a learned router that selects search strategy per query
3. Add a query decomposition step for multi-part questions
4. Build a feedback loop where retrieval evaluation adjusts k automatically

## Related Reading

- Retrieval Query Design (`notes7/instructional-materials/retrieval-query-design.md`)
- Vector Databases (`notes7/instructional-materials/vector-databases.md`)
