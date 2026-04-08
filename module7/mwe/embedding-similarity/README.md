# Embedding Similarity MWE

**Explore embedding properties and similarity metrics for RAG — requires sentence-transformers.**

## Overview

This MWE demonstrates how embedding models represent text and how different similarity metrics affect retrieval:

1. **Cosine vs L2 distance** — same queries ranked by different metrics
2. **Semantic vs keyword matching** — embeddings understand synonyms
3. **Dimension inspection** — what embedding vectors look like
4. **Batch encoding performance** — 1-at-a-time vs batch speedup
5. **Cross-domain similarity** — embedding space structure across topics

## Prerequisites

- Python 3.9+
- ~500MB disk space (for model download on first run)

## Setup (3 steps)

```bash
# 1. Navigate to the MWE directory
cd module7/mwe/embedding-similarity

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the demo
python main.py
```

## Expected Output

| Demo | Comparison | Key Lesson |
|------|-----------|------------|
| 1. Cosine vs L2 | Same rankings for normalized vectors | Cosine is scale-invariant |
| 2. Semantic vs keyword | 'automobile'='car' with 0 word overlap | Embeddings capture meaning |
| 3. Dimension inspection | 384-dim dense vectors | All dims carry information |
| 4. Batch performance | ~10× speedup with batch_size=32 | Always batch in production |
| 5. Cross-domain | MLOps↔RAG closer than RAG↔Cooking | Embedding space has structure |

## Key Concepts Demonstrated

| Concept | Implementation |
|---------|----------------|
| Cosine similarity | `cosine_similarity()` — dot product / magnitudes |
| L2 distance | `l2_distance()` — Euclidean norm of difference |
| Semantic matching | Synonym pairs with zero keyword overlap |
| Vector inspection | Magnitude, min/max, mean, std, density |
| Batch encoding | `model.encode(texts, batch_size=32)` |
| Domain clustering | Cross-domain similarity matrix |

## Extension Prompt

**Challenge:** Modify the MWE to:
1. Compare two different embedding models (e.g., MiniLM vs mpnet)
2. Visualize embeddings in 2D using PCA or t-SNE
3. Measure how embedding quality degrades with truncated text
4. Add a "nearest neighbor" demo that finds the most similar document pair

## Related Reading

- Embedding Models (`notes7/instructional-materials/embedding-models.md`)
- Vector Databases (`notes7/instructional-materials/vector-databases.md`)
