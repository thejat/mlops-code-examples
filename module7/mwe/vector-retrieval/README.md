# Vector Retrieval MWE

**Semantic search using sentence embeddings and FAISS vector index.**

## Overview

This MWE demonstrates the core pattern of vector retrieval for RAG (Retrieval-Augmented Generation) systems:

1. Generate dense vector embeddings from text using sentence-transformers
2. Store embeddings in a FAISS index for efficient similarity search
3. Query the index to find semantically similar documents

## Prerequisites

- Python 3.9+
- Linux, macOS, or Windows
- ~500MB disk space (for model download on first run)

## Setup (3 steps)

```bash
# 1. Clone and navigate
git clone <repository-url>
cd src/module7/learning-activities/mwe/vector-retrieval

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the demo
python main.py
```

## Expected Output

The script will:
1. Load the `all-MiniLM-L6-v2` embedding model (384 dimensions)
2. Create embeddings for 10 sample documents about MLOps topics
3. Build a FAISS index with exact L2 search
4. Run 3 semantic queries and show top-3 matches with distances

Sample output showing semantic matching:
```
Query: "How do I train a machine learning model?"
Top matches:
  1. [dist=0.8234] Machine learning models require training data to learn patterns.
  2. [dist=1.0156] Neural networks use backpropagation to update weights during training.
  3. [dist=1.4521] Data drift occurs when production data differs from training data.
```

## Key Concepts Demonstrated

| Concept | Implementation |
|---------|----------------|
| Embedding Generation | `SentenceTransformer.encode()` |
| Vector Storage | `faiss.IndexFlatL2()` |
| Similarity Search | `index.search(query, k)` |
| Distance Metric | L2 (Euclidean) distance |

## Extension Prompt

**Challenge:** Modify the MWE to:
1. Add metadata filtering (e.g., filter by document category)
2. Compare cosine similarity vs L2 distance results
3. Implement index persistence with `faiss.write_index()` / `faiss.read_index()`
4. Benchmark search latency with 1000+ documents

## Related Reading

- Embedding Models
- Vector Databases
- [FAISS Documentation](https://github.com/facebookresearch/faiss/wiki)