# Document Chunking MWE

**Compare chunking strategies for RAG pipelines — pure Python, no external dependencies.**

## Overview

This MWE demonstrates three document chunking strategies and analyzes their trade-offs:

1. **Fixed-size splitting** — cut every N characters regardless of content
2. **Sentence-boundary splitting** — accumulate sentences up to a size limit
3. **Recursive splitting** — try paragraph → sentence → character boundaries in order

It also compares overlap vs no-overlap and produces a summary metrics table.

## Prerequisites

- Python 3.10+
- No external packages required

## Setup (2 steps)

```bash
# 1. Navigate to the MWE directory
cd module7/mwe/document-chunking

# 2. Run the demo
python main.py
```

## Expected Output

The script runs 5 demos:

| Demo | What It Shows | Key Lesson |
|------|---------------|------------|
| 1. Fixed-size | Chunks at exact character boundaries | Breaks words and sentences |
| 2. Sentence-boundary | Accumulates full sentences | Clean boundaries, variable sizes |
| 3. Recursive | Tries paragraph breaks first | Best structure/size balance |
| 4. Overlap comparison | 0% vs 20% overlap side-by-side | Storage overhead vs boundary context |
| 5. Strategy summary | Metrics table for all strategies | Quantitative comparison |

## Key Concepts Demonstrated

| Concept | Implementation |
|---------|----------------|
| Fixed-size chunking | `fixed_size_chunking()` — simple slice with step |
| Sentence splitting | `sentence_boundary_chunking()` — regex on sentence endings |
| Recursive splitting | `recursive_chunking()` — cascading separators |
| Overlap | `overlap` parameter on fixed-size and recursive |
| Quality metrics | `boundary_quality()` — broken vs clean sentence endings |

## Extension Prompt

**Challenge:** Modify the MWE to:
1. Add a semantic chunking strategy that splits on markdown headers (`##`, `###`)
2. Implement a chunk-size histogram using ASCII bar charts
3. Measure how overlap affects duplicate retrieval in a simulated search
4. Add metadata tracking (source document, chunk position) to each chunk

## Related Reading

- Document Chunking Strategies (`notes7/instructional-materials/document-chunking.md`)
- RAG Architecture Overview (`notes7/instructional-materials/rag-architecture-overview.md`)
