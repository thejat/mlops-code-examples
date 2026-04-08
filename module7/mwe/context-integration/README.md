# Context Integration MWE

**RAG prompt construction, source attribution, and grounding patterns — pure Python, no external dependencies.**

## Overview

This MWE demonstrates how to combine retrieved documents with user queries into effective prompts for grounded LLM generation:

1. **Basic RAG prompt** — template with context, question, and grounding instructions
2. **Source attribution** — numbered citations with source files and relevance scores
3. **Multi-chunk formatting** — compare plain, labeled, and numbered styles
4. **Insufficient context** — handle empty or irrelevant retrieval results
5. **Context window budget** — token estimation and allocation strategy

## Prerequisites

- Python 3.10+
- No external packages required

## Setup (2 steps)

```bash
# 1. Navigate to the MWE directory
cd module7/mwe/context-integration

# 2. Run the demo
python main.py
```

## Expected Output

| Demo | What It Shows | Key Lesson |
|------|---------------|------------|
| 1. Basic prompt | Minimal RAG template | Structure: system + grounding + context + question |
| 2. Citations | Numbered sources with scores | Enables verification and evaluation |
| 3. Formatting styles | Plain vs labeled vs numbered | Trade-off: length vs traceability |
| 4. No context | Empty and irrelevant retrievals | Grounding instruction prevents hallucination |
| 5. Budget analysis | Token counting across chunk counts | Context window allocation strategy |

## Key Concepts Demonstrated

| Concept | Implementation |
|---------|----------------|
| Prompt template | `PromptBuilder.build_prompt()` — configurable sections |
| Chunk formatting | `format_chunks()` — plain, labeled, numbered styles |
| Grounding instructions | System-level "answer only from context" directive |
| Citation format | `[source: filename.txt]` pattern for LLM to follow |
| No-context handling | `_build_no_context_prompt()` — graceful fallback |
| Token estimation | `estimate_tokens()` — ~4 chars/token heuristic |

## Extension Prompt

**Challenge:** Modify the MWE to:
1. Add a "confidence score" section that tells the LLM to rate its confidence
2. Implement chunk re-ranking by relevance before formatting
3. Add a "conversation history" section for multi-turn RAG
4. Build a prompt that asks the LLM to output structured JSON with citations

## Related Reading

- Context Integration & Prompting (`notes7/instructional-materials/context-integration-prompting.md`)
- RAG Architecture Overview (`notes7/instructional-materials/rag-architecture-overview.md`)
