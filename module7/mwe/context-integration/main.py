#!/usr/bin/env python3
"""
Context Integration MWE - RAG prompt construction and grounding patterns.

This script demonstrates how to combine retrieved documents with user
queries into effective prompts for grounded LLM generation:

  1. Basic RAG prompt template
  2. Source attribution with numbered citations
  3. Multi-chunk formatting with separators
  4. Insufficient context handling
  5. Prompt length analysis and context window budgeting
"""

import textwrap
from dataclasses import dataclass, field

# =============================================================================
# Mock Retrieved Chunks
# =============================================================================

@dataclass
class RetrievedChunk:
    """A chunk retrieved from the vector database."""
    content: str
    source: str
    score: float
    chunk_index: int = 0


SAMPLE_CHUNKS = [
    RetrievedChunk(
        content="RAG (Retrieval-Augmented Generation) combines retrieval with "
                "generation to produce grounded responses. It queries external "
                "knowledge bases at inference time to reduce hallucination.",
        source="rag_intro.txt",
        score=0.92,
        chunk_index=0,
    ),
    RetrievedChunk(
        content="The key components of RAG include a document store, an embedding "
                "model, a vector database for similarity search, and a language "
                "model for generation. Each component must be monitored.",
        source="rag_intro.txt",
        score=0.85,
        chunk_index=1,
    ),
    RetrievedChunk(
        content="Document chunking splits large documents into smaller segments "
                "suitable for embedding. Common strategies include fixed-size, "
                "sentence-boundary, and recursive character splitting.",
        source="chunking_guide.txt",
        score=0.78,
        chunk_index=0,
    ),
    RetrievedChunk(
        content="Vector databases store high-dimensional embeddings for fast "
                "similarity search. FAISS is an in-memory library, while Chroma "
                "provides persistent storage with metadata filtering.",
        source="vector_db.txt",
        score=0.71,
        chunk_index=0,
    ),
]

IRRELEVANT_CHUNKS = [
    RetrievedChunk(
        content="Kubernetes orchestrates container deployment across clusters.",
        source="devops_guide.txt",
        score=0.25,
        chunk_index=0,
    ),
    RetrievedChunk(
        content="Python virtual environments isolate project dependencies.",
        source="python_basics.txt",
        score=0.18,
        chunk_index=0,
    ),
]


# =============================================================================
# Prompt Templates
# =============================================================================

class PromptBuilder:
    """Configurable RAG prompt builder with template sections."""

    def __init__(
        self,
        system_instruction: str = "You are a helpful assistant.",
        grounding_instruction: str = (
            "Answer the question using ONLY the provided context. "
            "If the context doesn't contain enough information, say "
            "\"I cannot find this in the provided documents.\""
        ),
        citation_instruction: str = (
            "When citing specific information, reference the source "
            "document in brackets, e.g., [source: filename.txt]."
        ),
        max_context_chars: int = 2000,
    ):
        self.system_instruction = system_instruction
        self.grounding_instruction = grounding_instruction
        self.citation_instruction = citation_instruction
        self.max_context_chars = max_context_chars

    def format_chunks(
        self,
        chunks: list[RetrievedChunk],
        style: str = "numbered",
    ) -> str:
        """Format retrieved chunks into a context section.

        Args:
            chunks: Retrieved chunks to format.
            style: 'numbered', 'labeled', or 'plain'.
        """
        sections = []
        for i, chunk in enumerate(chunks, 1):
            if style == "numbered":
                sections.append(
                    f"[{i}] (Source: {chunk.source}, Score: {chunk.score:.2f})\n"
                    f"{chunk.content}"
                )
            elif style == "labeled":
                sections.append(
                    f"--- Source: {chunk.source} ---\n"
                    f"{chunk.content}"
                )
            else:  # plain
                sections.append(chunk.content)

        return "\n\n".join(sections)

    def build_prompt(
        self,
        question: str,
        chunks: list[RetrievedChunk],
        style: str = "numbered",
        include_citations: bool = True,
    ) -> str:
        """Build a complete RAG prompt.

        Args:
            question: User's question.
            chunks: Retrieved context chunks.
            style: Chunk formatting style.
            include_citations: Whether to add citation instructions.
        """
        if not chunks:
            return self._build_no_context_prompt(question)

        context = self.format_chunks(chunks, style)

        # Truncate context if it exceeds budget
        if len(context) > self.max_context_chars:
            context = context[:self.max_context_chars] + "\n[... context truncated ...]"

        parts = [
            self.system_instruction,
            "",
            self.grounding_instruction,
        ]
        if include_citations:
            parts.append(self.citation_instruction)

        parts.extend([
            "",
            "Context:",
            context,
            "",
            f"Question: {question}",
            "",
            "Answer:",
        ])

        return "\n".join(parts)

    def _build_no_context_prompt(self, question: str) -> str:
        """Build a prompt when no relevant context was retrieved."""
        return "\n".join([
            self.system_instruction,
            "",
            "No relevant context was found in the knowledge base for this question.",
            "Respond by acknowledging that you cannot find the information and suggest "
            "the user try rephrasing their question or checking other sources.",
            "",
            f"Question: {question}",
            "",
            "Answer:",
        ])


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 characters per token for English text."""
    return len(text) // 4


# =============================================================================
# Demonstrations
# =============================================================================

def demo_basic_prompt():
    """Demo 1: Basic RAG prompt template."""
    print("\n" + "=" * 60)
    print("Demo 1: Basic RAG Prompt")
    print("=" * 60)
    print("A minimal prompt with context, question, and grounding instructions.\n")

    builder = PromptBuilder()
    prompt = builder.build_prompt(
        question="What is RAG?",
        chunks=SAMPLE_CHUNKS[:2],
        style="plain",
        include_citations=False,
    )

    print("  Generated prompt:")
    print("  " + "-" * 56)
    for line in prompt.split("\n"):
        print(f"  | {line}")
    print("  " + "-" * 56)
    print(f"\n  Prompt length: {len(prompt)} chars (~{estimate_tokens(prompt)} tokens)")


def demo_source_attribution():
    """Demo 2: Numbered source attribution."""
    print("\n" + "=" * 60)
    print("Demo 2: Source Attribution with Citations")
    print("=" * 60)
    print("Each chunk is numbered with source file and relevance score.\n")

    builder = PromptBuilder()
    prompt = builder.build_prompt(
        question="What are the components of a RAG system?",
        chunks=SAMPLE_CHUNKS[:3],
        style="numbered",
        include_citations=True,
    )

    print("  Generated prompt:")
    print("  " + "-" * 56)
    for line in prompt.split("\n"):
        print(f"  | {line}")
    print("  " + "-" * 56)

    print("\n  Why citations matter:")
    print("  • Users can verify claims against source documents")
    print("  • Helps diagnose retrieval quality issues")
    print("  • Required for Milestone 6 grounding evaluation")


def demo_multi_chunk():
    """Demo 3: Multi-chunk formatting comparison."""
    print("\n" + "=" * 60)
    print("Demo 3: Multi-Chunk Formatting Styles")
    print("=" * 60)
    print("Compare three formatting styles for the same 4 chunks.\n")

    builder = PromptBuilder()
    chunks = SAMPLE_CHUNKS[:4]

    for style in ["plain", "labeled", "numbered"]:
        formatted = builder.format_chunks(chunks, style=style)
        lines = formatted.split("\n")
        print(f"  Style: '{style}' ({len(formatted)} chars)")
        # Show first 3 lines only
        for line in lines[:3]:
            print(f"    {line}")
        print(f"    ... ({len(lines)} total lines)")
        print()

    print("  Trade-offs:")
    print("  • Plain: shortest, no source tracking, hardest to evaluate")
    print("  • Labeled: clear sources, moderate length, easy to read")
    print("  • Numbered: best for citation, LLM can reference [1], [2], etc.")


def demo_insufficient_context():
    """Demo 4: Handling insufficient or irrelevant context."""
    print("\n" + "=" * 60)
    print("Demo 4: Insufficient Context Handling")
    print("=" * 60)

    builder = PromptBuilder()

    # Case 1: No chunks at all
    print("\n  Case 1: No context retrieved")
    prompt_empty = builder.build_prompt(
        question="What is the capital of France?",
        chunks=[],
    )
    print("  " + "-" * 56)
    for line in prompt_empty.split("\n"):
        print(f"  | {line}")
    print("  " + "-" * 56)

    # Case 2: Irrelevant chunks (low scores)
    print("\n  Case 2: Irrelevant context (scores < 0.3)")
    prompt_irrelevant = builder.build_prompt(
        question="What is the capital of France?",
        chunks=IRRELEVANT_CHUNKS,
        style="numbered",
    )
    print("  " + "-" * 56)
    for line in prompt_irrelevant.split("\n"):
        print(f"  | {line}")
    print("  " + "-" * 56)

    print("\n  Key lesson: The grounding instruction tells the LLM to say")
    print("  \"I cannot find this\" rather than hallucinating an answer.")
    print("  In production, also filter chunks by score threshold before prompting.")


def demo_prompt_budget():
    """Demo 5: Prompt length analysis and context window budgeting."""
    print("\n" + "=" * 60)
    print("Demo 5: Context Window Budget Analysis")
    print("=" * 60)

    # Typical context window sizes
    windows = {
        "GPT-3.5 Turbo": 4096,
        "GPT-4o mini": 128000,
        "Llama 3 8B": 8192,
        "Local vLLM (4k)": 4096,
    }

    builder = PromptBuilder()
    question = "What is RAG and how does it work?"

    # Build prompts with increasing chunk counts
    print(f"\n  Question: \"{question}\"")
    print(f"\n  {'Chunks':>7}  {'Prompt Chars':>13}  {'~Tokens':>8}  {'% of 4k':>8}  {'% of 8k':>8}")
    print(f"  {'—'*7}  {'—'*13}  {'—'*8}  {'—'*8}  {'—'*8}")

    for n_chunks in [1, 2, 3, 4]:
        prompt = builder.build_prompt(question, SAMPLE_CHUNKS[:n_chunks])
        chars = len(prompt)
        tokens = estimate_tokens(prompt)
        pct_4k = tokens / 4096 * 100
        pct_8k = tokens / 8192 * 100
        print(f"  {n_chunks:>7}  {chars:>13}  {tokens:>8}  {pct_4k:>7.1f}%  {pct_8k:>7.1f}%")

    print(f"\n  Budget allocation for a 4096-token model:")
    print(f"    System + instructions:  ~100 tokens")
    print(f"    Question:               ~20 tokens")
    print(f"    Reserved for response:  ~500 tokens")
    print(f"    Available for context:  ~3476 tokens (~13,900 chars)")
    print(f"    At ~200 chars/chunk:    ~69 chunks max")

    print(f"\n  ⚠  With small context windows (4k), you must either:")
    print(f"     • Limit top-k retrieval (fewer but more relevant chunks)")
    print(f"     • Use shorter chunks (sacrifice context for precision)")
    print(f"     • Summarize chunks before inserting into prompt")


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("Context Integration MWE — RAG Prompt Construction")
    print("=" * 60)

    demo_basic_prompt()
    demo_source_attribution()
    demo_multi_chunk()
    demo_insufficient_context()
    demo_prompt_budget()

    print("\n" + "=" * 60)
    print("MWE Complete — Context integration patterns demonstrated!")
    print("=" * 60)


if __name__ == "__main__":
    main()
