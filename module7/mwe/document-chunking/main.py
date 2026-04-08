#!/usr/bin/env python3
"""
Document Chunking MWE - Compares chunking strategies for RAG pipelines.

This script demonstrates how different chunking strategies affect
chunk count, size distribution, and boundary quality — all using
pure Python with no external dependencies.

Strategies compared:
  1. Fixed-size character splitting
  2. Sentence-boundary splitting
  3. Recursive splitting (paragraph → sentence → character)
  4. Overlap vs no-overlap comparison
  5. Chunk-size analysis and metrics
"""

import re
import textwrap
from dataclasses import dataclass

# =============================================================================
# Sample Document
# =============================================================================

SAMPLE_DOCUMENT = textwrap.dedent("""\
    Retrieval-Augmented Generation (RAG) is an AI architecture that combines
    retrieval mechanisms with generative language models. Instead of relying
    solely on knowledge encoded during training, RAG systems query external
    knowledge bases at inference time. This approach significantly reduces
    hallucination by grounding responses in retrieved factual content.

    The key components of RAG include a document store containing chunked text,
    an embedding model to convert text to vectors, a vector database for
    efficient similarity search, and a language model for generation. Each
    component introduces latency, failure modes, and resource requirements
    that must be monitored and optimized in production.

    Document chunking is the first critical step. Large documents must be split
    into smaller segments that fit within embedding model token limits and enable
    precise retrieval. Poor chunking decisions propagate through the entire
    pipeline: if chunks break mid-sentence or split tables, retrieval accuracy
    suffers and the LLM receives incoherent context.

    Common chunking strategies include fixed-size splitting, sentence-boundary
    splitting, and recursive character splitting. Fixed-size splitting is the
    simplest but can break words and sentences. Sentence-boundary splitting
    respects linguistic units but produces variable-size chunks. Recursive
    splitting tries natural boundaries first before falling back to character
    limits.

    Chunk overlap helps preserve context at boundaries. When consecutive chunks
    share some text, information that spans a boundary is less likely to be lost.
    Typical overlap ranges from 10% to 20% of chunk size. However, overlap
    increases storage requirements and can cause duplicate retrieval results.

    Choosing the right chunk size involves balancing precision and context. Small
    chunks (128-256 characters) enable precise matching but may lack sufficient
    context. Large chunks (512-1024 characters) preserve more context but dilute
    the relevance signal in embeddings. Most RAG systems use 256-512 characters
    as a starting point, then tune based on evaluation metrics.
""")


# =============================================================================
# Chunking Strategies
# =============================================================================

@dataclass
class Chunk:
    """A text chunk with metadata."""
    text: str
    index: int
    strategy: str
    start_char: int
    end_char: int

    @property
    def length(self) -> int:
        return len(self.text)


def fixed_size_chunking(text: str, chunk_size: int = 200, overlap: int = 0) -> list[Chunk]:
    """
    Split text into fixed-size character chunks.

    This is the simplest strategy. It slices the text at exact character
    boundaries, which means words and sentences can be split mid-way.

    Args:
        text: Source text to chunk.
        chunk_size: Maximum characters per chunk.
        overlap: Characters shared between consecutive chunks.

    Returns:
        List of Chunk objects.
    """
    chunks = []
    step = chunk_size - overlap
    pos = 0
    idx = 0

    while pos < len(text):
        end = min(pos + chunk_size, len(text))
        chunk_text = text[pos:end].strip()
        if chunk_text:
            chunks.append(Chunk(
                text=chunk_text,
                index=idx,
                strategy="fixed-size",
                start_char=pos,
                end_char=end,
            ))
            idx += 1
        pos += step

    return chunks


def sentence_boundary_chunking(text: str, max_chunk_size: int = 200) -> list[Chunk]:
    """
    Split text respecting sentence boundaries.

    Sentences are accumulated into chunks until the next sentence would
    exceed the size limit. This produces variable-size chunks but never
    breaks mid-sentence.

    Args:
        text: Source text to chunk.
        max_chunk_size: Soft maximum characters per chunk.

    Returns:
        List of Chunk objects.
    """
    # Split on sentence-ending punctuation followed by whitespace
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())

    chunks = []
    current_sentences: list[str] = []
    current_len = 0
    char_pos = 0
    idx = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        # If adding this sentence exceeds the limit, flush current chunk
        if current_sentences and current_len + len(sentence) + 1 > max_chunk_size:
            chunk_text = " ".join(current_sentences)
            chunks.append(Chunk(
                text=chunk_text,
                index=idx,
                strategy="sentence-boundary",
                start_char=char_pos - len(chunk_text),
                end_char=char_pos,
            ))
            idx += 1
            current_sentences = []
            current_len = 0

        current_sentences.append(sentence)
        current_len += len(sentence) + (1 if current_len > 0 else 0)
        char_pos += len(sentence) + 1

    # Flush remaining
    if current_sentences:
        chunk_text = " ".join(current_sentences)
        chunks.append(Chunk(
            text=chunk_text,
            index=idx,
            strategy="sentence-boundary",
            start_char=char_pos - len(chunk_text),
            end_char=char_pos,
        ))

    return chunks


def recursive_chunking(
    text: str,
    max_chunk_size: int = 200,
    separators: list[str] | None = None,
    overlap: int = 0,
) -> list[Chunk]:
    """
    Recursive splitting: try paragraph → sentence → character boundaries.

    Attempts to split on the most meaningful separator first. Falls back
    to less meaningful separators only when a segment still exceeds the
    size limit.

    Args:
        text: Source text to chunk.
        max_chunk_size: Maximum characters per chunk.
        separators: Ordered list of separators (most → least meaningful).
        overlap: Characters shared between consecutive chunks.

    Returns:
        List of Chunk objects.
    """
    if separators is None:
        separators = ["\n\n", "\n", ". ", " ", ""]

    def _split_recursive(segment: str, sep_idx: int) -> list[str]:
        """Recursively split a segment using progressively finer separators."""
        if len(segment) <= max_chunk_size:
            return [segment]

        if sep_idx >= len(separators):
            # Fallback: hard cut
            return [segment[:max_chunk_size], segment[max_chunk_size:]]

        sep = separators[sep_idx]
        if sep == "":
            # Character-level split
            parts = [segment[i:i + max_chunk_size]
                      for i in range(0, len(segment), max_chunk_size)]
        else:
            parts = segment.split(sep)

        # Merge small parts and recurse large parts
        merged: list[str] = []
        current = ""
        for part in parts:
            candidate = (current + sep + part) if current else part
            if len(candidate) <= max_chunk_size:
                current = candidate
            else:
                if current:
                    merged.append(current)
                # If the part itself is too large, recurse with next separator
                if len(part) > max_chunk_size:
                    merged.extend(_split_recursive(part, sep_idx + 1))
                else:
                    current = part
                    continue
                current = ""
        if current:
            merged.append(current)

        return merged

    raw_chunks = _split_recursive(text.strip(), 0)

    # Apply overlap
    chunks: list[Chunk] = []
    char_pos = 0
    for idx, chunk_text in enumerate(raw_chunks):
        chunk_text = chunk_text.strip()
        if not chunk_text:
            continue

        # For overlap, prepend tail of previous chunk
        if overlap > 0 and idx > 0 and len(raw_chunks[idx - 1]) >= overlap:
            overlap_text = raw_chunks[idx - 1].strip()[-overlap:]
            chunk_text = overlap_text + " " + chunk_text

        chunks.append(Chunk(
            text=chunk_text,
            index=len(chunks),
            strategy="recursive",
            start_char=char_pos,
            end_char=char_pos + len(chunk_text),
        ))
        char_pos += len(chunk_text)

    return chunks


# =============================================================================
# Analysis Helpers
# =============================================================================

def chunk_metrics(chunks: list[Chunk]) -> dict:
    """Compute descriptive statistics for a list of chunks."""
    if not chunks:
        return {"count": 0}
    lengths = [c.length for c in chunks]
    return {
        "count": len(chunks),
        "avg_len": sum(lengths) / len(lengths),
        "min_len": min(lengths),
        "max_len": max(lengths),
        "total_chars": sum(lengths),
    }


def boundary_quality(chunks: list[Chunk]) -> dict:
    """Check how many chunks end mid-word or mid-sentence."""
    broken_words = 0
    broken_sentences = 0
    for c in chunks:
        text = c.text.rstrip()
        if text and text[-1].isalpha() and not text.endswith(('.', '!', '?')):
            broken_sentences += 1
        if text and text[-1].isalnum() and len(text) > 1 and text[-2].isalnum():
            # Heuristic: if last two chars are alphanumeric and no space/punct
            # near the end, likely a broken word
            if not text.endswith(('.', '!', '?', ',', ';', ':')):
                broken_words += 1
    return {
        "total_chunks": len(chunks),
        "broken_sentence_endings": broken_sentences,
        "clean_sentence_endings": len(chunks) - broken_sentences,
    }


# =============================================================================
# Demonstrations
# =============================================================================

def demo_fixed_size():
    """Demo 1: Fixed-size character splitting."""
    print("\n" + "=" * 60)
    print("Demo 1: Fixed-Size Character Splitting")
    print("=" * 60)
    print("Strategy: Cut every 200 characters regardless of content.\n")

    chunks = fixed_size_chunking(SAMPLE_DOCUMENT, chunk_size=200)

    for c in chunks[:3]:
        print(f"  Chunk {c.index} ({c.length} chars):")
        print(f"    \"{c.text[:80]}...\"")
        end_preview = c.text[-30:]
        print(f"    Ends with: \"...{end_preview}\"")
        print()

    metrics = chunk_metrics(chunks)
    quality = boundary_quality(chunks)
    print(f"  Total chunks: {metrics['count']}")
    print(f"  Size range: {metrics['min_len']}–{metrics['max_len']} chars")
    print(f"  Broken sentence endings: {quality['broken_sentence_endings']}/{quality['total_chunks']}")
    print("  ⚠  Fixed-size splitting breaks words and sentences at boundaries.")


def demo_sentence_boundary():
    """Demo 2: Sentence-boundary splitting."""
    print("\n" + "=" * 60)
    print("Demo 2: Sentence-Boundary Splitting")
    print("=" * 60)
    print("Strategy: Accumulate sentences until chunk exceeds 200 chars.\n")

    chunks = sentence_boundary_chunking(SAMPLE_DOCUMENT, max_chunk_size=200)

    for c in chunks[:3]:
        print(f"  Chunk {c.index} ({c.length} chars):")
        preview = c.text[:100].replace('\n', ' ')
        print(f"    \"{preview}...\"")
        print()

    metrics = chunk_metrics(chunks)
    quality = boundary_quality(chunks)
    print(f"  Total chunks: {metrics['count']}")
    print(f"  Size range: {metrics['min_len']}–{metrics['max_len']} chars")
    print(f"  Avg size: {metrics['avg_len']:.0f} chars")
    print(f"  Broken sentence endings: {quality['broken_sentence_endings']}/{quality['total_chunks']}")
    print("  ✓  Sentences are never split, but chunk sizes vary.")


def demo_recursive():
    """Demo 3: Recursive splitting (paragraph → sentence → character)."""
    print("\n" + "=" * 60)
    print("Demo 3: Recursive Splitting")
    print("=" * 60)
    print("Strategy: Try paragraph breaks first, then sentences, then chars.\n")

    chunks = recursive_chunking(SAMPLE_DOCUMENT, max_chunk_size=200)

    for c in chunks[:3]:
        print(f"  Chunk {c.index} ({c.length} chars):")
        preview = c.text[:100].replace('\n', ' ')
        print(f"    \"{preview}...\"")
        print()

    metrics = chunk_metrics(chunks)
    quality = boundary_quality(chunks)
    print(f"  Total chunks: {metrics['count']}")
    print(f"  Size range: {metrics['min_len']}–{metrics['max_len']} chars")
    print(f"  Avg size: {metrics['avg_len']:.0f} chars")
    print(f"  Broken sentence endings: {quality['broken_sentence_endings']}/{quality['total_chunks']}")
    print("  ✓  Respects document structure while staying under size limit.")


def demo_overlap_comparison():
    """Demo 4: Side-by-side overlap vs no-overlap."""
    print("\n" + "=" * 60)
    print("Demo 4: Overlap Comparison")
    print("=" * 60)
    print("Compare 0% vs 20% overlap on fixed-size chunks (size=200).\n")

    no_overlap = fixed_size_chunking(SAMPLE_DOCUMENT, chunk_size=200, overlap=0)
    with_overlap = fixed_size_chunking(SAMPLE_DOCUMENT, chunk_size=200, overlap=40)

    m_no = chunk_metrics(no_overlap)
    m_ov = chunk_metrics(with_overlap)

    print(f"  {'Metric':<25} {'No Overlap':>12} {'20% Overlap':>12}")
    print(f"  {'-'*25} {'-'*12} {'-'*12}")
    print(f"  {'Chunk count':<25} {m_no['count']:>12} {m_ov['count']:>12}")
    print(f"  {'Avg chunk size':<25} {m_no['avg_len']:>12.0f} {m_ov['avg_len']:>12.0f}")
    print(f"  {'Total characters stored':<25} {m_no['total_chars']:>12} {m_ov['total_chars']:>12}")

    overhead = (m_ov['total_chars'] - m_no['total_chars']) / m_no['total_chars'] * 100
    print(f"\n  Storage overhead from overlap: {overhead:.1f}%")
    print("  ⚠  Overlap preserves boundary context but increases storage + possible duplicate retrievals.")

    # Show boundary context preservation
    print("\n  Boundary example (chunks 0→1):")
    if len(no_overlap) >= 2:
        tail = no_overlap[0].text[-40:]
        head = no_overlap[1].text[:40]
        print(f"    No overlap:   \"...{tail}\" | \"{head}...\"")
    if len(with_overlap) >= 2:
        tail = with_overlap[0].text[-40:]
        head = with_overlap[1].text[:40]
        print(f"    With overlap: \"...{tail}\" | \"{head}...\"")
        print("    ↑ Shared text bridges the boundary")


def demo_strategy_comparison():
    """Demo 5: Full strategy comparison with metrics table."""
    print("\n" + "=" * 60)
    print("Demo 5: Strategy Comparison Summary")
    print("=" * 60)

    strategies = {
        "Fixed-size (200)": fixed_size_chunking(SAMPLE_DOCUMENT, 200),
        "Fixed-size (300)": fixed_size_chunking(SAMPLE_DOCUMENT, 300),
        "Sentence (200)": sentence_boundary_chunking(SAMPLE_DOCUMENT, 200),
        "Sentence (300)": sentence_boundary_chunking(SAMPLE_DOCUMENT, 300),
        "Recursive (200)": recursive_chunking(SAMPLE_DOCUMENT, 200),
        "Recursive (300)": recursive_chunking(SAMPLE_DOCUMENT, 300),
    }

    header = f"  {'Strategy':<20} {'Chunks':>7} {'Avg':>7} {'Min':>7} {'Max':>7} {'Clean%':>7}"
    print(f"\n{header}")
    print(f"  {'-'*20} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")

    for name, chunks in strategies.items():
        m = chunk_metrics(chunks)
        q = boundary_quality(chunks)
        clean_pct = q['clean_sentence_endings'] / q['total_chunks'] * 100 if q['total_chunks'] else 0
        print(f"  {name:<20} {m['count']:>7} {m['avg_len']:>7.0f} {m['min_len']:>7} {m['max_len']:>7} {clean_pct:>6.0f}%")

    print("\n  Key takeaways:")
    print("  • Fixed-size: most chunks, worst boundary quality")
    print("  • Sentence: fewest boundary breaks, variable sizes")
    print("  • Recursive: best balance of structure and size control")
    print("  • Larger chunk size → fewer chunks but less retrieval precision")


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("Document Chunking MWE — Comparing Strategies for RAG")
    print("=" * 60)

    demo_fixed_size()
    demo_sentence_boundary()
    demo_recursive()
    demo_overlap_comparison()
    demo_strategy_comparison()

    print("\n" + "=" * 60)
    print("MWE Complete — Document chunking strategies demonstrated!")
    print("=" * 60)


if __name__ == "__main__":
    main()
