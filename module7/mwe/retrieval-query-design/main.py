#!/usr/bin/env python3
"""
Retrieval Query Design MWE - Strategies for effective RAG retrieval.

This script demonstrates query design patterns that improve retrieval
quality in RAG pipelines, using simulated embeddings (no ML deps):

  1. Top-k selection sweep — precision/recall trade-off at different k
  2. Metadata filtering — filter by category before similarity search
  3. Query expansion — add synonyms to improve recall
  4. Score thresholding — reject low-confidence results
  5. Hybrid search — combine keyword + semantic scores
"""

import math
import random
from dataclasses import dataclass, field

# =============================================================================
# Simulated Vector Store
# =============================================================================

# Fixed seed for reproducible results
random.seed(42)


@dataclass
class Document:
    """A document in the knowledge base with metadata."""
    doc_id: str
    content: str
    category: str
    date: str
    embedding: list[float] = field(default_factory=list)


def random_embedding(dim: int = 32) -> list[float]:
    """Generate a random unit vector (simulated embedding)."""
    vec = [random.gauss(0, 1) for _ in range(dim)]
    magnitude = math.sqrt(sum(x * x for x in vec))
    return [x / magnitude for x in vec]


def make_similar_embedding(base: list[float], similarity: float) -> list[float]:
    """Create an embedding with approximately the given cosine similarity to base."""
    noise_scale = 1.0 - similarity
    noisy = [b + random.gauss(0, noise_scale) for b in base]
    magnitude = math.sqrt(sum(x * x for x in noisy))
    return [x / magnitude for x in noisy]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    return dot / (mag_a * mag_b) if mag_a * mag_b > 0 else 0.0


# Build a document collection with controlled similarity structure
_BASE_RAG = random_embedding()
_BASE_CHUNK = random_embedding()
_BASE_VECTOR = random_embedding()
_BASE_AGENT = random_embedding()
_BASE_UNRELATED = random_embedding()

DOCUMENTS = [
    # RAG-related documents (relevant to "What is RAG?" queries)
    Document("doc1", "RAG combines retrieval with generation for grounded responses.",
             "architecture", "2024-01", make_similar_embedding(_BASE_RAG, 0.95)),
    Document("doc2", "RAG reduces hallucination by accessing external knowledge.",
             "architecture", "2024-02", make_similar_embedding(_BASE_RAG, 0.88)),
    Document("doc3", "Retrieval-augmented generation uses vector search at inference.",
             "architecture", "2024-01", make_similar_embedding(_BASE_RAG, 0.82)),

    # Chunking-related documents
    Document("doc4", "Document chunking splits text into embedding-suitable segments.",
             "preprocessing", "2024-03", make_similar_embedding(_BASE_CHUNK, 0.90)),
    Document("doc5", "Recursive character splitting respects paragraph boundaries.",
             "preprocessing", "2024-02", make_similar_embedding(_BASE_CHUNK, 0.85)),
    Document("doc6", "Chunk overlap preserves context at segment boundaries.",
             "preprocessing", "2024-03", make_similar_embedding(_BASE_CHUNK, 0.75)),

    # Vector DB documents
    Document("doc7", "FAISS provides exact and approximate nearest neighbor search.",
             "storage", "2024-01", make_similar_embedding(_BASE_VECTOR, 0.92)),
    Document("doc8", "Chroma offers persistent vector storage with metadata filtering.",
             "storage", "2024-02", make_similar_embedding(_BASE_VECTOR, 0.80)),

    # Agent-related documents
    Document("doc9", "Agentic AI systems use ReAct patterns for tool selection.",
             "agents", "2024-03", make_similar_embedding(_BASE_AGENT, 0.88)),
    Document("doc10", "Multi-tool agents coordinate retrieval and summarization.",
              "agents", "2024-02", make_similar_embedding(_BASE_AGENT, 0.82)),

    # Unrelated documents
    Document("doc11", "Kubernetes manages container orchestration at scale.",
             "devops", "2023-12", make_similar_embedding(_BASE_UNRELATED, 0.90)),
    Document("doc12", "Python virtual environments isolate project dependencies.",
             "devops", "2023-11", make_similar_embedding(_BASE_UNRELATED, 0.85)),
]


def similarity_search(
    query_embedding: list[float],
    documents: list[Document],
    k: int = 5,
) -> list[tuple[Document, float]]:
    """Find top-k documents by cosine similarity."""
    scored = [(doc, cosine_similarity(query_embedding, doc.embedding))
              for doc in documents]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:k]


def keyword_search(query: str, documents: list[Document], k: int = 5) -> list[tuple[Document, float]]:
    """Simple keyword matching search (BM25-like scoring)."""
    query_terms = set(query.lower().split())
    scored = []
    for doc in documents:
        doc_terms = set(doc.content.lower().split())
        overlap = query_terms.intersection(doc_terms)
        score = len(overlap) / len(query_terms) if query_terms else 0.0
        scored.append((doc, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:k]


# =============================================================================
# Demonstrations
# =============================================================================

def demo_topk_sweep():
    """Demo 1: Top-k selection sweep."""
    print("\n" + "=" * 60)
    print("Demo 1: Top-k Selection Sweep")
    print("=" * 60)
    print("How does changing k affect retrieval quality?\n")

    query_embedding = make_similar_embedding(_BASE_RAG, 0.9)
    relevant_ids = {"doc1", "doc2", "doc3"}  # Ground truth

    print(f"  Query: (similar to RAG topic)")
    print(f"  Relevant docs: {relevant_ids}")

    print(f"\n  {'k':>3}  {'Retrieved':>10}  {'Precision':>10}  {'Recall':>8}  {'Top docs'}")
    print(f"  {'—'*3}  {'—'*10}  {'—'*10}  {'—'*8}  {'—'*30}")

    for k in [1, 2, 3, 5, 8, 12]:
        results = similarity_search(query_embedding, DOCUMENTS, k=k)
        retrieved_ids = {doc.doc_id for doc, _ in results}

        found = retrieved_ids.intersection(relevant_ids)
        precision = len(found) / k
        recall = len(found) / len(relevant_ids)

        top_ids = [doc.doc_id for doc, _ in results[:5]]
        print(f"  {k:>3}  {len(results):>10}  {precision:>10.2f}  {recall:>8.2f}  {top_ids}")

    print("\n  Key insight: k=3 finds all relevant docs with highest precision.")
    print("  Increasing k beyond that adds noise without improving recall.")


def demo_metadata_filtering():
    """Demo 2: Metadata filtering before search."""
    print("\n" + "=" * 60)
    print("Demo 2: Metadata Filtering")
    print("=" * 60)
    print("Filter by category before similarity search.\n")

    query_embedding = make_similar_embedding(_BASE_CHUNK, 0.85)

    # Without filter
    results_all = similarity_search(query_embedding, DOCUMENTS, k=5)
    print("  Without filter (all categories):")
    for doc, score in results_all:
        print(f"    {doc.doc_id:<6} [{doc.category:<15}] {score:.3f}  {doc.content[:50]}...")

    # With category filter
    filtered_docs = [d for d in DOCUMENTS if d.category == "preprocessing"]
    results_filtered = similarity_search(query_embedding, filtered_docs, k=5)
    print(f"\n  With filter (category='preprocessing'):")
    for doc, score in results_filtered:
        print(f"    {doc.doc_id:<6} [{doc.category:<15}] {score:.3f}  {doc.content[:50]}...")

    # With date filter
    recent_docs = [d for d in DOCUMENTS if d.date >= "2024-02"]
    results_recent = similarity_search(query_embedding, recent_docs, k=5)
    print(f"\n  With filter (date >= 2024-02):")
    for doc, score in results_recent:
        print(f"    {doc.doc_id:<6} [{doc.date}] {score:.3f}  {doc.content[:50]}...")

    print("\n  Key insight: Metadata filters reduce search space and improve")
    print("  precision by excluding irrelevant categories before scoring.")


def demo_query_expansion():
    """Demo 3: Query expansion with synonyms."""
    print("\n" + "=" * 60)
    print("Demo 3: Query Expansion")
    print("=" * 60)
    print("Add related terms to the query to improve recall.\n")

    # Synonym map for query expansion
    synonyms = {
        "rag": ["retrieval-augmented generation", "retrieval augmented"],
        "chunking": ["splitting", "segmentation", "text splitting"],
        "vector": ["embedding", "dense representation"],
        "search": ["retrieval", "lookup", "query"],
    }

    original_query = "How does RAG search work?"
    print(f"  Original query: \"{original_query}\"")

    # Original keyword search
    results_original = keyword_search(original_query, DOCUMENTS, k=5)
    print(f"\n  Original keyword search (top 5):")
    for doc, score in results_original:
        if score > 0:
            print(f"    {doc.doc_id:<6} score={score:.2f}  {doc.content[:55]}...")

    # Expand query
    expanded_terms = set(original_query.lower().split())
    for word in list(expanded_terms):
        if word in synonyms:
            expanded_terms.update(synonyms[word])
    expanded_query = " ".join(expanded_terms)
    print(f"\n  Expanded query: \"{expanded_query}\"")

    results_expanded = keyword_search(expanded_query, DOCUMENTS, k=5)
    print(f"\n  Expanded keyword search (top 5):")
    for doc, score in results_expanded:
        if score > 0:
            print(f"    {doc.doc_id:<6} score={score:.2f}  {doc.content[:55]}...")

    orig_found = sum(1 for _, s in results_original if s > 0)
    expanded_found = sum(1 for _, s in results_expanded if s > 0)
    print(f"\n  Documents with matches: {orig_found} → {expanded_found}")
    print("  Query expansion improves recall by matching synonym variations.")


def demo_score_thresholding():
    """Demo 4: Score thresholding to reject low-confidence results."""
    print("\n" + "=" * 60)
    print("Demo 4: Score Thresholding")
    print("=" * 60)
    print("Reject results below a similarity threshold.\n")

    # Query that's somewhat off-topic
    query_embedding = make_similar_embedding(_BASE_RAG, 0.6)  # Moderate similarity
    results = similarity_search(query_embedding, DOCUMENTS, k=8)

    thresholds = [0.0, 0.3, 0.5, 0.7]

    print(f"  All results (k=8):")
    for doc, score in results:
        print(f"    {doc.doc_id:<6} score={score:.3f}  {doc.content[:50]}...")

    print(f"\n  {'Threshold':>10}  {'Kept':>5}  {'Rejected':>9}  {'Avg Score (kept)':>17}")
    print(f"  {'—'*10}  {'—'*5}  {'—'*9}  {'—'*17}")

    for threshold in thresholds:
        kept = [(doc, s) for doc, s in results if s >= threshold]
        rejected = len(results) - len(kept)
        avg_score = sum(s for _, s in kept) / len(kept) if kept else 0
        print(f"  {threshold:>10.1f}  {len(kept):>5}  {rejected:>9}  {avg_score:>17.3f}")

    print("\n  Key insight: Thresholding at 0.5 removes noise while keeping")
    print("  relevant results. Too aggressive (0.7+) may drop good matches.")
    print("  Tune threshold on your evaluation set, not in production blindly.")


def demo_hybrid_search():
    """Demo 5: Hybrid search combining keyword and semantic scores."""
    print("\n" + "=" * 60)
    print("Demo 5: Hybrid Search (Keyword + Semantic)")
    print("=" * 60)
    print("Combine keyword matching with vector similarity.\n")

    query = "retrieval augmented generation"
    query_embedding = make_similar_embedding(_BASE_RAG, 0.85)

    # Get both scores
    semantic_results = similarity_search(query_embedding, DOCUMENTS, k=len(DOCUMENTS))
    keyword_results = keyword_search(query, DOCUMENTS, k=len(DOCUMENTS))

    # Build score maps
    semantic_scores = {doc.doc_id: score for doc, score in semantic_results}
    keyword_scores = {doc.doc_id: score for doc, score in keyword_results}
    doc_map = {doc.doc_id: doc for doc in DOCUMENTS}

    # Weighted fusion
    alpha = 0.6  # Weight for semantic
    hybrid_scores = {}
    for doc_id in semantic_scores:
        s_score = semantic_scores.get(doc_id, 0.0)
        k_score = keyword_scores.get(doc_id, 0.0)
        hybrid_scores[doc_id] = alpha * s_score + (1 - alpha) * k_score

    # Rank by each method
    ranked_semantic = sorted(semantic_scores.items(), key=lambda x: x[1], reverse=True)
    ranked_keyword = sorted(keyword_scores.items(), key=lambda x: x[1], reverse=True)
    ranked_hybrid = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)

    print(f"  Query: \"{query}\"")
    print(f"  Fusion: {alpha:.0%} semantic + {1-alpha:.0%} keyword\n")

    print(f"  {'Rank':>4}  {'Semantic':>25}  {'Keyword':>25}  {'Hybrid':>25}")
    print(f"  {'—'*4}  {'—'*25}  {'—'*25}  {'—'*25}")

    for i in range(5):
        s_id, s_score = ranked_semantic[i]
        k_id, k_score = ranked_keyword[i]
        h_id, h_score = ranked_hybrid[i]
        print(f"  {i+1:>4}  {s_id:<6} ({s_score:.3f})         "
              f"{k_id:<6} ({k_score:.3f})         "
              f"{h_id:<6} ({h_score:.3f})")

    print(f"\n  Hybrid advantages:")
    print(f"  • Semantic catches paraphrases ('RAG' ↔ 'retrieval-augmented')")
    print(f"  • Keyword catches exact matches missed by embeddings")
    print(f"  • Fusion balances both signals for more robust retrieval")


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("Retrieval Query Design MWE")
    print("=" * 60)

    demo_topk_sweep()
    demo_metadata_filtering()
    demo_query_expansion()
    demo_score_thresholding()
    demo_hybrid_search()

    print("\n" + "=" * 60)
    print("MWE Complete — Retrieval query design patterns demonstrated!")
    print("=" * 60)


if __name__ == "__main__":
    main()
