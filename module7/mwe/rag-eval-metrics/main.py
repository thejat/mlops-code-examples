#!/usr/bin/env python3
"""
RAG Evaluation Metrics MWE - Measures retrieval and generation quality.

This script implements the key metrics for evaluating RAG pipelines:
  1. Precision@k  — fraction of top-k retrieved docs that are relevant
  2. Recall@k     — fraction of relevant docs found in top-k
  3. MRR          — mean reciprocal rank of first relevant result
  4. Groundedness — keyword overlap between response and retrieved context
  5. Multi-query evaluation report

All metrics use mock retrieval data — no external dependencies required.
"""

from dataclasses import dataclass

# =============================================================================
# Evaluation Data (mock retrieval results)
# =============================================================================

@dataclass
class EvalQuery:
    """A test query with ground-truth relevant documents and mock results."""
    query: str
    relevant_doc_ids: set[str]       # Ground truth: which docs are relevant
    retrieved_doc_ids: list[str]     # System output: ranked list of retrieved docs
    retrieved_texts: list[str]       # The actual text of retrieved docs
    generated_response: str          # The LLM's response


EVAL_QUERIES = [
    EvalQuery(
        query="What is RAG and how does it reduce hallucination?",
        relevant_doc_ids={"doc1", "doc5"},
        retrieved_doc_ids=["doc1", "doc3", "doc5", "doc7", "doc2"],
        retrieved_texts=[
            "RAG combines retrieval with generation to ground LLM outputs in factual content.",
            "Vector databases enable fast similarity search over embeddings.",
            "RAG reduces hallucination by providing retrieved context to constrain outputs.",
            "Kubernetes orchestrates container deployment at scale.",
            "Document chunking splits text into segments for embedding.",
        ],
        generated_response="RAG (Retrieval-Augmented Generation) combines retrieval with "
                           "generation. It reduces hallucination by grounding LLM outputs "
                           "in retrieved factual content from external knowledge bases.",
    ),
    EvalQuery(
        query="How does document chunking work?",
        relevant_doc_ids={"doc2", "doc4"},
        retrieved_doc_ids=["doc2", "doc6", "doc4", "doc1", "doc8"],
        retrieved_texts=[
            "Document chunking splits text into smaller segments suitable for embedding.",
            "FastAPI provides high-performance REST API development.",
            "Chunking strategies include fixed-size, sentence-boundary, and recursive.",
            "RAG combines retrieval with generation for grounded responses.",
            "MLflow tracks experiments, models, and artifacts.",
        ],
        generated_response="Document chunking splits large documents into smaller segments "
                           "for embedding. Common strategies include fixed-size splitting, "
                           "sentence-boundary splitting, and recursive approaches.",
    ),
    EvalQuery(
        query="What embedding models are used for semantic search?",
        relevant_doc_ids={"doc3", "doc6"},
        retrieved_doc_ids=["doc7", "doc3", "doc1", "doc6", "doc9"],
        retrieved_texts=[
            "Kubernetes handles container orchestration in cloud environments.",
            "Sentence-transformers like all-MiniLM-L6-v2 generate 384-dim embeddings.",
            "RAG architectures use retrieval to augment generation.",
            "Embedding models convert text to dense vectors capturing semantic meaning.",
            "Docker containers isolate application dependencies.",
        ],
        generated_response="Semantic search uses embedding models like sentence-transformers "
                           "(e.g., all-MiniLM-L6-v2) which generate 384-dimensional vectors. "
                           "These dense vectors capture semantic meaning for similarity search.",
    ),
    EvalQuery(
        query="What is grounding in LLM systems?",
        relevant_doc_ids={"doc5", "doc8"},
        retrieved_doc_ids=["doc5", "doc8", "doc3", "doc1", "doc4"],
        retrieved_texts=[
            "Grounding constrains LLM outputs to information in retrieved context.",
            "Effective grounding requires clear prompt instructions and source attribution.",
            "Vector databases store embeddings for similarity search.",
            "RAG combines retrieval with generation.",
            "Chunk overlap preserves context at boundaries.",
        ],
        generated_response="Grounding constrains LLM outputs to information present in "
                           "retrieved context. It requires clear prompt instructions, "
                           "source attribution, and handling of insufficient context.",
    ),
    EvalQuery(
        query="How do vector databases work?",
        relevant_doc_ids={"doc3", "doc9"},
        retrieved_doc_ids=["doc3", "doc2", "doc5", "doc1", "doc9"],
        retrieved_texts=[
            "Vector databases store high-dimensional embeddings for fast similarity search.",
            "Document chunking is essential for effective RAG systems.",
            "Grounding reduces hallucination in LLM outputs.",
            "RAG systems query external knowledge bases at inference time.",
            "FAISS and Chroma are popular vector database options.",
        ],
        generated_response="Vector databases store high-dimensional embeddings and enable "
                           "fast similarity search. Popular options include FAISS for "
                           "in-memory search and Chroma for persistent storage.",
    ),
]


# =============================================================================
# Retrieval Metrics
# =============================================================================

def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """
    Fraction of top-k retrieved documents that are relevant.

    Precision@k = |relevant ∩ top-k| / k

    High precision means few irrelevant results in the top-k.
    """
    top_k = set(retrieved[:k])
    relevant_in_top_k = top_k.intersection(relevant)
    return len(relevant_in_top_k) / k if k > 0 else 0.0


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """
    Fraction of relevant documents found in top-k.

    Recall@k = |relevant ∩ top-k| / |relevant|

    High recall means most relevant documents were retrieved.
    """
    top_k = set(retrieved[:k])
    relevant_in_top_k = top_k.intersection(relevant)
    return len(relevant_in_top_k) / len(relevant) if relevant else 0.0


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    """
    Reciprocal of the rank of the first relevant result.

    RR = 1 / rank_of_first_relevant

    If relevant doc is rank 1 → RR=1.0, rank 2 → RR=0.5, etc.
    """
    for rank, doc_id in enumerate(retrieved, 1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def mean_reciprocal_rank(queries: list[EvalQuery]) -> float:
    """
    Average reciprocal rank across multiple queries.

    MRR = (1/|Q|) × Σ RR(q)
    """
    rr_scores = [reciprocal_rank(q.retrieved_doc_ids, q.relevant_doc_ids)
                 for q in queries]
    return sum(rr_scores) / len(rr_scores) if rr_scores else 0.0


# =============================================================================
# Generation Metrics
# =============================================================================

def groundedness_score(response: str, context_texts: list[str]) -> dict:
    """
    Measure how well the response is grounded in retrieved context.

    Uses keyword overlap as a simple proxy for groundedness:
      - Extract content words (>3 chars) from context and response
      - Compute overlap ratio

    Returns dict with score and diagnostic details.
    """
    def extract_words(text: str) -> set[str]:
        """Extract meaningful content words."""
        words = set()
        for word in text.lower().split():
            # Strip punctuation
            cleaned = ''.join(c for c in word if c.isalnum())
            if len(cleaned) > 3:  # Skip short function words
                words.add(cleaned)
        return words

    context_words = set()
    for text in context_texts:
        context_words.update(extract_words(text))

    response_words = extract_words(response)

    if not response_words:
        return {"score": 0.0, "grounded_words": 0, "total_response_words": 0,
                "ungrounded_words": []}

    grounded = response_words.intersection(context_words)
    ungrounded = response_words - context_words

    return {
        "score": len(grounded) / len(response_words),
        "grounded_words": len(grounded),
        "total_response_words": len(response_words),
        "ungrounded_words": sorted(ungrounded)[:10],  # Show top 10
    }


def faithfulness_check(response: str, context_texts: list[str]) -> dict:
    """
    Check if response contains claims not present in context.

    Simple heuristic: flag sentences in the response that share
    fewer than 30% of content words with any context chunk.
    """
    import re
    response_sentences = re.split(r'(?<=[.!?])\s+', response.strip())
    context_combined = " ".join(context_texts).lower()

    flagged = []
    supported = []

    for sentence in response_sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        words = [w.lower().strip('.,!?;:') for w in sentence.split() if len(w) > 3]
        if not words:
            continue
        matches = sum(1 for w in words if w in context_combined)
        coverage = matches / len(words)
        if coverage < 0.3:
            flagged.append({"sentence": sentence, "coverage": coverage})
        else:
            supported.append({"sentence": sentence, "coverage": coverage})

    return {
        "faithful": len(flagged) == 0,
        "supported_sentences": len(supported),
        "flagged_sentences": len(flagged),
        "flags": flagged,
    }


# =============================================================================
# Demonstrations
# =============================================================================

def demo_precision_recall():
    """Demo 1 & 2: Precision@k and Recall@k."""
    print("\n" + "=" * 60)
    print("Demo 1 & 2: Precision@k and Recall@k")
    print("=" * 60)

    q = EVAL_QUERIES[0]
    print(f"\n  Query: \"{q.query}\"")
    print(f"  Relevant docs: {q.relevant_doc_ids}")
    print(f"  Retrieved (ranked): {q.retrieved_doc_ids}")

    print(f"\n  {'k':>3}  {'Precision@k':>12}  {'Recall@k':>10}  {'Relevant found'}")
    print(f"  {'—'*3}  {'—'*12}  {'—'*10}  {'—'*20}")

    for k in [1, 2, 3, 4, 5]:
        p = precision_at_k(q.retrieved_doc_ids, q.relevant_doc_ids, k)
        r = recall_at_k(q.retrieved_doc_ids, q.relevant_doc_ids, k)
        found = set(q.retrieved_doc_ids[:k]).intersection(q.relevant_doc_ids)
        print(f"  {k:>3}  {p:>12.2f}  {r:>10.2f}  {found or '{}'}")

    print("\n  Key insight: Increasing k improves recall but may lower precision.")
    print("  At k=3, both relevant docs (doc1, doc5) are found → recall=1.0")


def demo_mrr():
    """Demo 3: Mean Reciprocal Rank."""
    print("\n" + "=" * 60)
    print("Demo 3: Mean Reciprocal Rank (MRR)")
    print("=" * 60)
    print("\n  MRR measures how quickly the system returns a relevant result.")

    print(f"\n  {'Query':<50} {'1st Rel Rank':>13} {'RR':>6}")
    print(f"  {'—'*50} {'—'*13} {'—'*6}")

    rr_scores = []
    for q in EVAL_QUERIES:
        rr = reciprocal_rank(q.retrieved_doc_ids, q.relevant_doc_ids)
        rr_scores.append(rr)
        # Find rank of first relevant
        rank = next(
            (i + 1 for i, d in enumerate(q.retrieved_doc_ids)
             if d in q.relevant_doc_ids),
            None
        )
        short_query = q.query[:48]
        print(f"  {short_query:<50} {rank:>13} {rr:>6.2f}")

    mrr = sum(rr_scores) / len(rr_scores)
    print(f"\n  MRR = {mrr:.3f}")
    print(f"  Interpretation: On average, the first relevant doc appears at ~rank {1/mrr:.1f}")


def demo_groundedness():
    """Demo 4: Groundedness scoring."""
    print("\n" + "=" * 60)
    print("Demo 4: Groundedness Scoring")
    print("=" * 60)
    print("\n  Measures keyword overlap between response and retrieved context.")

    for i, q in enumerate(EVAL_QUERIES[:3]):
        g = groundedness_score(q.generated_response, q.retrieved_texts)
        f = faithfulness_check(q.generated_response, q.retrieved_texts)

        print(f"\n  Query {i+1}: \"{q.query}\"")
        print(f"    Response: \"{q.generated_response[:80]}...\"")
        print(f"    Groundedness: {g['score']:.0%} "
              f"({g['grounded_words']}/{g['total_response_words']} content words)")
        if g['ungrounded_words']:
            print(f"    Ungrounded words: {g['ungrounded_words']}")
        print(f"    Faithful: {'✓' if f['faithful'] else '✗'} "
              f"({f['supported_sentences']} supported, {f['flagged_sentences']} flagged)")


def demo_full_report():
    """Demo 5: Multi-query evaluation report."""
    print("\n" + "=" * 60)
    print("Demo 5: Full Evaluation Report")
    print("=" * 60)

    k = 3
    print(f"\n  Evaluation over {len(EVAL_QUERIES)} queries at k={k}")

    # Header
    header = (f"  {'#':>2}  {'P@3':>5}  {'R@3':>5}  {'RR':>5}  "
              f"{'Ground%':>8}  {'Faithful':>9}")
    print(f"\n{header}")
    print(f"  {'—'*2}  {'—'*5}  {'—'*5}  {'—'*5}  {'—'*8}  {'—'*9}")

    all_p = []
    all_r = []
    all_rr = []
    all_g = []
    all_f = []

    for i, q in enumerate(EVAL_QUERIES, 1):
        p = precision_at_k(q.retrieved_doc_ids, q.relevant_doc_ids, k)
        r = recall_at_k(q.retrieved_doc_ids, q.relevant_doc_ids, k)
        rr = reciprocal_rank(q.retrieved_doc_ids, q.relevant_doc_ids)
        g = groundedness_score(q.generated_response, q.retrieved_texts)
        f = faithfulness_check(q.generated_response, q.retrieved_texts)

        all_p.append(p)
        all_r.append(r)
        all_rr.append(rr)
        all_g.append(g['score'])
        all_f.append(1.0 if f['faithful'] else 0.0)

        faithful_str = "✓" if f['faithful'] else "✗"
        print(f"  {i:>2}  {p:>5.2f}  {r:>5.2f}  {rr:>5.2f}  "
              f"{g['score']:>7.0%}  {faithful_str:>9}")

    # Aggregates
    print(f"  {'—'*2}  {'—'*5}  {'—'*5}  {'—'*5}  {'—'*8}  {'—'*9}")
    avg_p = sum(all_p) / len(all_p)
    avg_r = sum(all_r) / len(all_r)
    mrr = sum(all_rr) / len(all_rr)
    avg_g = sum(all_g) / len(all_g)
    avg_f = sum(all_f) / len(all_f)
    print(f"  {'Avg':>2}  {avg_p:>5.2f}  {avg_r:>5.2f}  {mrr:>5.2f}  "
          f"{avg_g:>7.0%}  {avg_f:>8.0%}")

    print(f"\n  Summary:")
    print(f"    Mean Precision@{k}: {avg_p:.2f}")
    print(f"    Mean Recall@{k}:    {avg_r:.2f}")
    print(f"    MRR:                {mrr:.3f}")
    print(f"    Avg Groundedness:   {avg_g:.0%}")
    print(f"    Faithfulness Rate:  {avg_f:.0%}")

    # Diagnosis
    print(f"\n  Diagnostic Guide:")
    if avg_p < 0.5:
        print("    ⚠  Low precision → too many irrelevant docs in top-k")
        print("       Fix: improve embedding quality or add metadata filters")
    if avg_r < 0.5:
        print("    ⚠  Low recall → relevant docs not being retrieved")
        print("       Fix: increase k, improve chunking, or use query expansion")
    if mrr < 0.5:
        print("    ⚠  Low MRR → relevant docs appear too late in results")
        print("       Fix: tune similarity metric or re-rank results")
    if avg_g < 0.7:
        print("    ⚠  Low groundedness → response contains claims not in context")
        print("       Fix: improve prompt instructions for grounding")
    if avg_p >= 0.5 and avg_r >= 0.5 and mrr >= 0.5 and avg_g >= 0.7:
        print("    ✓  All metrics in acceptable range")


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("RAG Evaluation Metrics MWE")
    print("=" * 60)

    demo_precision_recall()
    demo_mrr()
    demo_groundedness()
    demo_full_report()

    print("\n" + "=" * 60)
    print("MWE Complete — RAG evaluation metrics demonstrated!")
    print("=" * 60)


if __name__ == "__main__":
    main()
