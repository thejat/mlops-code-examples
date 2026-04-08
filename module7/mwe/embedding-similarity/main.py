#!/usr/bin/env python3
"""
Embedding Similarity MWE - Explore embedding properties for RAG.

This script demonstrates how embedding models represent text and how
different similarity metrics affect retrieval:

  1. Cosine vs L2 distance — same queries, different metric behavior
  2. Semantic vs keyword matching — 'automobile' ↔ 'car' similarity
  3. Embedding dimension inspection — what vectors look like
  4. Batch encoding performance — 1-at-a-time vs batch
  5. Cross-domain similarity — embedding space structure

Requires: sentence-transformers, numpy
"""

import time
import numpy as np
from sentence_transformers import SentenceTransformer


# =============================================================================
# Helpers
# =============================================================================

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def l2_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute L2 (Euclidean) distance between two vectors."""
    return float(np.linalg.norm(a - b))


def rank_by_cosine(query_emb: np.ndarray, doc_embs: np.ndarray, texts: list[str]) -> list[tuple[str, float]]:
    """Rank documents by cosine similarity to query."""
    scores = [(texts[i], cosine_similarity(query_emb, doc_embs[i]))
              for i in range(len(texts))]
    return sorted(scores, key=lambda x: x[1], reverse=True)


def rank_by_l2(query_emb: np.ndarray, doc_embs: np.ndarray, texts: list[str]) -> list[tuple[str, float]]:
    """Rank documents by L2 distance to query (lower = more similar)."""
    scores = [(texts[i], l2_distance(query_emb, doc_embs[i]))
              for i in range(len(texts))]
    return sorted(scores, key=lambda x: x[1])


# =============================================================================
# Demonstrations
# =============================================================================

def demo_cosine_vs_l2(model: SentenceTransformer):
    """Demo 1: Compare cosine similarity and L2 distance rankings."""
    print("\n" + "=" * 60)
    print("Demo 1: Cosine Similarity vs L2 Distance")
    print("=" * 60)

    documents = [
        "Machine learning models learn patterns from training data.",
        "RAG combines retrieval with language generation.",
        "Docker containers package applications with dependencies.",
        "Neural networks update weights through backpropagation.",
        "Vector databases enable fast similarity search.",
    ]

    query = "How do ML models learn?"
    print(f"\n  Query: \"{query}\"\n")

    # Encode all at once
    all_texts = [query] + documents
    embeddings = model.encode(all_texts, convert_to_numpy=True)
    query_emb = embeddings[0]
    doc_embs = embeddings[1:]

    # Rank by both metrics
    cosine_ranked = rank_by_cosine(query_emb, doc_embs, documents)
    l2_ranked = rank_by_l2(query_emb, doc_embs, documents)

    print(f"  {'Rank':>4}  {'Cosine (higher=better)':<45} {'L2 (lower=better)':<45}")
    print(f"  {'—'*4}  {'—'*45} {'—'*45}")

    for i in range(len(documents)):
        c_text, c_score = cosine_ranked[i]
        l_text, l_score = l2_ranked[i]
        print(f"  {i+1:>4}  {c_text[:35]:<35} {c_score:>6.4f}   "
              f"{l_text[:35]:<35} {l_score:>6.4f}")

    print(f"\n  Key insight: For normalized embeddings, cosine and L2 produce")
    print(f"  the same ranking. Cosine is scale-invariant; L2 depends on magnitude.")


def demo_semantic_vs_keyword(model: SentenceTransformer):
    """Demo 2: Semantic similarity catches what keywords miss."""
    print("\n" + "=" * 60)
    print("Demo 2: Semantic vs Keyword Matching")
    print("=" * 60)

    pairs = [
        ("automobile", "car"),
        ("machine learning", "AI model training"),
        ("happy", "joyful"),
        ("dog", "cat"),
        ("retrieval-augmented generation", "RAG"),
        ("Python programming", "Python snake"),
        ("bank account", "river bank"),
    ]

    print(f"\n  {'Text A':<35} {'Text B':<35} {'Cosine':>7}  {'Keyword':>8}")
    print(f"  {'—'*35} {'—'*35} {'—'*7}  {'—'*8}")

    for text_a, text_b in pairs:
        embs = model.encode([text_a, text_b], convert_to_numpy=True)
        cos_sim = cosine_similarity(embs[0], embs[1])

        # Simple keyword overlap
        words_a = set(text_a.lower().split())
        words_b = set(text_b.lower().split())
        keyword_sim = len(words_a & words_b) / max(len(words_a | words_b), 1)

        flag = "  ←" if abs(cos_sim - keyword_sim) > 0.3 else ""
        print(f"  {text_a:<35} {text_b:<35} {cos_sim:>7.3f}  {keyword_sim:>8.3f}{flag}")

    print(f"\n  ← = large gap between semantic and keyword scores")
    print(f"  Embeddings understand 'automobile'='car' with zero word overlap.")
    print(f"  But also: 'Python programming' vs 'Python snake' are distinguished.")


def demo_dimension_inspection(model: SentenceTransformer):
    """Demo 3: What do embedding vectors actually look like?"""
    print("\n" + "=" * 60)
    print("Demo 3: Embedding Dimension Inspection")
    print("=" * 60)

    texts = [
        "RAG combines retrieval with generation.",
        "Document chunking splits text into segments.",
        "Vector databases store embeddings.",
    ]

    embeddings = model.encode(texts, convert_to_numpy=True)

    print(f"\n  Model: all-MiniLM-L6-v2")
    print(f"  Embedding dimension: {embeddings.shape[1]}")
    print(f"  Data type: {embeddings.dtype}")

    for i, text in enumerate(texts):
        emb = embeddings[i]
        print(f"\n  Text: \"{text}\"")
        print(f"    First 10 dims: [{', '.join(f'{v:.4f}' for v in emb[:10])}]")
        print(f"    Magnitude:      {np.linalg.norm(emb):.4f}")
        print(f"    Min value:      {emb.min():.4f}")
        print(f"    Max value:      {emb.max():.4f}")
        print(f"    Mean:           {emb.mean():.4f}")
        print(f"    Std:            {emb.std():.4f}")
        non_zero = np.count_nonzero(np.abs(emb) > 0.001)
        print(f"    Non-zero dims:  {non_zero}/{len(emb)} "
              f"({non_zero/len(emb)*100:.0f}% dense)")

    print(f"\n  Key insight: Sentence-transformer embeddings are dense —")
    print(f"  nearly all dimensions carry information (unlike sparse TF-IDF).")


def demo_batch_performance(model: SentenceTransformer):
    """Demo 4: Batch encoding vs one-at-a-time performance."""
    print("\n" + "=" * 60)
    print("Demo 4: Batch Encoding Performance")
    print("=" * 60)

    # Generate 50 sample texts
    base_texts = [
        "Machine learning models require training data.",
        "RAG reduces hallucination through retrieval.",
        "Vector databases enable fast similarity search.",
        "Document chunking splits text for embedding.",
        "Agents use tools to complete complex tasks.",
    ]
    texts = [f"{t} (variant {i})" for i in range(10) for t in base_texts]

    print(f"\n  Encoding {len(texts)} texts...\n")

    # One at a time
    start = time.time()
    for text in texts:
        model.encode([text], convert_to_numpy=True)
    time_sequential = time.time() - start

    # All at once (batch)
    start = time.time()
    model.encode(texts, convert_to_numpy=True, batch_size=32)
    time_batch = time.time() - start

    # Small batches
    start = time.time()
    for i in range(0, len(texts), 8):
        model.encode(texts[i:i+8], convert_to_numpy=True)
    time_small_batch = time.time() - start

    speedup = time_sequential / time_batch if time_batch > 0 else 0

    print(f"  {'Method':<25} {'Time (s)':>10} {'Per text (ms)':>14}")
    print(f"  {'—'*25} {'—'*10} {'—'*14}")
    print(f"  {'One-at-a-time':<25} {time_sequential:>10.3f} {time_sequential/len(texts)*1000:>14.1f}")
    print(f"  {'Batch (size=8)':<25} {time_small_batch:>10.3f} {time_small_batch/len(texts)*1000:>14.1f}")
    print(f"  {'Batch (size=32)':<25} {time_batch:>10.3f} {time_batch/len(texts)*1000:>14.1f}")
    print(f"\n  Batch speedup: {speedup:.1f}× faster than one-at-a-time")
    print(f"  Key insight: Always batch encode in production. The model's")
    print(f"  internal parallelism makes batching significantly faster.")


def demo_cross_domain(model: SentenceTransformer):
    """Demo 5: Cross-domain similarity shows embedding space structure."""
    print("\n" + "=" * 60)
    print("Demo 5: Cross-Domain Similarity")
    print("=" * 60)
    print("How similar are texts from different domains?\n")

    domains = {
        "MLOps": [
            "Model versioning tracks changes to ML models.",
            "CI/CD pipelines automate model deployment.",
        ],
        "RAG": [
            "RAG retrieves documents to ground LLM responses.",
            "Vector search finds semantically similar chunks.",
        ],
        "Cooking": [
            "Preheat the oven to 350 degrees Fahrenheit.",
            "Whisk the eggs until fluffy before adding flour.",
        ],
        "Sports": [
            "The goalkeeper made a spectacular diving save.",
            "Marathon runners train for months before the race.",
        ],
    }

    # Encode all texts
    all_texts = []
    all_labels = []
    for domain, texts in domains.items():
        for text in texts:
            all_texts.append(text)
            all_labels.append(domain)

    embeddings = model.encode(all_texts, convert_to_numpy=True)

    # Compute pairwise domain similarities
    domain_names = list(domains.keys())
    print(f"  Average cosine similarity between domains:\n")
    print(f"  {'':>10}", end="")
    for d in domain_names:
        print(f"  {d:>10}", end="")
    print()
    print(f"  {'—'*10}", end="")
    for _ in domain_names:
        print(f"  {'—'*10}", end="")
    print()

    for i, d1 in enumerate(domain_names):
        print(f"  {d1:>10}", end="")
        for j, d2 in enumerate(domain_names):
            # Get indices for each domain
            idx1 = [k for k, l in enumerate(all_labels) if l == d1]
            idx2 = [k for k, l in enumerate(all_labels) if l == d2]

            sims = []
            for a in idx1:
                for b in idx2:
                    if a != b:
                        sims.append(cosine_similarity(embeddings[a], embeddings[b]))
            avg_sim = sum(sims) / len(sims) if sims else 1.0
            print(f"  {avg_sim:>10.3f}", end="")
        print()

    print(f"\n  Key insight: Related domains (MLOps ↔ RAG) are closer in")
    print(f"  embedding space than unrelated domains (RAG ↔ Cooking).")
    print(f"  This structure is what makes semantic search work.")


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("Embedding Similarity MWE — Exploring Vector Representations")
    print("=" * 60)

    # Load model once for all demos
    print("\nLoading model: all-MiniLM-L6-v2...")
    start = time.time()
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    print(f"Model loaded in {time.time() - start:.2f}s "
          f"(dimension: {model.get_sentence_embedding_dimension()})")

    demo_cosine_vs_l2(model)
    demo_semantic_vs_keyword(model)
    demo_dimension_inspection(model)
    demo_batch_performance(model)
    demo_cross_domain(model)

    print("\n" + "=" * 60)
    print("MWE Complete — Embedding similarity patterns demonstrated!")
    print("=" * 60)


if __name__ == "__main__":
    main()
