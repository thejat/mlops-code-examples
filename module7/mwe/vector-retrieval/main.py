#!/usr/bin/env python3
"""
Vector Retrieval MWE - Demonstrates semantic search using embeddings and FAISS.

This script shows how to:
1. Generate embeddings from text using sentence-transformers
2. Store embeddings in a FAISS index
3. Query for semantically similar documents
"""

import time
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss


def create_sample_documents() -> list[str]:
    """Create sample documents for the knowledge base."""
    return [
        "Machine learning models require training data to learn patterns.",
        "Neural networks use backpropagation to update weights during training.",
        "RAG combines retrieval with language generation for better accuracy.",
        "Vector databases store embeddings for fast similarity search.",
        "Docker containers package applications with their dependencies.",
        "Kubernetes orchestrates container deployment at scale.",
        "Python virtual environments isolate project dependencies.",
        "FastAPI provides high-performance REST API development.",
        "MLflow tracks experiments, models, and artifacts.",
        "Data drift occurs when production data differs from training data.",
    ]


def main():
    print("=" * 60)
    print("Vector Retrieval MWE - Semantic Search Demo")
    print("=" * 60)

    # Step 1: Load embedding model
    print("\n[1] Loading embedding model...")
    start = time.time()
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    dimension = model.get_sentence_embedding_dimension()
    print(f"    Model: all-MiniLM-L6-v2 (dimension: {dimension})")
    print(f"    Load time: {time.time() - start:.2f}s")

    # Step 2: Create and embed documents
    print("\n[2] Creating document embeddings...")
    documents = create_sample_documents()
    start = time.time()
    embeddings = model.encode(documents, convert_to_numpy=True)
    embeddings = embeddings.astype(np.float32)  # FAISS requires float32
    print(f"    Documents indexed: {len(documents)}")
    print(f"    Embedding shape: {embeddings.shape}")
    print(f"    Embedding time: {time.time() - start:.3f}s")

    # Step 3: Build FAISS index
    print("\n[3] Building FAISS index...")
    start = time.time()
    index = faiss.IndexFlatL2(dimension)  # L2 (Euclidean) distance
    index.add(embeddings)
    print(f"    Index type: IndexFlatL2 (exact search)")
    print(f"    Vectors in index: {index.ntotal}")
    print(f"    Index build time: {time.time() - start:.4f}s")

    # Step 4: Query the index
    queries = [
        "How do I train a machine learning model?",
        "What is containerization?",
        "How to track ML experiments?",
    ]

    print("\n[4] Running semantic search queries...")
    print("-" * 60)

    for query in queries:
        # Embed the query
        start = time.time()
        query_embedding = model.encode([query], convert_to_numpy=True)
        query_embedding = query_embedding.astype(np.float32)

        # Search for k nearest neighbors
        k = 3
        distances, indices = index.search(query_embedding, k)
        query_time = time.time() - start

        print(f"\nQuery: \"{query}\"")
        print(f"Search time: {query_time * 1000:.2f}ms")
        print("Top matches:")
        for rank, (dist, idx) in enumerate(zip(distances[0], indices[0]), 1):
            print(f"  {rank}. [dist={dist:.4f}] {documents[idx]}")

    print("\n" + "=" * 60)
    print("MWE Complete - Vector retrieval demonstrated successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()