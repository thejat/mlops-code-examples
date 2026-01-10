"""
Lab: Module 7 - RAG Pipeline Fundamentals
Time: ~30 minutes
Prerequisites: Read the following instructional materials first:
  - src/module7/instructional-materials/rag-architecture-overview.md
  - src/module7/instructional-materials/document-chunking.md
  - src/module7/instructional-materials/embedding-models.md
  - src/module7/instructional-materials/vector-databases.md

Learning Objectives:
- LO1: Implement document chunking with configurable parameters
- LO2: Generate embeddings and store them in a FAISS vector index
- LO3: Perform similarity search to retrieve relevant chunks
- LO4: Create a grounded prompt that combines query and context
- LO5: Measure latency at each pipeline stage

Milestone 6 Alignment:
- CG3.LO3: Design RAG systems integrating retrievers, vector databases, 
           and model endpoints for knowledge-grounded responses.
"""

# === REQUIREMENTS ===
# Run: pip install sentence-transformers faiss-cpu numpy

import time
from dataclasses import dataclass
from typing import Optional
import numpy as np

# === SAMPLE DATA (provided) ===
# These documents simulate a small knowledge base about MLOps concepts

SAMPLE_DOCUMENTS = [
    {
        "content": """
        Retrieval-Augmented Generation (RAG) is an AI architecture that combines 
        retrieval mechanisms with generative language models. Instead of relying 
        solely on knowledge encoded during training, RAG systems query external 
        knowledge bases at inference time. This approach significantly reduces 
        hallucination by grounding responses in retrieved factual content.
        
        The key components of RAG include: a document store containing chunked 
        text, an embedding model to convert text to vectors, a vector database 
        for efficient similarity search, and a language model for generation.
        """,
        "metadata": {"source": "rag_intro.txt", "topic": "architecture"}
    },
    {
        "content": """
        Document chunking is essential for effective RAG systems. Large documents 
        must be split into smaller segments that fit within embedding model limits 
        and enable precise retrieval. Common strategies include fixed-size chunking 
        with overlap, recursive character splitting that respects semantic boundaries, 
        and semantic chunking based on document structure like headers.
        
        Chunk size typically ranges from 256-1024 characters. Overlap of 10-20% 
        helps preserve context at chunk boundaries. The RecursiveCharacterTextSplitter 
        is popular because it tries paragraph breaks before sentence breaks.
        """,
        "metadata": {"source": "chunking_guide.txt", "topic": "preprocessing"}
    },
    {
        "content": """
        Vector databases store high-dimensional embeddings and enable fast 
        similarity search. FAISS (Facebook AI Similarity Search) is an in-memory 
        library optimized for dense vector search. It supports multiple index types:
        
        - IndexFlatL2: Exact search using L2 distance (slow but accurate)
        - IndexIVF: Approximate search using inverted file structure
        - IndexHNSW: Graph-based approximate nearest neighbor search
        
        For small datasets under 10,000 vectors, IndexFlatL2 provides exact 
        results. Larger datasets benefit from approximate methods that trade 
        slight accuracy loss for dramatic speed improvements.
        """,
        "metadata": {"source": "vector_db.txt", "topic": "storage"}
    },
    {
        "content": """
        Embedding models convert text into dense numerical vectors that capture 
        semantic meaning. The sentence-transformers library provides pre-trained 
        models optimized for semantic similarity tasks. Popular choices include:
        
        - all-MiniLM-L6-v2: Fast, 384 dimensions, good for prototyping
        - all-mpnet-base-v2: Balanced, 768 dimensions, production quality
        - bge-large-en-v1.5: High accuracy, 1024 dimensions, slower
        
        Embeddings enable semantic search where "car" matches "automobile" even 
        without exact word overlap. Cosine similarity is the standard metric.
        """,
        "metadata": {"source": "embeddings.txt", "topic": "models"}
    },
    {
        "content": """
        Grounding is the process of constraining LLM outputs to information 
        present in retrieved context. Without grounding, models may hallucinate 
        plausible but incorrect information. Effective grounding requires:
        
        1. Clear prompt instructions to use only provided context
        2. Source attribution so users can verify claims
        3. Explicit handling of cases where context is insufficient
        
        A well-designed RAG prompt includes the context, the question, and 
        instructions specifying that answers must come from the context only.
        """,
        "metadata": {"source": "grounding.txt", "topic": "generation"}
    }
]

# Test queries for evaluation
TEST_QUERIES = [
    "What is RAG and how does it reduce hallucination?",
    "What chunk sizes are recommended for RAG?",
    "What vector database options are available for similarity search?",
    "How do embedding models enable semantic search?",
    "What is grounding and why is it important?",
]


# === HELPER CLASS (provided) ===

@dataclass
class TimingResult:
    """Track timing for each pipeline stage."""
    stage: str
    duration_ms: float
    item_count: int


# === TODO 1: Implement Document Chunking ===
# Create a function that splits documents into chunks with overlap.
# 
# Requirements:
# - Split text by paragraphs first (double newlines), then by sentences if needed
# - Ensure no chunk exceeds max_chunk_size characters
# - Add overlap between consecutive chunks
# - Preserve metadata for each chunk
#
# Hint: Use simple string splitting, no external libraries needed

def chunk_documents(
    documents: list[dict],
    max_chunk_size: int = 300,
    overlap: int = 50
) -> list[dict]:
    """
    Split documents into overlapping chunks.
    
    Args:
        documents: List of dicts with 'content' and 'metadata' keys
        max_chunk_size: Maximum characters per chunk
        overlap: Number of characters to overlap between chunks
    
    Returns:
        List of chunk dicts with 'content', 'metadata', and 'chunk_index'
    """
    chunks = []
    
    # TODO: Implement chunking logic
    # For each document:
    #   1. Clean the content (strip whitespace)
    #   2. Split into paragraphs (split on "\n\n")
    #   3. For each paragraph, if it exceeds max_chunk_size, split further
    #   4. Create chunks with overlap from previous chunk
    #   5. Add metadata including source and chunk_index
    
    # Your code here
    pass
    
    return chunks


# === TODO 2: Generate Embeddings ===
# Create a function that converts text chunks into numerical vectors.
#
# Requirements:
# - Use sentence-transformers library
# - Handle batching for efficiency
# - Return numpy array of shape (num_chunks, embedding_dim)

def generate_embeddings(texts: list[str], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """
    Generate embeddings for a list of texts.
    
    Args:
        texts: List of text strings to embed
        model_name: Sentence transformer model name
    
    Returns:
        Numpy array of shape (len(texts), embedding_dimension)
    """
    # TODO: Implement embedding generation
    # 1. Import and load the SentenceTransformer model
    # 2. Encode all texts (the model handles batching)
    # 3. Return the embeddings as a numpy array
    
    # Your code here
    pass


# === TODO 3: Create FAISS Index ===
# Create a function that builds a FAISS index from embeddings.
#
# Requirements:
# - Use FAISS IndexFlatL2 for exact search (appropriate for small datasets)
# - Ensure embeddings are float32 (FAISS requirement)
# - Return the index object

def create_faiss_index(embeddings: np.ndarray):
    """
    Create a FAISS index from embeddings.
    
    Args:
        embeddings: Numpy array of shape (num_items, dimension)
    
    Returns:
        FAISS index object
    """
    # TODO: Implement FAISS index creation
    # 1. Import faiss
    # 2. Get the embedding dimension from the array shape
    # 3. Create IndexFlatL2 with that dimension
    # 4. Convert embeddings to float32 if needed
    # 5. Add embeddings to the index
    # 6. Return the index
    
    # Your code here
    pass


# === TODO 4: Implement Similarity Search ===
# Create a function that finds the k most similar chunks to a query.
#
# Requirements:
# - Embed the query using the same model as documents
# - Search the FAISS index for nearest neighbors
# - Return chunks with their similarity scores

def similarity_search(
    query: str,
    index,
    chunks: list[dict],
    embed_fn,
    k: int = 3
) -> list[dict]:
    """
    Find k most similar chunks to the query.
    
    Args:
        query: Search query string
        index: FAISS index
        chunks: List of chunk dictionaries
        embed_fn: Function to generate embeddings
        k: Number of results to return
    
    Returns:
        List of dicts with 'content', 'metadata', 'score', and 'rank'
    """
    # TODO: Implement similarity search
    # 1. Generate embedding for the query
    # 2. Reshape to (1, dimension) for FAISS
    # 3. Ensure float32 dtype
    # 4. Call index.search(query_embedding, k)
    # 5. Build result list with chunk content, metadata, and distance score
    
    # Your code here
    pass


# === TODO 5: Create Grounded Prompt ===
# Create a function that formats retrieved context into a RAG prompt.
#
# Requirements:
# - Include source attribution for each chunk
# - Add clear instructions for grounded generation
# - Handle the case where no relevant context is found

def create_grounded_prompt(query: str, retrieved_chunks: list[dict]) -> str:
    """
    Create a prompt for grounded generation.
    
    Args:
        query: User's question
        retrieved_chunks: List of retrieved chunk dictionaries
    
    Returns:
        Formatted prompt string
    """
    # TODO: Implement prompt creation
    # 1. Format each chunk with its source (from metadata)
    # 2. Combine into a context section
    # 3. Add the question
    # 4. Add instructions to answer ONLY from context
    # 5. Handle empty chunks case
    
    # Your code here
    pass


# === TODO 6: Complete RAG Pipeline with Timing ===
# Create a class that orchestrates the full RAG pipeline with latency tracking.
#
# Requirements:
# - Track timing for each stage (chunking, embedding, indexing, retrieval)
# - Provide a query method that runs the full pipeline
# - Include a timing report method

class RAGPipeline:
    """Complete RAG pipeline with latency tracking."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.timings: list[TimingResult] = []
        self.chunks: list[dict] = []
        self.index = None
        self.model = None
        
    def _time_operation(self, stage: str, func, *args, **kwargs):
        """Execute function and record timing."""
        start = time.time()
        result = func(*args, **kwargs)
        duration_ms = (time.time() - start) * 1000
        item_count = len(result) if isinstance(result, (list, np.ndarray)) else 1
        self.timings.append(TimingResult(stage, duration_ms, item_count))
        return result
    
    def ingest(self, documents: list[dict]) -> int:
        """
        Ingest documents into the pipeline.
        
        Returns:
            Number of chunks created
        """
        # TODO: Implement ingestion with timing
        # 1. Load the embedding model (time as "model_load")
        # 2. Chunk documents (time as "chunking")  
        # 3. Generate embeddings for all chunks (time as "embedding")
        # 4. Create FAISS index (time as "indexing")
        # 5. Store chunks and index as instance attributes
        # 6. Return number of chunks
        
        # Your code here
        pass
    
    def query(self, question: str, k: int = 3) -> dict:
        """
        Execute full RAG query.
        
        Args:
            question: User's question
            k: Number of chunks to retrieve
        
        Returns:
            Dict with prompt, retrieved_chunks, and latency info
        """
        # TODO: Implement query with timing
        # 1. Perform similarity search (time as "retrieval")
        # 2. Create grounded prompt
        # 3. Return results with timing info
        
        # Your code here
        pass
    
    def get_timing_report(self) -> dict:
        """Get summary of all timing measurements."""
        # TODO: Aggregate timings by stage
        # Return dict with stage -> {count, total_ms, avg_ms}
        
        # Your code here
        pass


# === EXPECTED OUTPUT ===
# When you run this lab successfully, you should see output similar to:
#
# === RAG Pipeline Lab ===
# 
# Ingesting 5 documents...
# Created 8-12 chunks (depends on your chunking parameters)
# 
# === Query Results ===
# 
# Query: What is RAG and how does it reduce hallucination?
# Retrieved 3 chunks
# Top result source: rag_intro.txt
# 
# Query: What chunk sizes are recommended for RAG?
# Retrieved 3 chunks
# Top result source: chunking_guide.txt
# 
# ... (more queries)
# 
# === Timing Report ===
# model_load: 1234.5ms (1 call)
# chunking: 1.2ms (5 documents)
# embedding: 456.7ms (10 chunks)
# indexing: 0.5ms (10 chunks)
# retrieval: 2.3ms avg (5 queries)
# 
# ✅ All checks passed!


# === SELF-CHECK ===

def run_self_check():
    """Validate your implementation."""
    print("=" * 60)
    print("Running Self-Check...")
    print("=" * 60)
    
    errors = []
    
    # Test 1: Chunking produces valid output
    print("\n[Test 1] Document Chunking...")
    try:
        chunks = chunk_documents(SAMPLE_DOCUMENTS[:1], max_chunk_size=200, overlap=30)
        assert chunks is not None, "chunk_documents returned None"
        assert len(chunks) > 0, "chunk_documents returned empty list"
        assert all("content" in c for c in chunks), "Chunks missing 'content' key"
        assert all("metadata" in c for c in chunks), "Chunks missing 'metadata' key"
        assert all(len(c["content"]) <= 250 for c in chunks), "Some chunks exceed max size"
        print(f"   ✓ Created {len(chunks)} chunks from 1 document")
    except Exception as e:
        errors.append(f"Chunking failed: {e}")
        print(f"   ✗ Error: {e}")
    
    # Test 2: Embeddings have correct shape
    print("\n[Test 2] Embedding Generation...")
    try:
        test_texts = ["Hello world", "RAG systems are useful"]
        embeddings = generate_embeddings(test_texts)
        assert embeddings is not None, "generate_embeddings returned None"
        assert isinstance(embeddings, np.ndarray), "Embeddings must be numpy array"
        assert embeddings.shape[0] == 2, f"Expected 2 embeddings, got {embeddings.shape[0]}"
        assert embeddings.shape[1] > 100, "Embedding dimension seems too small"
        print(f"   ✓ Generated embeddings with shape {embeddings.shape}")
    except Exception as e:
        errors.append(f"Embedding generation failed: {e}")
        print(f"   ✗ Error: {e}")
    
    # Test 3: FAISS index works
    print("\n[Test 3] FAISS Index Creation...")
    try:
        test_embeddings = np.random.randn(5, 384).astype(np.float32)
        index = create_faiss_index(test_embeddings)
        assert index is not None, "create_faiss_index returned None"
        assert index.ntotal == 5, f"Index should have 5 vectors, has {index.ntotal}"
        print(f"   ✓ Created index with {index.ntotal} vectors")
    except Exception as e:
        errors.append(f"FAISS index creation failed: {e}")
        print(f"   ✗ Error: {e}")
    
    # Test 4: Similarity search returns results
    print("\n[Test 4] Similarity Search...")
    try:
        # Setup: create small index
        test_chunks = [
            {"content": "RAG reduces hallucination", "metadata": {"source": "test1"}},
            {"content": "Vector databases store embeddings", "metadata": {"source": "test2"}},
            {"content": "Chunking splits documents", "metadata": {"source": "test3"}},
        ]
        chunk_texts = [c["content"] for c in test_chunks]
        embs = generate_embeddings(chunk_texts)
        idx = create_faiss_index(embs)
        
        results = similarity_search(
            "What is RAG?", 
            idx, 
            test_chunks, 
            generate_embeddings, 
            k=2
        )
        assert results is not None, "similarity_search returned None"
        assert len(results) == 2, f"Expected 2 results, got {len(results)}"
        assert "content" in results[0], "Results missing 'content'"
        assert "score" in results[0], "Results missing 'score'"
        print(f"   ✓ Retrieved {len(results)} results, top match: '{results[0]['content'][:40]}...'")
    except Exception as e:
        errors.append(f"Similarity search failed: {e}")
        print(f"   ✗ Error: {e}")
    
    # Test 5: Prompt generation
    print("\n[Test 5] Grounded Prompt Creation...")
    try:
        test_retrieved = [
            {"content": "RAG is great", "metadata": {"source": "doc1.txt"}, "score": 0.1},
            {"content": "It reduces errors", "metadata": {"source": "doc2.txt"}, "score": 0.2},
        ]
        prompt = create_grounded_prompt("What is RAG?", test_retrieved)
        assert prompt is not None, "create_grounded_prompt returned None"
        assert "RAG" in prompt, "Prompt should contain query term"
        assert "doc1.txt" in prompt or "source" in prompt.lower(), "Prompt should include source attribution"
        assert len(prompt) > 100, "Prompt seems too short"
        print(f"   ✓ Created prompt with {len(prompt)} characters")
    except Exception as e:
        errors.append(f"Prompt creation failed: {e}")
        print(f"   ✗ Error: {e}")
    
    # Test 6: Full pipeline
    print("\n[Test 6] Complete RAG Pipeline...")
    try:
        pipeline = RAGPipeline()
        num_chunks = pipeline.ingest(SAMPLE_DOCUMENTS)
        assert num_chunks is not None, "ingest returned None"
        assert num_chunks > 0, "ingest returned 0 chunks"
        
        result = pipeline.query("What is RAG?", k=2)
        assert result is not None, "query returned None"
        assert "retrieved_chunks" in result, "Result missing 'retrieved_chunks'"
        assert "prompt" in result, "Result missing 'prompt'"
        
        timing = pipeline.get_timing_report()
        assert timing is not None, "get_timing_report returned None"
        assert len(timing) > 0, "Timing report is empty"
        print(f"   ✓ Pipeline ingested {num_chunks} chunks and processed query")
        print(f"   ✓ Timing report has {len(timing)} stages")
    except Exception as e:
        errors.append(f"Full pipeline failed: {e}")
        print(f"   ✗ Error: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    if errors:
        print(f"❌ {len(errors)} check(s) failed:")
        for err in errors:
            print(f"   - {err}")
        print("\nFix the errors above and run again.")
    else:
        print("✅ All checks passed!")
        print("\nYour RAG pipeline is working correctly.")
        print("Next steps for Milestone 6:")
        print("  1. Add more test queries (you need 10 for the evaluation)")
        print("  2. Implement actual LLM generation (currently mock)")
        print("  3. Analyze retrieval accuracy and grounding quality")
        print("  4. Document your design decisions")
    print("=" * 60)
    
    return len(errors) == 0


# === MAIN ===

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Module 7 Lab: RAG Pipeline Fundamentals")
    print("=" * 60)
    
    # Run the self-check to validate implementation
    success = run_self_check()
    
    if success:
        # Demo the pipeline with all test queries
        print("\n" + "=" * 60)
        print("Running Full Demo...")
        print("=" * 60)
        
        pipeline = RAGPipeline()
        num_chunks = pipeline.ingest(SAMPLE_DOCUMENTS)
        print(f"\nIngested {len(SAMPLE_DOCUMENTS)} documents → {num_chunks} chunks")
        
        print("\n--- Query Results ---")
        for query in TEST_QUERIES:
            result = pipeline.query(query, k=2)
            top_source = result["retrieved_chunks"][0]["metadata"]["source"]
            print(f"\nQ: {query}")
            print(f"   Top source: {top_source}")
            print(f"   Retrieval latency: {result.get('latency', {}).get('retrieval_ms', 'N/A')}ms")
        
        print("\n--- Timing Report ---")
        timing = pipeline.get_timing_report()
        for stage, metrics in timing.items():
            avg = metrics.get("avg_ms", metrics.get("total_ms", 0))
            count = metrics.get("count", 1)
            print(f"{stage}: {avg:.1f}ms avg ({count} calls)")