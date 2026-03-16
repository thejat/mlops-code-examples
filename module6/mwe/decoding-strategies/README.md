# Decoding Strategies MWE

Demonstrates different token sampling strategies for LLM inference using a simulated vocabulary. Shows how greedy, temperature, top-k, and top-p sampling affect output diversity and determinism.

## Prerequisites

- Python 3.8+
- numpy
- Any OS (Linux, macOS, Windows)

## Quick Start

```bash
# 1. Clone the repository
git clone <repo-url>
cd module6/mwe/decoding-strategies

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the demo
python main.py
```

## What It Demonstrates

| Strategy | Speed | Deterministic | Best For |
|----------|-------|---------------|----------|
| **Greedy** | Fastest | Yes | Factual Q&A |
| **Temperature** | Fast | No | Creative writing |
| **Top-k** | Medium | No | Brainstorming |
| **Top-p** | Medium | No | General chat |

## Key Concepts

**Temperature** controls distribution sharpness:
- `temp → 0`: approaches argmax (deterministic)
- `temp = 1`: standard softmax
- `temp > 1`: flatter distribution (more random)

**Top-k** limits candidates to the k most likely tokens, preventing very unlikely tokens from being sampled.

**Top-p (nucleus)** dynamically sizes the candidate set: sample from the smallest set of tokens whose cumulative probability exceeds p.

**Caching implication:** Only greedy decoding (`temperature=0`) produces deterministic results suitable for caching.

## Extension Challenges

1. **Multi-step generation:** Chain sampling steps to generate full sentences
2. **Beam search:** Implement beam search as an alternative to sampling
3. **Repetition penalty:** Add a penalty for recently generated tokens
4. **Benchmark latency:** Measure the speed difference between strategies at scale

## Related Materials

- Decoding Strategies Impact Latency (module6 slides, lines 735–747)
- Tokenization and BPE
