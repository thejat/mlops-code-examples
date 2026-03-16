# GPU Memory Calculator MWE

Estimates LLM memory requirements and serving capacity across different GPU types, model sizes, and quantization precisions. No GPU or external libraries needed — pure arithmetic.

## Prerequisites

- Python 3.8+
- Any OS (Linux, macOS, Windows)

## Quick Start

```bash
# 1. Clone the repository
git clone <repo-url>
cd module6/mwe/gpu-memory-calculator

# 2. No dependencies to install (uses only standard library)

# 3. Run the calculator
python main.py
```

## What It Calculates

| Metric | Formula | Example |
|--------|---------|---------|
| **Model memory** | `params × bytes_per_param / 1e9` | 7B × 2 (FP16) / 1e9 = 14 GB |
| **KV cache/request** | `2 × layers × ctx × hidden × bytes / 1e9` | ~2.15 GB for 7B at ctx=4096 |
| **Max batch size** | `(GPU_mem × 0.85 - model_mem) / kv_per_req` | 9 for A100 40GB + 7B FP16 at ctx=4096 |
| **Tokens/sec** | `bandwidth_GB_s / model_mem_GB` | ~143 for A100 80GB + 7B FP16 |

**Note on slides vs computed values:** The slides use a simplified ~1 GB KV cache estimate. This calculator computes the full formula (`2 × 32 layers × 4096 ctx × 4096 hidden × 2 bytes / 1e9 = 2.15 GB`), which gives a lower max batch size (9 vs ~20). The difference shows why precise calculation matters for capacity planning.

## Key Insights

- **Quantization** reduces model size, freeing memory for more concurrent requests AND increasing token throughput
- **KV cache** grows with context length — long-context models need significantly more memory per request
- **Memory bandwidth** (TB/s) is the primary bottleneck during decode, not compute (TFLOPS)
- **Safety margin** (85%) accounts for framework overhead, activations, and runtime allocations

## Extension Challenges

1. **Add multi-GPU support:** Calculate capacity when model is split across 2/4/8 GPUs
2. **Interactive mode:** Accept user inputs for custom model configurations
3. **Cost integration:** Combine with GPU pricing to compute $/token at each precision

## Related Materials

- GPU Memory Components (module6 slides, lines 352–397)
- Memory Bandwidth Bottleneck
- Quantization Fundamentals
