# Inference Cost Calculator MWE

Estimates LLM serving costs with explicit unit tracking. Compares API pricing vs self-hosted, analyzes break-even points, and walks through a 10x cost reduction case study.

## Prerequisites

- Python 3.8+
- Any OS (Linux, macOS, Windows)

## Quick Start

```bash
# 1. Clone the repository
git clone <repo-url>
cd module6/mwe/inference-cost-calculator

# 2. No dependencies to install (uses only standard library)

# 3. Run the calculator
python main.py
```

## What It Calculates

### The Master Cost Formula

```
Cost per request ($) = output_tokens × time_per_token_sec × (gpu_cost_per_hour / 3600)
                       ÷ batch_size
                       × (1 - cache_hit_rate)
```

**Dimensional analysis:**
```
tokens × sec/token × $/hour × hour/3600sec ÷ batch × (1 - hit_rate) = $
```

### Optimization Levers

| Variable | Impact | Optimization |
|----------|--------|-------------|
| `output_tokens` | Linear | Limit max_tokens |
| `time_per_token` | Linear | Quantization, faster GPU |
| `gpu_cost_per_hour` | Linear | Reserved/spot instances |
| `batch_size` | Inverse (÷) | Continuous batching |
| `cache_hit_rate` | Inverse (1-x) | Better caching strategy |

### Sections

1. **Baseline cost** — Unoptimized single-request serving
2. **Cumulative optimizations** — Quantization → batching → caching → reserved instances
3. **API vs self-hosted** — Price comparison across providers
4. **Break-even analysis** — When self-hosting becomes cost-effective
5. **10x cost reduction** — Step-by-step case study

## Key Insights

- Optimizations **compound**: FP8 (2x) × batching (8x) × caching (1.3x) = ~21x total reduction
- Self-hosting break-even depends heavily on **which API tier** you're comparing against:
  - vs GPT-4 class ($15/M tokens): break-even around 350M tokens/month
  - vs GPT-4o mini ($0.60/M tokens): self-hosting rarely wins on cost alone
- Hidden costs of self-hosting include DevOps labor ($2K+/month) and operational overhead
- Start with APIs, migrate to self-hosting when economics justify the engineering investment

## Extension Challenges

1. **Interactive calculator:** Accept user inputs for custom configurations
2. **Merge with gpu-memory-calculator:** Combine memory and cost analysis
3. **Multi-GPU cost modeling:** Account for tensor parallelism overhead
4. **TCO comparison:** Add training costs, model development time

## Related Materials

- LLM Inference Economics (module6 slides, lines 1914–2073)
- The Inference Cost Equation
- Build vs Buy Decision Matrix
