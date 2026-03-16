#!/usr/bin/env python3
"""
Inference Cost Calculator MWE

Estimates LLM serving costs with explicit dimensional analysis:
- Master cost formula with unit tracking
- API pricing vs self-hosted cost comparison
- Break-even analysis at different token volumes
- Optimization impact (quantization, batching, caching)
- 10x cost reduction case study walkthrough

Usage:
    python main.py
"""


def cost_per_request(
    output_tokens: int,
    time_per_token_sec: float,
    gpu_cost_per_hour: float,
    batch_size: int = 1,
    cache_hit_rate: float = 0.0,
) -> float:
    """Calculate cost per request in dollars.

    Dimensional analysis:
        tokens × (sec/token) × ($/hour) × (hour/3600sec) / batch_size × (1 - cache_hit_rate)
        = tokens × sec/token × $/sec / batch_size × (1 - hit_rate)
        = $ per request

    Args:
        output_tokens: Number of tokens generated per request
        time_per_token_sec: Time to generate one token (seconds)
        gpu_cost_per_hour: GPU instance cost ($/hour)
        batch_size: Number of requests processed per weight load
        cache_hit_rate: Fraction of requests served from cache (0.0–1.0)

    Returns:
        Cost in dollars per request
    """
    # $/sec = $/hour ÷ 3600 sec/hour
    gpu_cost_per_sec = gpu_cost_per_hour / 3600

    # GPU time for this request's tokens (seconds)
    gpu_time_sec = output_tokens * time_per_token_sec

    # Cost before batching and caching
    raw_cost = gpu_time_sec * gpu_cost_per_sec

    # Batching: amortize weight-loading cost across batch
    batched_cost = raw_cost / batch_size

    # Caching: fraction of requests skip GPU entirely
    effective_cost = batched_cost * (1 - cache_hit_rate)

    return effective_cost


def cost_per_million_tokens(
    time_per_token_sec: float,
    gpu_cost_per_hour: float,
    batch_size: int = 1,
    cache_hit_rate: float = 0.0,
) -> float:
    """Calculate cost per million output tokens.

    Dimensional analysis:
        1,000,000 tokens × (sec/token) × ($/hour ÷ 3600) / batch_size × (1 - hit_rate)
        = $ per million tokens
    """
    return cost_per_request(
        output_tokens=1_000_000,
        time_per_token_sec=time_per_token_sec,
        gpu_cost_per_hour=gpu_cost_per_hour,
        batch_size=batch_size,
        cache_hit_rate=cache_hit_rate,
    )


def monthly_cost(
    tokens_per_month: int,
    time_per_token_sec: float,
    gpu_cost_per_hour: float,
    batch_size: int = 1,
    cache_hit_rate: float = 0.0,
) -> float:
    """Calculate monthly serving cost.

    Dimensional analysis:
        tokens/month × sec/token × $/hour / 3600 / batch_size × (1 - hit_rate)
        = $/month
    """
    return cost_per_request(
        output_tokens=tokens_per_month,
        time_per_token_sec=time_per_token_sec,
        gpu_cost_per_hour=gpu_cost_per_hour,
        batch_size=batch_size,
        cache_hit_rate=cache_hit_rate,
    )


def main():
    print("=" * 72)
    print("INFERENCE COST CALCULATOR")
    print("=" * 72)

    # --- Section 1: The Master Formula ---
    print("\n--- The Master Cost Formula ---\n")
    print("  Cost per request ($) =")
    print("    output_tokens × time_per_token_sec × (gpu_cost_per_hour / 3600)")
    print("    ÷ batch_size")
    print("    × (1 - cache_hit_rate)")
    print()
    print("  Dimensional analysis:")
    print("    tokens × sec/token × $/hour × hour/3600sec ÷ batch × (1-hit_rate)")
    print("    = $ per request")

    # --- Section 2: Baseline calculation ---
    print("\n--- Baseline: Unoptimized Serving ---\n")

    # 7B model FP16 on A100 80GB
    baseline = {
        "output_tokens": 200,
        "time_per_token_sec": 0.01,  # 10ms = 100 tok/s
        "gpu_cost_per_hour": 4.50,
        "batch_size": 1,
        "cache_hit_rate": 0.0,
    }

    baseline_cost = cost_per_request(**baseline)
    baseline_per_m = cost_per_million_tokens(
        baseline["time_per_token_sec"],
        baseline["gpu_cost_per_hour"],
        baseline["batch_size"],
        baseline["cache_hit_rate"],
    )

    print(f"  Configuration:")
    print(f"    Model: 7B FP16 on A100 80GB")
    print(f"    Throughput: {1/baseline['time_per_token_sec']:.0f} tokens/sec")
    print(f"    GPU cost: ${baseline['gpu_cost_per_hour']:.2f}/hour")
    print(f"    Batch size: {baseline['batch_size']} (no batching)")
    print(f"    Cache hit rate: {baseline['cache_hit_rate']:.0%}")
    print()
    print(f"  Cost per request (200 tokens): ${baseline_cost:.4f}")
    print(f"  Cost per million tokens:       ${baseline_per_m:.2f}")

    # --- Section 3: Optimization impact ---
    print("\n--- Optimization Impact (each applied cumulatively) ---\n")

    optimizations = [
        ("Baseline (unoptimized)", 0.01, 4.50, 1, 0.0),
        ("+ FP8 Quantization (2x faster)", 0.005, 4.50, 1, 0.0),
        ("+ Continuous Batching (8x)", 0.005, 4.50, 8, 0.0),
        ("+ Response Caching (25%)", 0.005, 4.50, 8, 0.25),
        ("+ Reserved Instance (-33%)", 0.005, 3.00, 8, 0.25),
    ]

    header = f"  {'Optimization':<38s} {'$/M tokens':>11s} {'Relative':>9s} {'Savings':>8s}"
    print(header)
    print("  " + "-" * 68)

    baseline_cpm = None
    for name, tpt, gpu_cost, batch, cache in optimizations:
        cpm = cost_per_million_tokens(tpt, gpu_cost, batch, cache)
        if baseline_cpm is None:
            baseline_cpm = cpm
        relative = cpm / baseline_cpm
        savings = (1 - relative) * 100
        print(f"  {name:<38s} ${cpm:>9.2f} {relative:>8.2f}x {savings:>6.0f}%")

    # --- Section 4: API vs Self-Hosted ---
    print("\n--- API vs Self-Hosted Comparison ---\n")

    api_prices = [
        ("GPT-3.5 Turbo (output)", 1.50),
        ("GPT-4o mini (output)", 0.60),
        ("Claude 3.5 Haiku (output)", 1.00),
    ]

    # Self-hosted: optimized config
    self_hosted_cpm = cost_per_million_tokens(0.005, 4.50, 8, 0.25)

    print(f"  Self-hosted (7B, FP8, batched, cached): ${self_hosted_cpm:.2f}/M tokens")
    print()

    header = f"  {'API Provider':<30s} {'$/M tokens':>11s} {'vs Self-Host':>13s}"
    print(header)
    print("  " + "-" * 56)

    for name, price in api_prices:
        ratio = price / self_hosted_cpm if self_hosted_cpm > 0 else 0
        comparison = f"{ratio:.1f}x more" if ratio > 1 else f"{1/ratio:.1f}x less"
        print(f"  {name:<30s} ${price:>9.2f} {comparison:>13s}")

    # --- Section 5: Break-even analysis ---
    print("\n--- Break-Even Analysis: When to Self-Host ---\n")

    volumes = [1_000_000, 10_000_000, 100_000_000, 1_000_000_000]
    gpu_fixed_monthly = 4.50 * 730  # $/month for one A100 running 24/7
    ops_overhead = 2000  # $/month for DevOps overhead

    # Compare at two API price points to show crossover depends on API tier
    api_tiers = [
        ("GPT-4o mini ($0.60/M)", 0.60),
        ("GPT-4 class ($15/M)", 15.00),
    ]

    for tier_name, api_rate in api_tiers:
        print(f"  API tier: {tier_name}")
        header = (f"    {'Monthly Tokens':>15s} {'API Cost':>10s} "
                  f"{'Self-Host':>10s} {'Ops':>8s} {'Total SH':>10s} {'Winner':>10s}")
        print(header)
        print("    " + "-" * 67)

        for vol in volumes:
            api_cost = vol / 1_000_000 * api_rate
            sh_gpu = monthly_cost(vol, 0.005, 4.50, 8, 0.25)
            # Need at least 1 GPU running 24/7
            sh_gpu = max(sh_gpu, gpu_fixed_monthly)
            sh_total = sh_gpu + ops_overhead
            winner = "API" if api_cost < sh_total else "Self-host"
            vol_str = f"{vol/1e6:.0f}M"
            print(f"    {vol_str:>15s} ${api_cost:>8,.0f} "
                  f"${sh_gpu:>8,.0f} ${ops_overhead:>6,d} ${sh_total:>8,.0f} {winner:>10s}")
        print()

    print("  Key insight: Break-even depends heavily on API pricing tier.")
    print("  Cheap APIs (GPT-4o mini) beat self-hosting at most volumes.")
    print("  Expensive APIs (GPT-4 class) lose to self-hosting above ~350M tokens/month.")
    print("  Self-hosting also includes $2,000/month ops overhead (DevOps labor).")

    # --- Section 6: 10x Cost Reduction Walkthrough ---
    print("\n--- Case Study: 10x Cost Reduction ---\n")

    # Month 3: 4 GPUs for 70B, caching reduces effective load by 30%
    # Daily GPU: 4 × $3/hr × 24hr × 0.7 effective = $201.60, plus Redis $500/30 = $16.67
    month3_daily = 3.0 * 4 * 24 * 0.7 + 500 / 30

    # Month 4: Cascade routes 40% to 7B (1 GPU), 60% stays on 70B
    # 70B only needs 60% of previous capacity: 4 × 0.6 ≈ 2.4 → 3 GPUs (round up)
    # But with caching already applied, the 70B load is further reduced
    # Total: 1 GPU (7B) + 2 GPUs (70B, reduced load) = 3 GPUs instead of 4
    month4_daily = (3.0 * 1 + 3.0 * 2) * 24 * 0.7 + 500 / 30

    steps = [
        ("Month 0: Baseline (API)",
         "GPT-4 API, 1B tokens/day",
         15000, "day"),
        ("Month 1: Self-host with vLLM",
         "LLaMA 70B on 8× A100, $3/hr/GPU",
         3.0 * 8 * 24, "day"),
        ("Month 2: FP8 Quantization",
         "Reduce to 4× A100 (same throughput)",
         3.0 * 4 * 24, "day"),
        ("Month 3: Add Caching (30% hit rate)",
         "Redis layer, reduce effective tokens",
         month3_daily, "day"),
        ("Month 4: Model Cascade",
         "Route 40% to 7B (1 GPU), 70B on 2 GPUs",
         month4_daily, "day"),
    ]

    print(f"  {'Step':<38s} {'Daily Cost':>11s} {'Monthly':>10s} {'Reduction':>10s}")
    print("  " + "-" * 72)

    baseline_daily = None
    for name, desc, daily, _ in steps:
        monthly_val = daily * 30
        if baseline_daily is None:
            baseline_daily = daily
        reduction = baseline_daily / daily if daily > 0 else 0
        print(f"  {name:<38s} ${daily:>9,.0f} ${monthly_val:>8,.0f} {reduction:>9.1f}x")

    print()
    print("  Total reduction: ~{:.0f}x from baseline".format(
        baseline_daily / steps[-1][2] if steps[-1][2] > 0 else 0))
    print("  Key: Each optimization compounds on previous ones.")

    # --- Summary ---
    print("\n" + "=" * 72)
    print("KEY TAKEAWAYS")
    print("=" * 72)
    print()
    print("  Every variable in the cost equation is an optimization lever:")
    print()
    print("  Variable           Impact      How to Optimize")
    print("  " + "-" * 60)
    print("  output_tokens      Linear      Limit max_tokens, shorter prompts")
    print("  time_per_token     Linear      Quantization, faster GPU")
    print("  gpu_cost_per_hour  Linear      Reserved instances, spot pricing")
    print("  batch_size         Inverse     Continuous batching")
    print("  cache_hit_rate     Inverse     Better caching, semantic cache")
    print()
    print("  Decision guide (for GPT-4 class API pricing):")
    print("  - < 350M tokens/month → Use APIs (self-hosting overhead exceeds savings)")
    print("  - 350M–1B tokens/month → Evaluate self-hosting (break-even zone)")
    print("  - > 1B tokens/month   → Self-host, optimize aggressively")
    print("  Note: With cheap APIs (GPT-4o mini), thresholds shift much higher.")
    print("=" * 72)


if __name__ == "__main__":
    main()
