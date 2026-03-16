#!/usr/bin/env python3
"""
GPU Memory Calculator MWE

Estimates LLM memory requirements and serving capacity:
- Model memory at different precisions (FP32/FP16/INT8/INT4)
- KV cache memory per request
- Maximum batch size for a given GPU
- Tokens/sec estimate from memory bandwidth

Usage:
    python main.py
"""


def model_memory_gb(num_params: int, bytes_per_param: int) -> float:
    """Calculate model weight memory in GB.

    Formula: Memory (GB) = num_params × bytes_per_param / 1e9
    Example: 7B params × 2 bytes (FP16) / 1e9 = 14.0 GB
    """
    return num_params * bytes_per_param / 1e9


def kv_cache_per_request_gb(
    num_layers: int,
    hidden_dim: int,
    context_length: int,
    bytes_per_element: int = 2,
) -> float:
    """Estimate KV cache memory per request in GB.

    Each layer stores K and V tensors of shape (context_length, hidden_dim).
    Formula: 2 × num_layers × context_length × hidden_dim × bytes / 1e9
    The factor of 2 accounts for both K and V matrices.
    """
    # 2 for K and V
    return 2 * num_layers * context_length * hidden_dim * bytes_per_element / 1e9


def max_batch_size(
    gpu_memory_gb: float,
    model_mem_gb: float,
    kv_per_request_gb: float,
    safety_margin: float = 0.85,
) -> int:
    """Calculate maximum concurrent requests given GPU memory constraints.

    Reserves (1 - safety_margin) of GPU memory for framework overhead,
    activations, and other runtime allocations.
    """
    available = (gpu_memory_gb * safety_margin) - model_mem_gb
    if available <= 0 or kv_per_request_gb <= 0:
        return 0
    return int(available / kv_per_request_gb)


def tokens_per_sec(model_mem_gb: float, bandwidth_tb_s: float) -> float:
    """Estimate decode throughput from memory bandwidth.

    During decode, each token generation requires loading all model weights.
    Throughput ≈ bandwidth / model_size.

    Units: (TB/s) / (GB) = (1000 GB/s) / (GB) = tokens/s
    """
    if model_mem_gb <= 0:
        return 0.0
    bandwidth_gb_s = bandwidth_tb_s * 1000  # convert TB/s → GB/s
    return bandwidth_gb_s / model_mem_gb


def main():
    print("=" * 72)
    print("GPU MEMORY CALCULATOR")
    print("=" * 72)

    # --- Section 1: Model memory at different precisions ---
    print("\n--- Model Memory at Different Precisions ---\n")

    precisions = [
        ("FP32", 4),
        ("FP16/BF16", 2),
        ("INT8", 1),
        ("INT4", 0.5),
    ]
    model_sizes = [
        ("7B", 7_000_000_000),
        ("13B", 13_000_000_000),
        ("70B", 70_000_000_000),
    ]

    header = f"  {'Model':<8s}"
    for name, _ in precisions:
        header += f" {name:>10s}"
    print(header)
    print("  " + "-" * (8 + 11 * len(precisions)))

    for model_name, num_params in model_sizes:
        row = f"  {model_name:<8s}"
        for _, bpp in precisions:
            mem = num_params * bpp / 1e9
            row += f" {mem:>8.1f} GB"
        print(row)

    print()
    print("  Formula: Memory (GB) = Parameters × Bytes_per_param / 1e9")

    # --- Section 2: KV cache memory per request ---
    print("\n--- KV Cache Memory per Request ---\n")

    model_configs = [
        ("7B (32 layers, d=4096)", 32, 4096),
        ("13B (40 layers, d=5120)", 40, 5120),
        ("70B (80 layers, d=8192)", 80, 8192),
    ]
    context_lengths = [2048, 4096, 8192]

    header = f"  {'Model':<28s}"
    for ctx in context_lengths:
        header += f" {'ctx=' + str(ctx):>12s}"
    print(header)
    print("  " + "-" * (28 + 13 * len(context_lengths)))

    for config_name, layers, hidden in model_configs:
        row = f"  {config_name:<28s}"
        for ctx in context_lengths:
            kv_gb = kv_cache_per_request_gb(layers, hidden, ctx, bytes_per_element=2)
            row += f" {kv_gb:>9.2f} GB"
        print(row)

    print()
    print("  Formula: KV cache = 2 × layers × context × hidden_dim × 2 bytes / 1e9")

    # --- Section 3: Maximum batch size ---
    print("\n--- Maximum Batch Size per GPU ---\n")

    gpus = [
        ("A100 40GB", 40.0, 1.5),
        ("A100 80GB", 80.0, 2.0),
        ("H100 80GB", 80.0, 3.35),
    ]

    # Use 7B FP16 and 4096 context as reference
    ref_model_mem = model_memory_gb(7_000_000_000, 2)  # 14 GB
    ref_kv = kv_cache_per_request_gb(32, 4096, 4096, 2)

    print(f"  Reference: 7B model FP16 ({ref_model_mem:.0f} GB weights)")
    print(f"             KV cache per request at ctx=4096: {ref_kv:.2f} GB")
    print(f"             Safety margin: 85%")
    print()

    header = f"  {'GPU':<14s} {'Memory':>8s} {'Available':>10s} {'Max Batch':>10s} {'Tok/s':>8s}"
    print(header)
    print("  " + "-" * 54)

    for gpu_name, mem_gb, bw_tbs in gpus:
        avail = mem_gb * 0.85 - ref_model_mem
        mb = max_batch_size(mem_gb, ref_model_mem, ref_kv)
        tps = tokens_per_sec(ref_model_mem, bw_tbs)
        print(f"  {gpu_name:<14s} {mem_gb:>6.0f} GB {avail:>8.1f} GB {mb:>10d} {tps:>7.0f}")

    # --- Section 4: Precision impact on capacity ---
    print("\n--- Precision Impact on Serving Capacity (A100 80GB, 7B model) ---\n")

    gpu_mem = 80.0
    gpu_bw = 2.0

    header = f"  {'Precision':<12s} {'Model Mem':>10s} {'Max Batch':>10s} {'Tok/s':>8s} {'Relative':>10s}"
    print(header)
    print("  " + "-" * 54)

    baseline_tps = None
    for prec_name, bpp in precisions:
        mem = model_memory_gb(7_000_000_000, bpp)
        # KV cache precision matches model precision
        kv = kv_cache_per_request_gb(32, 4096, 4096, bpp)
        mb = max_batch_size(gpu_mem, mem, kv) if kv > 0 else 0
        tps = tokens_per_sec(mem, gpu_bw)
        if baseline_tps is None:
            baseline_tps = tps
        relative = tps / baseline_tps if baseline_tps > 0 else 0
        print(f"  {prec_name:<12s} {mem:>8.1f} GB {mb:>10d} {tps:>7.0f} {relative:>9.1f}x")

    print()
    print("  Key insight: Quantization reduces model size → more bandwidth")
    print("  for token generation AND more memory for concurrent requests.")

    # --- Summary ---
    print("\n" + "=" * 72)
    print("KEY FORMULAS")
    print("=" * 72)
    print("  Model memory (GB) = Parameters × Bytes_per_param / 1e9")
    print("  KV cache (GB)     = 2 × layers × context × hidden × bytes / 1e9")
    print("  Max batch size    = (GPU_mem × 0.85 - model_mem) / kv_per_request")
    print("  Tokens/sec        = Memory_bandwidth (GB/s) / model_mem (GB)")
    print("=" * 72)


if __name__ == "__main__":
    main()
