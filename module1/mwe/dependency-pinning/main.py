#!/usr/bin/env python3
"""
Dependency Pinning MWE - Demonstrates why version pinning matters for reproducibility.

This script shows how different package versions can produce different results,
illustrating why pinned dependencies are critical in MLOps.
"""

import sys
import hashlib
import json
from datetime import datetime


def get_environment_snapshot():
    """Capture current environment details for reproducibility verification."""
    try:
        import numpy as np
        numpy_version = np.__version__
        # NumPy's random behavior can vary between versions
        np.random.seed(42)
        numpy_sample = np.random.rand(5).tolist()
    except ImportError:
        numpy_version = "NOT INSTALLED"
        numpy_sample = []

    return {
        "python_version": sys.version,
        "numpy_version": numpy_version,
        "numpy_random_sample": numpy_sample,
        "timestamp": datetime.utcnow().isoformat(),
    }


def compute_reproducibility_hash(snapshot):
    """Generate a hash to verify environment reproducibility."""
    # Exclude timestamp from hash (it will always differ)
    hashable = {
        "python_version": snapshot["python_version"],
        "numpy_version": snapshot["numpy_version"],
        "numpy_random_sample": [round(x, 10) for x in snapshot["numpy_random_sample"]],
    }
    content = json.dumps(hashable, sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def demonstrate_version_sensitivity():
    """Show how numerical operations can vary with library versions."""
    try:
        import numpy as np
        
        # This seed + operation combo can produce slightly different 
        # results across NumPy versions due to RNG algorithm changes
        np.random.seed(42)
        data = np.random.randn(1000)
        
        result = {
            "mean": float(np.mean(data)),
            "std": float(np.std(data)),
            "sum": float(np.sum(data)),
        }
        return result
    except ImportError:
        return {"error": "numpy not installed"}


def main():
    """Main demonstration of dependency pinning concepts."""
    print("=" * 60)
    print("DEPENDENCY PINNING - Reproducibility Demonstration")
    print("=" * 60)
    
    # Capture environment
    snapshot = get_environment_snapshot()
    env_hash = compute_reproducibility_hash(snapshot)
    
    print(f"\n📦 Environment Snapshot:")
    print(f"   Python: {snapshot['python_version'].split()[0]}")
    print(f"   NumPy:  {snapshot['numpy_version']}")
    print(f"   Hash:   {env_hash}")
    
    # Demonstrate version sensitivity
    print(f"\n🔢 Numerical Results (seed=42):")
    results = demonstrate_version_sensitivity()
    if "error" not in results:
        print(f"   Mean: {results['mean']:.10f}")
        print(f"   Std:  {results['std']:.10f}")
        print(f"   Sum:  {results['sum']:.10f}")
    else:
        print(f"   ⚠️  {results['error']}")
    
    # Key lesson
    print(f"\n💡 Key Insight:")
    print("   If your teammate gets a DIFFERENT hash, your environments differ!")
    print("   This is why we pin dependencies with exact versions (==).")
    
    print(f"\n✅ Expected hash with pinned requirements: a1b2c3d4e5f6...")
    print("   (Compare this with expected_output/hash.txt)")
    
    print("\n" + "=" * 60)
    return env_hash


if __name__ == "__main__":
    main()