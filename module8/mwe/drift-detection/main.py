"""
Drift Detection MWE - Population Stability Index (PSI)

Demonstrates detecting data drift by comparing reference (training) 
and current (production) distributions using PSI.

PSI Interpretation:
  < 0.1:  No significant change
  0.1-0.2: Moderate change, investigate
  > 0.2:  Significant drift, action required
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class DriftResult:
    """Result of drift detection for a single feature."""
    feature: str
    psi: float
    drift_detected: bool
    severity: str  # "none", "low", "medium", "high"


def calculate_psi(reference: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
    """
    Calculate Population Stability Index between two distributions.
    
    Args:
        reference: Values from reference (training) distribution
        current: Values from current (production) distribution
        n_bins: Number of bins for discretization
    
    Returns:
        PSI score (float)
    """
    # Create bins from reference distribution
    _, bin_edges = np.histogram(reference, bins=n_bins)
    
    # Count samples in each bin
    ref_counts, _ = np.histogram(reference, bins=bin_edges)
    cur_counts, _ = np.histogram(current, bins=bin_edges)
    
    # Convert to percentages (with smoothing to avoid division by zero)
    ref_pct = (ref_counts + 1) / (len(reference) + n_bins)
    cur_pct = (cur_counts + 1) / (len(current) + n_bins)
    
    # Calculate PSI: sum of (current - reference) * ln(current / reference)
    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    
    return psi


def detect_drift(reference_data: Dict[str, np.ndarray], 
                 current_data: Dict[str, np.ndarray],
                 threshold: float = 0.2) -> List[DriftResult]:
    """
    Detect drift across multiple features.
    
    Args:
        reference_data: Dict mapping feature names to reference arrays
        current_data: Dict mapping feature names to current arrays
        threshold: PSI threshold for drift detection (default 0.2)
    
    Returns:
        List of DriftResult for each feature
    """
    results = []
    
    for feature in reference_data.keys():
        psi = calculate_psi(reference_data[feature], current_data[feature])
        
        # Determine severity
        if psi < 0.1:
            severity = "none"
        elif psi < 0.2:
            severity = "low"
        elif psi < 0.25:
            severity = "medium"
        else:
            severity = "high"
        
        results.append(DriftResult(
            feature=feature,
            psi=psi,
            drift_detected=psi > threshold,
            severity=severity
        ))
    
    return results


if __name__ == "__main__":
    np.random.seed(42)
    
    # Simulate reference (training) data
    print("=== Drift Detection Demo ===\n")
    print("Generating synthetic data...")
    
    reference = {
        "age": np.random.normal(35, 10, 5000),           # Mean 35, std 10
        "income": np.random.normal(50000, 15000, 5000),  # Mean 50k, std 15k
        "score": np.random.uniform(0, 100, 5000),        # Uniform 0-100
    }
    
    # Simulate current (production) data with drift
    current = {
        "age": np.random.normal(42, 12, 5000),           # DRIFTED: older population
        "income": np.random.normal(52000, 15000, 5000),  # Slight shift
        "score": np.random.uniform(0, 100, 5000),        # No drift
    }
    
    print("Comparing reference vs current distributions...\n")
    
    # Detect drift
    results = detect_drift(reference, current)
    
    # Display results
    print(f"{'Feature':<10} {'PSI':>8} {'Severity':<8} {'Drift?'}")
    print("-" * 40)
    
    for r in results:
        status = "YES" if r.drift_detected else "no"
        print(f"{r.feature:<10} {r.psi:>8.4f} {r.severity:<8} {status}")
    
    # Summary
    drifted = [r for r in results if r.drift_detected]
    print(f"\n=== Summary ===")
    print(f"Features analyzed: {len(results)}")
    print(f"Features with drift: {len(drifted)}")
    
    if drifted:
        print(f"Action: Investigate {[r.feature for r in drifted]}")
    else:
        print("Action: Continue monitoring")