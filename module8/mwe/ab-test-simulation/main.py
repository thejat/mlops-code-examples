"""
A/B Test Simulation MWE - Complete Experimental Workflow

Demonstrates A/B testing methodology for ML model evaluation:
- Sample size calculation
- User assignment via consistent hashing
- Outcome simulation with realistic patterns
- Statistical analysis (z-test, confidence intervals)
- Decision recommendation

Run: python main.py
"""

import numpy as np
import hashlib
from dataclasses import dataclass
from typing import Dict, List, Tuple
from datetime import datetime, timedelta


@dataclass
class ExperimentResult:
    """Complete results of A/B test analysis."""
    n_control: int
    n_treatment: int
    control_rate: float
    treatment_rate: float
    relative_lift: float
    p_value: float
    confidence_interval: Tuple[float, float]
    is_significant: bool
    recommendation: str


def calculate_sample_size(
    baseline_rate: float,
    minimum_detectable_effect: float,
    alpha: float = 0.05,
    power: float = 0.80
) -> int:
    """
    Calculate required sample size per variant for a proportion test.
    
    Args:
        baseline_rate: Current conversion rate (e.g., 0.10 for 10%)
        minimum_detectable_effect: Relative lift to detect (e.g., 0.05 for 5%)
        alpha: Significance level (Type I error rate)
        power: Statistical power (1 - Type II error rate)
    
    Returns:
        Required sample size per variant
    """
    from scipy import stats
    
    p1 = baseline_rate
    p2 = baseline_rate * (1 + minimum_detectable_effect)
    
    # Effect size (Cohen's h for proportions)
    effect_size = 2 * (np.arcsin(np.sqrt(p2)) - np.arcsin(np.sqrt(p1)))
    
    # Z-scores
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_power = stats.norm.ppf(power)
    
    # Sample size per group
    n = int(np.ceil(2 * ((z_alpha + z_power) / effect_size) ** 2))
    
    return n


def assign_variant(user_id: str, experiment_name: str, control_weight: float = 0.5) -> str:
    """
    Deterministically assign user to variant using consistent hashing.
    Same user always gets same variant for given experiment.
    
    Args:
        user_id: Unique user identifier
        experiment_name: Experiment identifier
        control_weight: Proportion of users in control group
    
    Returns:
        "control" or "treatment"
    """
    hash_input = f"{user_id}:{experiment_name}"
    hash_value = hashlib.md5(hash_input.encode()).hexdigest()
    hash_float = int(hash_value[:8], 16) / (16 ** 8)
    
    return "control" if hash_float < control_weight else "treatment"


def simulate_experiment(
    n_users: int,
    experiment_name: str,
    control_rate: float,
    treatment_effect: float,
    seed: int = 42
) -> Dict[str, List[bool]]:
    """
    Simulate A/B test with synthetic users and conversions.
    
    Args:
        n_users: Total number of users to simulate
        experiment_name: Experiment identifier for hashing
        control_rate: Base conversion rate for control
        treatment_effect: Relative lift for treatment (e.g., 0.05 for 5%)
        seed: Random seed for reproducibility
    
    Returns:
        Dict with "control" and "treatment" conversion lists
    """
    np.random.seed(seed)
    
    results = {"control": [], "treatment": []}
    
    for i in range(n_users):
        user_id = f"user_{i:06d}"
        variant = assign_variant(user_id, experiment_name)
        
        if variant == "control":
            converted = np.random.random() < control_rate
        else:
            converted = np.random.random() < (control_rate * (1 + treatment_effect))
        
        results[variant].append(converted)
    
    return results


def analyze_experiment(results: Dict[str, List[bool]], alpha: float = 0.05) -> ExperimentResult:
    """
    Perform statistical analysis on experiment results.
    
    Args:
        results: Dict with "control" and "treatment" conversion lists
        alpha: Significance level
    
    Returns:
        ExperimentResult with all statistics
    """
    from scipy import stats
    
    control = results["control"]
    treatment = results["treatment"]
    
    n_control = len(control)
    n_treatment = len(treatment)
    
    control_conversions = sum(control)
    treatment_conversions = sum(treatment)
    
    control_rate = control_conversions / n_control
    treatment_rate = treatment_conversions / n_treatment
    
    # Two-proportion z-test
    p_pooled = (control_conversions + treatment_conversions) / (n_control + n_treatment)
    se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n_control + 1/n_treatment))
    
    z_stat = (treatment_rate - control_rate) / se if se > 0 else 0
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    
    # Confidence interval for difference
    se_diff = np.sqrt(
        control_rate * (1 - control_rate) / n_control +
        treatment_rate * (1 - treatment_rate) / n_treatment
    )
    z_crit = stats.norm.ppf(1 - alpha/2)
    ci_low = (treatment_rate - control_rate) - z_crit * se_diff
    ci_high = (treatment_rate - control_rate) + z_crit * se_diff
    
    # Relative lift
    relative_lift = (treatment_rate - control_rate) / control_rate if control_rate > 0 else 0
    
    # Generate recommendation
    is_significant = p_value < alpha
    if is_significant and relative_lift > 0:
        recommendation = "SHIP_TREATMENT"
    elif is_significant and relative_lift < 0:
        recommendation = "KEEP_CONTROL"
    else:
        recommendation = "EXTEND_EXPERIMENT"
    
    return ExperimentResult(
        n_control=n_control,
        n_treatment=n_treatment,
        control_rate=control_rate,
        treatment_rate=treatment_rate,
        relative_lift=relative_lift,
        p_value=p_value,
        confidence_interval=(ci_low, ci_high),
        is_significant=is_significant,
        recommendation=recommendation
    )


def print_results(result: ExperimentResult, experiment_name: str):
    """Print formatted experiment results."""
    print("=" * 60)
    print("A/B TEST SIMULATION RESULTS")
    print("=" * 60)
    print(f"\nExperiment: {experiment_name}")
    
    print(f"\n--- Sample Sizes ---")
    print(f"  Control: {result.n_control:,}")
    print(f"  Treatment: {result.n_treatment:,}")
    
    print(f"\n--- Conversion Rates ---")
    print(f"  Control: {result.control_rate:.4f} ({result.control_rate*100:.2f}%)")
    print(f"  Treatment: {result.treatment_rate:.4f} ({result.treatment_rate*100:.2f}%)")
    print(f"  Relative Lift: {result.relative_lift:+.2%}")
    
    print(f"\n--- Statistical Analysis ---")
    print(f"  P-value: {result.p_value:.6f}")
    print(f"  95% CI: [{result.confidence_interval[0]:+.4f}, {result.confidence_interval[1]:+.4f}]")
    print(f"  Significant (α=0.05): {'Yes' if result.is_significant else 'No'}")
    
    print(f"\n--- Recommendation ---")
    print(f"  {result.recommendation}")
    
    if result.recommendation == "SHIP_TREATMENT":
        print("  → Deploy treatment to all users")
    elif result.recommendation == "KEEP_CONTROL":
        print("  → Treatment is worse, keep control")
    else:
        print("  → Collect more data before deciding")


if __name__ == "__main__":
    # Configuration
    EXPERIMENT_NAME = "model_v2_conversion_test"
    BASELINE_RATE = 0.10  # 10% conversion rate
    TREATMENT_EFFECT = 0.05  # 5% relative lift
    TOTAL_USERS = 20000
    
    print("=== A/B Test Simulation Demo ===\n")
    
    # Step 1: Sample size calculation
    print("Step 1: Sample Size Calculation")
    required_n = calculate_sample_size(
        baseline_rate=BASELINE_RATE,
        minimum_detectable_effect=TREATMENT_EFFECT
    )
    print(f"  Required per variant: {required_n:,}")
    print(f"  Simulating: {TOTAL_USERS:,} total users\n")
    
    # Step 2: Run simulation
    print("Step 2: Simulating Experiment")
    results = simulate_experiment(
        n_users=TOTAL_USERS,
        experiment_name=EXPERIMENT_NAME,
        control_rate=BASELINE_RATE,
        treatment_effect=TREATMENT_EFFECT
    )
    print(f"  Control users: {len(results['control']):,}")
    print(f"  Treatment users: {len(results['treatment']):,}\n")
    
    # Step 3: Analyze results
    print("Step 3: Statistical Analysis")
    analysis = analyze_experiment(results)
    
    # Step 4: Print final results
    print("")
    print_results(analysis, EXPERIMENT_NAME)