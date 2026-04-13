"""
Statistical Testing MWE - Multiple Comparison Corrections for ML Experiments

Demonstrates statistical testing methodology for A/B experiments:

  1. Two-proportion z-test — compare conversion rates between variants
  2. T-test for continuous metrics — compare means (e.g., latency, scores)
  3. Bonferroni correction — control family-wise error rate for multiple tests
  4. Confidence intervals — CI-based decision making
  5. Guardrail checks — secondary metrics that must not degrade
  6. Integrated decision framework — combine all checks into a recommendation

Requires: numpy, scipy

Run: python main.py
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from scipy import stats


# =============================================================================
# Statistical Tests
# =============================================================================

@dataclass
class TestResult:
    """Result of a single statistical test."""
    metric_name: str
    test_type: str  # "proportion_ztest" or "ttest"
    control_value: float
    treatment_value: float
    difference: float
    relative_lift: float
    statistic: float
    p_value: float
    ci_lower: float
    ci_upper: float
    is_significant: bool
    alpha: float


def two_proportion_ztest(
    control_conversions: int,
    control_n: int,
    treatment_conversions: int,
    treatment_n: int,
    alpha: float = 0.05,
    metric_name: str = "conversion_rate",
) -> TestResult:
    """
    Two-proportion z-test for comparing conversion rates.

    Tests H0: p_treatment = p_control
    vs   H1: p_treatment ≠ p_control (two-sided)
    """
    p_control = control_conversions / control_n
    p_treatment = treatment_conversions / treatment_n

    # Pooled proportion under H0
    p_pooled = (control_conversions + treatment_conversions) / (
        control_n + treatment_n
    )

    # Standard error under H0
    se = np.sqrt(
        p_pooled * (1 - p_pooled) * (1 / control_n + 1 / treatment_n)
    )

    # Z-statistic
    z_stat = (p_treatment - p_control) / se if se > 0 else 0.0
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    # Confidence interval for the difference
    se_diff = np.sqrt(
        p_control * (1 - p_control) / control_n
        + p_treatment * (1 - p_treatment) / treatment_n
    )
    z_crit = stats.norm.ppf(1 - alpha / 2)
    diff = p_treatment - p_control
    ci_lower = diff - z_crit * se_diff
    ci_upper = diff + z_crit * se_diff

    relative_lift = diff / p_control if p_control > 0 else 0.0

    return TestResult(
        metric_name=metric_name,
        test_type="proportion_ztest",
        control_value=p_control,
        treatment_value=p_treatment,
        difference=diff,
        relative_lift=relative_lift,
        statistic=z_stat,
        p_value=p_value,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        is_significant=p_value < alpha,
        alpha=alpha,
    )


def two_sample_ttest(
    control_values: np.ndarray,
    treatment_values: np.ndarray,
    alpha: float = 0.05,
    metric_name: str = "continuous_metric",
) -> TestResult:
    """
    Two-sample t-test for comparing continuous metrics.

    Tests H0: μ_treatment = μ_control
    vs   H1: μ_treatment ≠ μ_control (two-sided)
    """
    control_mean = control_values.mean()
    treatment_mean = treatment_values.mean()
    diff = treatment_mean - control_mean

    # Welch's t-test (unequal variances)
    t_stat, p_value = stats.ttest_ind(
        treatment_values, control_values, equal_var=False
    )

    # Confidence interval for the difference
    n1, n2 = len(control_values), len(treatment_values)
    se = np.sqrt(
        control_values.var(ddof=1) / n1
        + treatment_values.var(ddof=1) / n2
    )

    # Degrees of freedom (Welch-Satterthwaite)
    s1_sq = control_values.var(ddof=1)
    s2_sq = treatment_values.var(ddof=1)
    df_num = (s1_sq / n1 + s2_sq / n2) ** 2
    df_den = (
        (s1_sq / n1) ** 2 / (n1 - 1) + (s2_sq / n2) ** 2 / (n2 - 1)
    )
    df = df_num / df_den if df_den > 0 else n1 + n2 - 2

    t_crit = stats.t.ppf(1 - alpha / 2, df)
    ci_lower = diff - t_crit * se
    ci_upper = diff + t_crit * se

    relative_lift = diff / control_mean if control_mean != 0 else 0.0

    return TestResult(
        metric_name=metric_name,
        test_type="welch_ttest",
        control_value=control_mean,
        treatment_value=treatment_mean,
        difference=diff,
        relative_lift=relative_lift,
        statistic=t_stat,
        p_value=p_value,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        is_significant=p_value < alpha,
        alpha=alpha,
    )


# =============================================================================
# Multiple Comparison Correction
# =============================================================================

def bonferroni_correction(
    results: List[TestResult],
    alpha: float = 0.05,
) -> List[Dict]:
    """
    Apply Bonferroni correction to multiple test results.

    Divides α by the number of tests to control family-wise error rate.
    """
    n_tests = len(results)
    adjusted_alpha = alpha / n_tests

    corrected = []
    for result in results:
        corrected.append({
            "metric": result.metric_name,
            "original_p": result.p_value,
            "adjusted_alpha": adjusted_alpha,
            "original_significant": result.is_significant,
            "corrected_significant": result.p_value < adjusted_alpha,
            "n_tests": n_tests,
        })

    return corrected


# =============================================================================
# Guardrail Check
# =============================================================================

@dataclass
class GuardrailResult:
    """Result of a guardrail metric check."""
    metric_name: str
    control_value: float
    treatment_value: float
    max_degradation: float  # e.g., 0.05 for 5%
    actual_change: float
    passed: bool
    reason: str


def check_guardrail(
    control_values: np.ndarray,
    treatment_values: np.ndarray,
    metric_name: str,
    max_degradation: float = 0.05,
    higher_is_better: bool = False,
) -> GuardrailResult:
    """
    Check whether a guardrail metric has degraded beyond threshold.

    For metrics where lower is better (latency, error rate):
      fails if treatment is more than max_degradation% higher

    For metrics where higher is better (accuracy):
      fails if treatment is more than max_degradation% lower
    """
    control_mean = control_values.mean()
    treatment_mean = treatment_values.mean()

    if higher_is_better:
        change = (treatment_mean - control_mean) / control_mean
        passed = change >= -max_degradation
        reason = (
            f"{'improved' if change >= 0 else 'degraded'} by {abs(change):.1%}"
        )
    else:
        change = (treatment_mean - control_mean) / control_mean
        passed = change <= max_degradation
        reason = (
            f"{'improved' if change <= 0 else 'degraded'} by {abs(change):.1%}"
        )

    return GuardrailResult(
        metric_name=metric_name,
        control_value=control_mean,
        treatment_value=treatment_mean,
        max_degradation=max_degradation,
        actual_change=change,
        passed=passed,
        reason=reason,
    )


# =============================================================================
# Demonstrations
# =============================================================================

def demo_proportion_test():
    """Demo 1: Two-proportion z-test for conversion rates."""
    print("=" * 60)
    print("Demo 1: Two-Proportion Z-Test (Conversion Rate)")
    print("=" * 60)

    np.random.seed(42)

    # Simulate experiment
    n_control = 10000
    n_treatment = 10000
    control_rate = 0.10
    treatment_rate = 0.105  # 5% relative lift

    control_conv = np.random.binomial(n_control, control_rate)
    treatment_conv = np.random.binomial(n_treatment, treatment_rate)

    result = two_proportion_ztest(
        control_conv, n_control,
        treatment_conv, n_treatment,
        metric_name="conversion_rate",
    )

    print(f"\n  Experiment: Model v2.0 vs v1.5")
    print(f"  Metric: Conversion rate (binary)\n")

    print(f"  {'Variant':<12} {'N':>8} {'Conversions':>13} {'Rate':>8}")
    print(f"  {'—' * 12} {'—' * 8} {'—' * 13} {'—' * 8}")
    print(f"  {'Control':<12} {n_control:>8} {control_conv:>13} {result.control_value:>7.2%}")
    print(f"  {'Treatment':<12} {n_treatment:>8} {treatment_conv:>13} {result.treatment_value:>7.2%}")

    print(f"\n  Relative lift: {result.relative_lift:+.2%}")
    print(f"  Z-statistic: {result.statistic:.4f}")
    print(f"  P-value: {result.p_value:.6f}")
    print(f"  95% CI: [{result.ci_lower:+.4f}, {result.ci_upper:+.4f}]")
    print(f"  Significant (α=0.05): {'Yes' if result.is_significant else 'No'}")

    return result


def demo_ttest():
    """Demo 2: T-test for continuous metrics."""
    print("\n" + "=" * 60)
    print("Demo 2: Welch's T-Test (Continuous Metrics)")
    print("=" * 60)

    np.random.seed(42)

    # Simulate retrieval similarity scores
    n = 5000
    control_scores = np.random.normal(0.72, 0.15, n)
    treatment_scores = np.random.normal(0.76, 0.14, n)  # Slight improvement

    result = two_sample_ttest(
        control_scores, treatment_scores,
        metric_name="retrieval_similarity",
    )

    print(f"\n  Experiment: RAG top-k=5 vs top-k=3")
    print(f"  Metric: Mean retrieval similarity score (continuous)\n")

    print(f"  {'Variant':<12} {'N':>8} {'Mean':>8} {'Std':>8}")
    print(f"  {'—' * 12} {'—' * 8} {'—' * 8} {'—' * 8}")
    print(
        f"  {'Control':<12} {n:>8} {result.control_value:>8.4f} "
        f"{control_scores.std():>8.4f}"
    )
    print(
        f"  {'Treatment':<12} {n:>8} {result.treatment_value:>8.4f} "
        f"{treatment_scores.std():>8.4f}"
    )

    print(f"\n  Mean difference: {result.difference:+.4f}")
    print(f"  Relative lift: {result.relative_lift:+.2%}")
    print(f"  T-statistic: {result.statistic:.4f}")
    print(f"  P-value: {result.p_value:.6f}")
    print(f"  95% CI: [{result.ci_lower:+.4f}, {result.ci_upper:+.4f}]")
    print(f"  Significant (α=0.05): {'Yes' if result.is_significant else 'No'}")

    print(f"\n  Key insight: For continuous metrics (latency, similarity scores),")
    print(f"  use t-test instead of the two-proportion z-test.")

    return result


def demo_bonferroni():
    """Demo 3: Multiple comparison correction."""
    print("\n" + "=" * 60)
    print("Demo 3: Bonferroni Correction (Multiple Comparisons)")
    print("=" * 60)

    np.random.seed(42)

    # Simulate 3 metrics tested simultaneously
    n = 8000

    results = []

    # Metric 1: Conversion rate (true effect)
    c_conv = np.random.binomial(n, 0.10)
    t_conv = np.random.binomial(n, 0.108)
    results.append(two_proportion_ztest(
        c_conv, n, t_conv, n, metric_name="conversion_rate"
    ))

    # Metric 2: Click-through rate (marginal effect)
    c_ctr = np.random.binomial(n, 0.25)
    t_ctr = np.random.binomial(n, 0.258)
    results.append(two_proportion_ztest(
        c_ctr, n, t_ctr, n, metric_name="click_through_rate"
    ))

    # Metric 3: Engagement score (no real effect - noise)
    c_eng = np.random.normal(4.2, 1.1, n)
    t_eng = np.random.normal(4.2, 1.1, n)  # Same distribution
    results.append(two_sample_ttest(
        c_eng, t_eng, metric_name="engagement_score"
    ))

    # Show uncorrected results
    print(f"\n  Testing 3 metrics simultaneously:\n")

    print(f"  {'Metric':<22} {'P-value':>10} {'α=0.05':>10} {'Significant?':>13}")
    print(f"  {'—' * 22} {'—' * 10} {'—' * 10} {'—' * 13}")

    for r in results:
        sig = "Yes ✓" if r.is_significant else "No"
        print(
            f"  {r.metric_name:<22} {r.p_value:>10.6f} {0.05:>10.3f} "
            f"{sig:>13}"
        )

    # Apply Bonferroni
    corrected = bonferroni_correction(results, alpha=0.05)

    print(f"\n  After Bonferroni correction (α/{len(results)} = "
          f"{0.05/len(results):.4f}):\n")

    print(
        f"  {'Metric':<22} {'P-value':>10} {'Adj. α':>10} "
        f"{'Original':>10} {'Corrected':>10}"
    )
    print(
        f"  {'—' * 22} {'—' * 10} {'—' * 10} {'—' * 10} {'—' * 10}"
    )

    for c in corrected:
        orig = "Yes" if c["original_significant"] else "No"
        corr = "Yes ✓" if c["corrected_significant"] else "No ✗"
        print(
            f"  {c['metric']:<22} {c['original_p']:>10.6f} "
            f"{c['adjusted_alpha']:>10.4f} {orig:>10} {corr:>10}"
        )

    # Count changes
    flipped = sum(
        1 for c in corrected
        if c["original_significant"] and not c["corrected_significant"]
    )
    if flipped > 0:
        print(f"\n  ⚠ {flipped} metric(s) lost significance after correction")
        print(f"  This prevents false discoveries from testing multiple metrics")
    else:
        print(f"\n  No changes in significance after correction")


def demo_confidence_intervals():
    """Demo 4: CI-based decision making."""
    print("\n" + "=" * 60)
    print("Demo 4: Confidence Interval Decision Framework")
    print("=" * 60)

    np.random.seed(42)

    scenarios = [
        {
            "name": "Strong positive effect",
            "c_rate": 0.10, "t_rate": 0.115, "n": 15000,
        },
        {
            "name": "Weak positive (underpowered)",
            "c_rate": 0.10, "t_rate": 0.103, "n": 3000,
        },
        {
            "name": "No effect (null result)",
            "c_rate": 0.10, "t_rate": 0.100, "n": 15000,
        },
        {
            "name": "Negative effect",
            "c_rate": 0.10, "t_rate": 0.088, "n": 15000,
        },
    ]

    print(f"\n  {'Scenario':<30} {'P-value':>8} {'95% CI':>22} {'Decision'}")
    print(f"  {'—' * 30} {'—' * 8} {'—' * 22} {'—' * 20}")

    for s in scenarios:
        c_conv = np.random.binomial(s["n"], s["c_rate"])
        t_conv = np.random.binomial(s["n"], s["t_rate"])

        result = two_proportion_ztest(
            c_conv, s["n"], t_conv, s["n"],
            metric_name=s["name"],
        )

        # Decision based on CI
        if result.ci_lower > 0 and result.is_significant:
            decision = "SHIP TREATMENT"
        elif result.ci_upper < 0 and result.is_significant:
            decision = "KEEP CONTROL"
        elif result.ci_upper - result.ci_lower > 0.02:
            decision = "EXTEND EXPERIMENT"
        else:
            decision = "NO DIFFERENCE"

        ci_str = f"[{result.ci_lower:+.4f}, {result.ci_upper:+.4f}]"
        print(
            f"  {s['name']:<30} {result.p_value:>8.4f} {ci_str:>22} "
            f"{decision}"
        )

    print(f"\n  Decision rules:")
    print(f"    CI entirely > 0 & significant → SHIP TREATMENT")
    print(f"    CI entirely < 0 & significant → KEEP CONTROL")
    print(f"    CI wide (spans 0, range > 2%) → EXTEND EXPERIMENT")
    print(f"    CI narrow around 0            → NO DIFFERENCE")


def demo_guardrails():
    """Demo 5: Guardrail metric checks."""
    print("\n" + "=" * 60)
    print("Demo 5: Guardrail Metric Checks")
    print("=" * 60)

    np.random.seed(42)
    n = 10000

    # Simulate guardrail metrics
    guardrails = [
        {
            "name": "p99_latency_ms",
            "control": np.random.exponential(100, n) + 50,
            "treatment": np.random.exponential(105, n) + 52,
            "max_degradation": 0.10,
            "higher_is_better": False,
        },
        {
            "name": "error_rate",
            "control": np.random.binomial(1, 0.02, n).astype(float),
            "treatment": np.random.binomial(1, 0.025, n).astype(float),
            "max_degradation": 0.50,  # 50% relative increase max
            "higher_is_better": False,
        },
        {
            "name": "cache_hit_rate",
            "control": np.random.binomial(1, 0.65, n).astype(float),
            "treatment": np.random.binomial(1, 0.63, n).astype(float),
            "max_degradation": 0.05,
            "higher_is_better": True,
        },
    ]

    print(f"\n  Guardrail metrics must not degrade beyond threshold:\n")

    print(
        f"  {'Metric':<18} {'Control':>10} {'Treatment':>10} "
        f"{'Change':>8} {'Max Deg':>8} {'Status':>10}"
    )
    print(
        f"  {'—' * 18} {'—' * 10} {'—' * 10} {'—' * 8} {'—' * 8} {'—' * 10}"
    )

    all_passed = True
    for g in guardrails:
        result = check_guardrail(
            g["control"], g["treatment"],
            metric_name=g["name"],
            max_degradation=g["max_degradation"],
            higher_is_better=g["higher_is_better"],
        )
        status = "✓ PASS" if result.passed else "✗ FAIL"
        if not result.passed:
            all_passed = False
        print(
            f"  {g['name']:<18} {result.control_value:>10.4f} "
            f"{result.treatment_value:>10.4f} {result.actual_change:>+7.1%} "
            f"{g['max_degradation']:>+7.0%} {status:>10}"
        )

    if all_passed:
        print(f"\n  ✓ All guardrails passed — safe to ship if primary metric significant")
    else:
        print(f"\n  ⚠ Guardrail failure — investigate before shipping")


def demo_integrated_decision():
    """Demo 6: Integrated decision framework."""
    print("\n" + "=" * 60)
    print("Demo 6: Integrated Decision Framework")
    print("=" * 60)

    np.random.seed(42)
    n = 12000

    # Primary metric: conversion rate
    c_conv = np.random.binomial(n, 0.10)
    t_conv = np.random.binomial(n, 0.108)
    primary = two_proportion_ztest(
        c_conv, n, t_conv, n, metric_name="conversion_rate"
    )

    # Secondary metric: engagement (continuous)
    c_eng = np.random.normal(4.2, 1.1, n)
    t_eng = np.random.normal(4.3, 1.1, n)
    secondary = two_sample_ttest(
        c_eng, t_eng, metric_name="engagement_score"
    )

    # Apply Bonferroni to primary + secondary
    corrected = bonferroni_correction([primary, secondary])

    # Guardrail: latency
    c_lat = np.random.exponential(100, n) + 50
    t_lat = np.random.exponential(103, n) + 51
    guardrail = check_guardrail(
        c_lat, t_lat, "p99_latency_ms",
        max_degradation=0.10, higher_is_better=False,
    )

    # Display integrated results
    print(f"\n  ┌{'─' * 54}┐")
    print(f"  │{'EXPERIMENT RESULTS SUMMARY':^54}│")
    print(f"  ├{'─' * 54}┤")

    print(f"  │ Primary: {primary.metric_name:<20} "
          f"lift={primary.relative_lift:+.2%} "
          f"p={primary.p_value:.4f} │")
    print(f"  │ Secondary: {secondary.metric_name:<18} "
          f"lift={secondary.relative_lift:+.2%} "
          f"p={secondary.p_value:.4f} │")
    print(f"  │ Guardrail: {guardrail.metric_name:<18} "
          f"{'PASS' if guardrail.passed else 'FAIL':>4} "
          f"({guardrail.actual_change:+.1%})"
          f"{'':>11}│")

    primary_sig = corrected[0]["corrected_significant"]
    secondary_sig = corrected[1]["corrected_significant"]
    adj_alpha = corrected[0]["adjusted_alpha"]

    print(f"  │{'':54}│")
    print(f"  │ Bonferroni α: {adj_alpha:.4f} (2 tests)"
          f"{'':>22}│")
    print(f"  │ Primary significant (corrected): "
          f"{'Yes' if primary_sig else 'No':>4}"
          f"{'':>14}│")
    print(f"  │ Secondary significant (corrected): "
          f"{'Yes' if secondary_sig else 'No':>4}"
          f"{'':>12}│")

    # Final decision
    if primary_sig and primary.relative_lift > 0 and guardrail.passed:
        decision = "SHIP TREATMENT"
        reason = "Primary metric significant, positive lift, guardrails pass"
    elif primary_sig and primary.relative_lift > 0 and not guardrail.passed:
        decision = "INVESTIGATE"
        reason = "Primary positive but guardrail failed — resolve before shipping"
    elif primary_sig and primary.relative_lift < 0:
        decision = "KEEP CONTROL"
        reason = "Treatment is significantly worse than control"
    else:
        decision = "EXTEND EXPERIMENT"
        reason = "Insufficient evidence — collect more data"

    print(f"  │{'':54}│")
    print(f"  │ Decision: {decision:<43}│")
    print(f"  │ {reason[:53]:<53}│")
    print(f"  └{'─' * 54}┘")

    print(f"\n  The decision integrates:")
    print(f"    1. Primary metric significance (with Bonferroni correction)")
    print(f"    2. Direction of effect (positive lift required)")
    print(f"    3. Guardrail compliance (no degradation beyond threshold)")
    print(f"    4. Confidence interval width (practical significance)")


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("Statistical Testing MWE — Multiple Comparisons for ML")
    print("=" * 60)

    demo_proportion_test()
    demo_ttest()
    demo_bonferroni()
    demo_confidence_intervals()
    demo_guardrails()
    demo_integrated_decision()

    print("\n" + "=" * 60)
    print("MWE Complete — Statistical testing for ML demonstrated!")
    print("=" * 60)


if __name__ == "__main__":
    main()
