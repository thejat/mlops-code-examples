"""
Bias & Fairness Assessment MWE - Quantitative Fairness Metrics

Demonstrates fairness assessment for ML classification systems:

  1. Synthetic dataset — loan approval with protected attributes
  2. Group metrics — per-group accuracy, TPR, FPR, selection rates
  3. Demographic parity — equal positive outcome rates across groups
  4. Equalized odds — equal TPR and FPR across groups
  5. Disparate impact — 80% rule analysis with violation detection
  6. Fairness report — formatted summary with recommendations

Requires: numpy

Run: python main.py
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any


# =============================================================================
# Data Generation
# =============================================================================

def generate_biased_dataset(
    n_samples: int = 5000,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Generate a synthetic loan approval dataset with intentional bias.

    The model approves at different rates for different demographic groups,
    simulating real-world disparate impact.
    """
    rng = np.random.RandomState(seed)

    # Protected attributes
    gender = rng.choice(["male", "female", "non_binary"], n_samples, p=[0.48, 0.48, 0.04])
    age_group = rng.choice(["18-25", "26-40", "41-55", "56+"], n_samples, p=[0.20, 0.35, 0.30, 0.15])

    # Ground truth: base approval rate varies slightly by risk factors
    base_rate = 0.60
    y_true = (rng.random(n_samples) < base_rate).astype(int)

    # Simulated model predictions with bias:
    # - Male group gets higher approval rates
    # - 18-25 age group gets lower approval rates
    y_pred = np.zeros(n_samples, dtype=int)
    y_prob = np.zeros(n_samples)

    for i in range(n_samples):
        # Base prediction (fairly good but imperfect model)
        if y_true[i] == 1:
            prob = rng.uniform(0.55, 0.95)  # True positive tendency
        else:
            prob = rng.uniform(0.10, 0.50)  # True negative tendency

        # Inject bias: gender affects approval probability
        if gender[i] == "male":
            prob += 0.08  # Bias toward male approval
        elif gender[i] == "non_binary":
            prob -= 0.12  # Bias against non-binary

        # Inject bias: young applicants penalized
        if age_group[i] == "18-25":
            prob -= 0.10
        elif age_group[i] == "56+":
            prob -= 0.05

        prob = np.clip(prob, 0.01, 0.99)
        y_prob[i] = prob
        y_pred[i] = 1 if prob >= 0.5 else 0

    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "gender": gender,
        "age_group": age_group,
    }


# =============================================================================
# Fairness Metrics
# =============================================================================

def compute_group_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group_membership: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    """Compute classification metrics for each demographic group."""
    groups = np.unique(group_membership)
    metrics = {}

    for group in groups:
        mask = group_membership == group
        n = mask.sum()
        if n == 0:
            continue

        y_t = y_true[mask]
        y_p = y_pred[mask]

        tp = ((y_t == 1) & (y_p == 1)).sum()
        fp = ((y_t == 0) & (y_p == 1)).sum()
        tn = ((y_t == 0) & (y_p == 0)).sum()
        fn = ((y_t == 1) & (y_p == 0)).sum()

        metrics[group] = {
            "n": int(n),
            "positive_rate": float(y_p.mean()),
            "base_rate": float(y_t.mean()),
            "tpr": float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
            "fpr": float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
            "precision": float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0,
            "accuracy": float((tp + tn) / n),
        }

    return metrics


def demographic_parity_ratio(
    group_metrics: Dict[str, Dict],
    reference_group: str,
) -> Dict[str, float]:
    """
    Calculate demographic parity ratio for each group vs reference.

    Ratio of 1.0 = perfect parity.
    Ratio < 0.8 often indicates disparate impact (80% rule).
    """
    ref_rate = group_metrics[reference_group]["positive_rate"]
    ratios = {}

    for group, metrics in group_metrics.items():
        if group == reference_group:
            ratios[group] = 1.0
        else:
            ratios[group] = (
                metrics["positive_rate"] / ref_rate if ref_rate > 0 else 0.0
            )

    return ratios


def equalized_odds_difference(
    group_metrics: Dict[str, Dict],
    reference_group: str,
) -> Dict[str, Dict[str, float]]:
    """
    Calculate equalized odds difference (TPR and FPR gaps).

    Difference of 0.0 = perfect equalized odds.
    """
    ref_tpr = group_metrics[reference_group]["tpr"]
    ref_fpr = group_metrics[reference_group]["fpr"]

    differences = {}
    for group, metrics in group_metrics.items():
        differences[group] = {
            "tpr_diff": abs(metrics["tpr"] - ref_tpr),
            "fpr_diff": abs(metrics["fpr"] - ref_fpr),
        }

    return differences


def disparate_impact_analysis(
    y_pred: np.ndarray,
    protected_attr: np.ndarray,
    reference_group: str,
    threshold: float = 0.8,
) -> Dict[str, Any]:
    """
    Perform disparate impact analysis using the 80% rule.

    The 80% rule: selection rate for any group should be at least
    80% of the selection rate for the most favored group.
    """
    groups = np.unique(protected_attr)
    selection_rates = {}

    for group in groups:
        mask = protected_attr == group
        selection_rates[group] = float(y_pred[mask].mean())

    ref_rate = selection_rates[reference_group]

    results = {
        "selection_rates": selection_rates,
        "reference_group": reference_group,
        "reference_rate": ref_rate,
        "disparate_impact_ratios": {},
        "violations": [],
    }

    for group, rate in selection_rates.items():
        ratio = rate / ref_rate if ref_rate > 0 else 0.0
        results["disparate_impact_ratios"][group] = ratio

        if ratio < threshold and group != reference_group:
            results["violations"].append({
                "group": group,
                "ratio": ratio,
                "threshold": threshold,
                "selection_rate": rate,
                "reference_rate": ref_rate,
            })

    results["has_disparate_impact"] = len(results["violations"]) > 0

    return results


# =============================================================================
# Demonstrations
# =============================================================================

def demo_dataset():
    """Demo 1: Generate and describe the synthetic dataset."""
    print("=" * 60)
    print("Demo 1: Synthetic Loan Approval Dataset")
    print("=" * 60)

    data = generate_biased_dataset()

    print(f"\n  Dataset: {len(data['y_true']):,} applicants")
    print(f"  Overall approval rate (ground truth): {data['y_true'].mean():.1%}")
    print(f"  Model approval rate (predictions):    {data['y_pred'].mean():.1%}")

    print(f"\n  Gender distribution:")
    for g in np.unique(data["gender"]):
        mask = data["gender"] == g
        print(
            f"    {g:<12}: {mask.sum():>5} applicants, "
            f"approval rate: {data['y_pred'][mask].mean():.1%}"
        )

    print(f"\n  Age group distribution:")
    for ag in ["18-25", "26-40", "41-55", "56+"]:
        mask = data["age_group"] == ag
        print(
            f"    {ag:<12}: {mask.sum():>5} applicants, "
            f"approval rate: {data['y_pred'][mask].mean():.1%}"
        )

    return data


def demo_group_metrics(data: dict):
    """Demo 2: Compute per-group classification metrics."""
    print("\n" + "=" * 60)
    print("Demo 2: Per-Group Classification Metrics (Gender)")
    print("=" * 60)

    metrics = compute_group_metrics(
        data["y_true"], data["y_pred"], data["gender"]
    )

    header = (
        f"  {'Group':<12} {'N':>6} {'Select%':>8} {'Base%':>7} "
        f"{'TPR':>6} {'FPR':>6} {'Acc':>6}"
    )
    print(f"\n{header}")
    print(f"  {'—' * 12} {'—' * 6} {'—' * 8} {'—' * 7} {'—' * 6} {'—' * 6} {'—' * 6}")

    for group in sorted(metrics.keys()):
        m = metrics[group]
        print(
            f"  {group:<12} {m['n']:>6} {m['positive_rate']:>7.1%} "
            f"{m['base_rate']:>6.1%} {m['tpr']:>6.3f} {m['fpr']:>6.3f} "
            f"{m['accuracy']:>6.3f}"
        )

    return metrics


def demo_demographic_parity(data: dict, gender_metrics: dict):
    """Demo 3: Demographic parity analysis."""
    print("\n" + "=" * 60)
    print("Demo 3: Demographic Parity (Gender)")
    print("=" * 60)

    ratios = demographic_parity_ratio(gender_metrics, reference_group="male")

    print(f"\n  Reference group: male")
    print(f"  Threshold: 0.80 (80% rule)\n")

    print(f"  {'Group':<12} {'Selection Rate':>15} {'Parity Ratio':>14} {'Status':>10}")
    print(f"  {'—' * 12} {'—' * 15} {'—' * 14} {'—' * 10}")

    for group in sorted(ratios.keys()):
        rate = gender_metrics[group]["positive_rate"]
        ratio = ratios[group]
        status = "✓ PASS" if ratio >= 0.8 else "✗ FAIL"
        print(
            f"  {group:<12} {rate:>14.1%} {ratio:>14.3f} {status:>10}"
        )

    # Interpretation
    violations = [g for g, r in ratios.items() if r < 0.8]
    if violations:
        print(f"\n  ⚠ Demographic parity violated for: {violations}")
        print(f"  These groups receive positive outcomes at < 80% the rate of 'male'")
    else:
        print(f"\n  ✓ Demographic parity satisfied for all groups")


def demo_equalized_odds(gender_metrics: dict):
    """Demo 4: Equalized odds analysis."""
    print("\n" + "=" * 60)
    print("Demo 4: Equalized Odds (Gender)")
    print("=" * 60)

    diffs = equalized_odds_difference(gender_metrics, reference_group="male")

    print(f"\n  Reference group: male")
    print(f"  Acceptable gap threshold: 0.05\n")

    print(f"  {'Group':<12} {'TPR':>8} {'FPR':>8} {'TPR Gap':>9} {'FPR Gap':>9} {'Status':>10}")
    print(f"  {'—' * 12} {'—' * 8} {'—' * 8} {'—' * 9} {'—' * 9} {'—' * 10}")

    for group in sorted(diffs.keys()):
        m = gender_metrics[group]
        d = diffs[group]
        tpr_ok = d["tpr_diff"] <= 0.05
        fpr_ok = d["fpr_diff"] <= 0.05
        status = "✓ PASS" if (tpr_ok and fpr_ok) else "✗ FAIL"
        print(
            f"  {group:<12} {m['tpr']:>8.3f} {m['fpr']:>8.3f} "
            f"{d['tpr_diff']:>9.3f} {d['fpr_diff']:>9.3f} {status:>10}"
        )

    # Interpretation
    print(f"\n  Equalized odds requires both TPR and FPR to be similar")
    print(f"  across groups. Large gaps indicate the model is more")
    print(f"  accurate for some groups than others.")


def demo_disparate_impact(data: dict):
    """Demo 5: Full disparate impact analysis."""
    print("\n" + "=" * 60)
    print("Demo 5: Disparate Impact Analysis (Age Groups)")
    print("=" * 60)

    result = disparate_impact_analysis(
        data["y_pred"],
        data["age_group"],
        reference_group="26-40",
        threshold=0.8,
    )

    print(f"\n  Reference group: 26-40")
    print(f"  Reference selection rate: {result['reference_rate']:.1%}")
    print(f"  Threshold: 80%\n")

    print(f"  {'Age Group':<12} {'Selection Rate':>15} {'DI Ratio':>10} {'Status':>10}")
    print(f"  {'—' * 12} {'—' * 15} {'—' * 10} {'—' * 10}")

    for group in ["18-25", "26-40", "41-55", "56+"]:
        rate = result["selection_rates"][group]
        ratio = result["disparate_impact_ratios"][group]
        status = "✓ PASS" if ratio >= 0.8 else "✗ FAIL"
        print(
            f"  {group:<12} {rate:>14.1%} {ratio:>10.3f} {status:>10}"
        )

    if result["has_disparate_impact"]:
        print(f"\n  ⚠ DISPARATE IMPACT DETECTED")
        print(f"  Violations:")
        for v in result["violations"]:
            print(
                f"    - {v['group']}: {v['ratio']:.1%} of reference rate "
                f"(below {v['threshold']:.0%} threshold)"
            )
    else:
        print(f"\n  ✓ No disparate impact detected")


def demo_combined_report(data: dict):
    """Demo 6: Combined fairness report with recommendations."""
    print("\n" + "=" * 60)
    print("Demo 6: Combined Fairness Report")
    print("=" * 60)

    # Compute all metrics
    gender_metrics = compute_group_metrics(
        data["y_true"], data["y_pred"], data["gender"]
    )
    age_metrics = compute_group_metrics(
        data["y_true"], data["y_pred"], data["age_group"]
    )

    gender_parity = demographic_parity_ratio(gender_metrics, "male")
    age_parity = demographic_parity_ratio(age_metrics, "26-40")
    gender_eo = equalized_odds_difference(gender_metrics, "male")

    gender_di = disparate_impact_analysis(
        data["y_pred"], data["gender"], "male"
    )
    age_di = disparate_impact_analysis(
        data["y_pred"], data["age_group"], "26-40"
    )

    # Summary
    print(f"\n  Fairness Assessment Summary")
    print(f"  {'=' * 50}\n")

    print(f"  {'Check':<35} {'Result':>10}")
    print(f"  {'—' * 35} {'—' * 10}")

    # Gender checks
    gender_dp_pass = all(r >= 0.8 for r in gender_parity.values())
    gender_eo_pass = all(
        d["tpr_diff"] <= 0.05 and d["fpr_diff"] <= 0.05
        for d in gender_eo.values()
    )
    gender_di_pass = not gender_di["has_disparate_impact"]

    print(f"  {'Gender: Demographic Parity':<35} {'✓ PASS' if gender_dp_pass else '✗ FAIL':>10}")
    print(f"  {'Gender: Equalized Odds':<35} {'✓ PASS' if gender_eo_pass else '✗ FAIL':>10}")
    print(f"  {'Gender: Disparate Impact (80%)':<35} {'✓ PASS' if gender_di_pass else '✗ FAIL':>10}")

    # Age checks
    age_dp_pass = all(r >= 0.8 for r in age_parity.values())
    age_di_pass = not age_di["has_disparate_impact"]

    print(f"  {'Age: Demographic Parity':<35} {'✓ PASS' if age_dp_pass else '✗ FAIL':>10}")
    print(f"  {'Age: Disparate Impact (80%)':<35} {'✓ PASS' if age_di_pass else '✗ FAIL':>10}")

    # Recommendations
    failures = []
    if not gender_dp_pass:
        failures.append("Gender demographic parity")
    if not gender_eo_pass:
        failures.append("Gender equalized odds")
    if not gender_di_pass:
        failures.append("Gender disparate impact")
    if not age_dp_pass:
        failures.append("Age demographic parity")
    if not age_di_pass:
        failures.append("Age disparate impact")

    print(f"\n  Recommendations:")
    if failures:
        print(f"  ⚠ {len(failures)} fairness check(s) failed:")
        for f in failures:
            print(f"    • {f}")
        print(f"\n  Suggested actions:")
        print(f"    1. Investigate features correlated with protected attributes")
        print(f"    2. Consider threshold adjustment per group (with legal review)")
        print(f"    3. Add fairness constraints to model training objective")
        print(f"    4. Document findings in model card limitations section")
        print(f"    5. Add to risk register with review triggers")
    else:
        print(f"  ✓ All fairness checks passed")
        print(f"  Continue monitoring — fairness can degrade over time")


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("Bias & Fairness Assessment MWE — Quantitative Fairness Metrics")
    print("=" * 60)

    data = demo_dataset()
    gender_metrics = demo_group_metrics(data)
    demo_demographic_parity(data, gender_metrics)
    demo_equalized_odds(gender_metrics)
    demo_disparate_impact(data)
    demo_combined_report(data)

    print("\n" + "=" * 60)
    print("MWE Complete — Bias and fairness assessment demonstrated!")
    print("=" * 60)


if __name__ == "__main__":
    main()
