"""
Model Validation Script with Quality Gates

This script validates a trained model against configurable thresholds.
It exits with code 1 if validation fails, causing CI/CD pipelines to stop.

Usage:
    python validate.py --min-accuracy 0.90 --min-f1 0.85
"""

import argparse
import json
import sys
import os


def validate_model(min_accuracy: float, min_f1: float, metrics_file: str = "metrics.json"):
    """
    Validate model metrics against thresholds.
    
    Returns:
        bool: True if all validations pass, False otherwise
    """
    
    # Check if metrics file exists
    if not os.path.exists(metrics_file):
        print(f"ERROR: Metrics file not found: {metrics_file}")
        print("Run 'python train.py' first to generate metrics.")
        sys.exit(1)
    
    # Load metrics
    with open(metrics_file) as f:
        metrics = json.load(f)
    
    accuracy = metrics.get("accuracy", 0)
    f1_score = metrics.get("f1_score", 0)
    
    # Track validation results
    validations = []
    all_passed = True
    
    # Check accuracy threshold
    accuracy_passed = accuracy >= min_accuracy
    validations.append({
        "metric": "accuracy",
        "value": accuracy,
        "threshold": min_accuracy,
        "passed": accuracy_passed
    })
    if not accuracy_passed:
        all_passed = False
    
    # Check F1 threshold
    f1_passed = f1_score >= min_f1
    validations.append({
        "metric": "f1_score",
        "value": f1_score,
        "threshold": min_f1,
        "passed": f1_passed
    })
    if not f1_passed:
        all_passed = False
    
    # Create validation report
    report = {
        "validations": validations,
        "overall_passed": all_passed,
        "thresholds": {
            "min_accuracy": min_accuracy,
            "min_f1": min_f1
        }
    }
    
    # Save report
    with open("validation_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    # Print results
    print("\n=== Model Validation Report ===")
    for v in validations:
        status = "✓ PASS" if v["passed"] else "✗ FAIL"
        print(f"{status}: {v['metric']} = {v['value']:.4f} (threshold: {v['threshold']:.4f})")
    
    print(f"\nOverall: {'PASSED' if all_passed else 'FAILED'}")
    print("Validation report saved to: validation_report.json")
    
    return all_passed


def main():
    parser = argparse.ArgumentParser(
        description="Validate model metrics against quality thresholds"
    )
    parser.add_argument(
        "--min-accuracy",
        type=float,
        default=0.90,
        help="Minimum required accuracy (default: 0.90)"
    )
    parser.add_argument(
        "--min-f1",
        type=float,
        default=0.85,
        help="Minimum required F1 score (default: 0.85)"
    )
    parser.add_argument(
        "--metrics-file",
        type=str,
        default="metrics.json",
        help="Path to metrics JSON file (default: metrics.json)"
    )
    
    args = parser.parse_args()
    
    passed = validate_model(
        min_accuracy=args.min_accuracy,
        min_f1=args.min_f1,
        metrics_file=args.metrics_file
    )
    
    # Exit with appropriate code
    # 0 = success (pipeline continues)
    # 1 = failure (pipeline stops)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()