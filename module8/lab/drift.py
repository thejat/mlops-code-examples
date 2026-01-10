"""
Lab: Module 8 - Drift Detection Fundamentals
Time: ~30 minutes
Prerequisites: Read the following instructional materials first:
  - src/module8/instructional-materials/drift-detection.md
  - src/module8/instructional-materials/drift-impact-analysis.md
  - src/module8/instructional-materials/production-monitoring-fundamentals.md

Learning Objectives:
- LO1: Calculate Population Stability Index (PSI) for numerical features
- LO2: Detect drift in categorical features using chi-square tests
- LO3: Weight drift scores by feature importance for prioritization
- LO4: Generate severity classifications and action recommendations
- LO5: Create a drift diagnostic report with visualizations

Final Project Alignment:
- CG4.LO1: Design monitoring dashboards to track drift, integrity, and latency
- Component 4: Data Integrity & Drift Detection scripts and diagnostic report
"""

# === REQUIREMENTS ===
# Run: pip install numpy pandas scipy matplotlib

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


# === SAMPLE DATA (provided) ===
# Simulates reference (training) and production data for a loan approval model

def generate_sample_data(seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate reference and production datasets with intentional drift.
    
    The reference dataset represents training data distribution.
    The production dataset has drift in some features to simulate real-world shift.
    """
    np.random.seed(seed)
    n_reference = 5000
    n_production = 2000
    
    # Reference data (training distribution)
    reference = pd.DataFrame({
        # Numerical features
        'age': np.random.normal(35, 10, n_reference).clip(18, 80),
        'income': np.random.lognormal(10.5, 0.5, n_reference),
        'credit_score': np.random.normal(680, 80, n_reference).clip(300, 850),
        'loan_amount': np.random.lognormal(10, 0.8, n_reference),
        'employment_years': np.random.exponential(5, n_reference).clip(0, 40),
        
        # Categorical features
        'employment_type': np.random.choice(
            ['full_time', 'part_time', 'self_employed', 'contract'],
            n_reference,
            p=[0.65, 0.15, 0.12, 0.08]
        ),
        'loan_purpose': np.random.choice(
            ['home', 'auto', 'education', 'personal', 'business'],
            n_reference,
            p=[0.30, 0.25, 0.15, 0.20, 0.10]
        )
    })
    
    # Production data (with intentional drift)
    production = pd.DataFrame({
        # 'age' - SIGNIFICANT DRIFT: older population
        'age': np.random.normal(42, 12, n_production).clip(18, 80),
        
        # 'income' - MODERATE DRIFT: slightly higher incomes
        'income': np.random.lognormal(10.7, 0.6, n_production),
        
        # 'credit_score' - NO SIGNIFICANT DRIFT
        'credit_score': np.random.normal(675, 85, n_production).clip(300, 850),
        
        # 'loan_amount' - MODERATE DRIFT: larger loans
        'loan_amount': np.random.lognormal(10.3, 0.9, n_production),
        
        # 'employment_years' - NO SIGNIFICANT DRIFT
        'employment_years': np.random.exponential(5.2, n_production).clip(0, 40),
        
        # 'employment_type' - SIGNIFICANT DRIFT: more self-employed
        'employment_type': np.random.choice(
            ['full_time', 'part_time', 'self_employed', 'contract'],
            n_production,
            p=[0.45, 0.15, 0.30, 0.10]  # Changed from 65/15/12/8
        ),
        
        # 'loan_purpose' - MODERATE DRIFT: more business loans
        'loan_purpose': np.random.choice(
            ['home', 'auto', 'education', 'personal', 'business'],
            n_production,
            p=[0.22, 0.20, 0.13, 0.20, 0.25]  # Changed from 30/25/15/20/10
        )
    })
    
    return reference, production


# Feature importance from a hypothetical trained model
FEATURE_IMPORTANCES = {
    'credit_score': 0.30,
    'income': 0.25,
    'age': 0.15,
    'loan_amount': 0.12,
    'employment_years': 0.08,
    'employment_type': 0.06,
    'loan_purpose': 0.04
}

# Features by type
NUMERICAL_FEATURES = ['age', 'income', 'credit_score', 'loan_amount', 'employment_years']
CATEGORICAL_FEATURES = ['employment_type', 'loan_purpose']


# === HELPER CLASSES (provided) ===

class DriftSeverity(Enum):
    """Drift severity levels based on PSI thresholds."""
    NONE = "none"        # PSI < 0.1
    LOW = "low"          # 0.1 <= PSI < 0.15
    MODERATE = "moderate"  # 0.15 <= PSI < 0.2
    HIGH = "high"        # 0.2 <= PSI < 0.25
    CRITICAL = "critical"  # PSI >= 0.25


@dataclass
class DriftResult:
    """Results from drift detection for a single feature."""
    feature_name: str
    drift_score: float  # PSI for numerical, chi-square p-value for categorical
    severity: DriftSeverity
    drift_detected: bool
    feature_type: str  # 'numerical' or 'categorical'
    importance_weight: float
    weighted_score: float
    details: Dict  # Additional metrics


# === TODO 1: Calculate PSI for Numerical Features ===
# Implement the Population Stability Index calculation.
#
# PSI Formula:
#   PSI = Σ (current_pct - reference_pct) * ln(current_pct / reference_pct)
#
# PSI Interpretation:
#   < 0.1: No significant drift
#   0.1 - 0.2: Moderate drift, investigate
#   > 0.2: Significant drift, action required
#
# Hint: Add a small value (smoothing) to avoid division by zero

def calculate_psi(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10
) -> Tuple[float, Dict]:
    """
    Calculate Population Stability Index (PSI).
    
    Args:
        reference: Array of values from reference (training) distribution
        current: Array of values from current (production) distribution
        n_bins: Number of bins for discretization
    
    Returns:
        Tuple of (psi_score, details_dict)
        details_dict should contain:
            - bin_edges: The bin boundaries
            - reference_pct: Percentage in each bin for reference
            - current_pct: Percentage in each bin for current
            - bin_contributions: PSI contribution from each bin
    """
    # TODO: Implement PSI calculation
    # 1. Create bins from reference distribution using np.histogram
    # 2. Calculate proportions in each bin for both distributions
    # 3. Add smoothing (e.g., add 1 to counts, adjust denominator)
    # 4. Calculate PSI using the formula
    # 5. Return PSI and details dictionary
    
    # Your code here
    pass


# === TODO 2: Detect Drift in Categorical Features ===
# Use chi-square test to detect distribution changes in categorical features.
#
# Requirements:
# - Calculate observed frequencies in current data
# - Calculate expected frequencies based on reference distribution
# - Perform chi-square test
# - Interpret p-value for drift detection

def calculate_categorical_drift(
    reference: pd.Series,
    current: pd.Series,
    alpha: float = 0.05
) -> Tuple[float, Dict]:
    """
    Detect drift in categorical features using chi-square test.
    
    Args:
        reference: Series of categorical values from reference distribution
        current: Series of categorical values from current distribution
        alpha: Significance level for drift detection
    
    Returns:
        Tuple of (psi_equivalent_score, details_dict)
        details_dict should contain:
            - chi2_statistic: The chi-square test statistic
            - p_value: The p-value from the test
            - category_counts_ref: Value counts from reference
            - category_counts_cur: Value counts from current
            - new_categories: Categories in current but not in reference
    """
    from scipy import stats
    
    # TODO: Implement categorical drift detection
    # 1. Get all unique categories from both distributions
    # 2. Calculate frequency for each category in reference and current
    # 3. Scale reference frequencies to match current sample size
    # 4. Perform chi-square test using scipy.stats.chisquare
    # 5. Convert to PSI-equivalent score for consistent severity assessment
    #    (e.g., use -log10(p_value) / 10 capped at 0.5, or similar mapping)
    # 6. Return the score and details
    
    # Your code here
    pass


# === TODO 3: Classify Drift Severity ===
# Map drift scores to severity levels.
#
# Requirements:
# - Use PSI thresholds for numerical features
# - Use p-value thresholds for categorical features
# - Return appropriate DriftSeverity enum value

def classify_severity(drift_score: float, feature_type: str) -> DriftSeverity:
    """
    Classify drift severity based on score and feature type.
    
    Args:
        drift_score: PSI score or PSI-equivalent for categorical
        feature_type: 'numerical' or 'categorical'
    
    Returns:
        DriftSeverity enum value
    """
    # TODO: Implement severity classification
    # Use these thresholds for PSI:
    #   < 0.1: NONE
    #   0.1 - 0.15: LOW  
    #   0.15 - 0.2: MODERATE
    #   0.2 - 0.25: HIGH
    #   >= 0.25: CRITICAL
    
    # Your code here
    pass


# === TODO 4: Calculate Importance-Weighted Drift ===
# Weight drift scores by feature importance to prioritize high-impact drift.
#
# A feature with high drift AND high importance should be flagged more urgently
# than a feature with high drift but low importance.

def calculate_weighted_drift(
    drift_score: float,
    importance: float,
    importance_weights: Dict[str, float]
) -> float:
    """
    Calculate importance-weighted drift score.
    
    Args:
        drift_score: Raw drift score (PSI or equivalent)
        importance: Feature importance weight (0-1)
        importance_weights: Dict of all feature importances (for normalization)
    
    Returns:
        Weighted drift score
    """
    # TODO: Implement weighted drift calculation
    # Formula: weighted_score = drift_score * normalized_importance * scaling_factor
    # Where normalized_importance = importance / sum(all_importances)
    # Use a scaling factor (e.g., 100) for readability
    
    # Your code here
    pass


# === TODO 5: Complete Drift Detector Class ===
# Orchestrate drift detection across all features.

class DriftDetector:
    """Multi-feature drift detection with importance weighting."""
    
    def __init__(
        self,
        reference_data: pd.DataFrame,
        feature_importances: Dict[str, float],
        numerical_features: List[str],
        categorical_features: List[str],
        psi_threshold: float = 0.2
    ):
        self.reference_data = reference_data
        self.feature_importances = feature_importances
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.psi_threshold = psi_threshold
    
    def detect_all_drift(
        self,
        current_data: pd.DataFrame
    ) -> Dict[str, DriftResult]:
        """
        Detect drift across all features.
        
        Args:
            current_data: Current/production dataset
        
        Returns:
            Dictionary mapping feature names to DriftResult objects
        """
        # TODO: Implement complete drift detection
        # 1. For each numerical feature:
        #    - Calculate PSI
        #    - Classify severity
        #    - Calculate weighted score
        #    - Create DriftResult
        # 2. For each categorical feature:
        #    - Calculate categorical drift
        #    - Classify severity
        #    - Calculate weighted score
        #    - Create DriftResult
        # 3. Return dictionary of all results
        
        # Your code here
        pass
    
    def get_summary(
        self,
        current_data: pd.DataFrame
    ) -> Dict:
        """
        Get summary statistics and recommendations.
        
        Returns dict with:
            - total_features: int
            - features_with_drift: int
            - drift_by_severity: Dict[str, List[str]]
            - top_concerns: List of (feature, weighted_score) tuples
            - recommendation: str
        """
        # TODO: Implement summary generation
        # 1. Call detect_all_drift
        # 2. Group features by severity
        # 3. Sort by weighted score to find top concerns
        # 4. Generate recommendation based on drift extent
        
        # Your code here
        pass
    
    def _generate_recommendation(
        self,
        results: Dict[str, DriftResult]
    ) -> str:
        """Generate action recommendation based on drift results."""
        # TODO: Implement recommendation logic
        # Consider:
        # - Number of features with drift
        # - Severity distribution
        # - Importance of drifted features
        # Return one of:
        #   "continue_monitoring"
        #   "investigate_features"  
        #   "consider_retraining"
        #   "retrain_urgently"
        
        # Your code here
        pass


# === TODO 6: Generate Diagnostic Report ===
# Create a formatted report summarizing drift analysis.

def generate_diagnostic_report(
    detector: DriftDetector,
    current_data: pd.DataFrame,
    reference_data: pd.DataFrame
) -> str:
    """
    Generate a markdown-formatted diagnostic report.
    
    Args:
        detector: Configured DriftDetector instance
        current_data: Current/production dataset
        reference_data: Reference/training dataset
    
    Returns:
        Markdown-formatted report string
    """
    # TODO: Implement report generation
    # Include sections for:
    # 1. Executive Summary (overall status, key findings)
    # 2. Feature-Level Analysis (table with all features)
    # 3. Top Concerns (most impactful drift)
    # 4. Recommendations (action items)
    #
    # Use markdown formatting:
    # - Headers with ##
    # - Tables with | syntax
    # - Bold for emphasis with **
    
    # Your code here
    pass


# === VISUALIZATION (provided helper) ===

def plot_drift_comparison(
    reference: np.ndarray,
    current: np.ndarray,
    feature_name: str,
    psi_score: float,
    save_path: Optional[str] = None
) -> None:
    """
    Create visualization comparing reference and current distributions.
    
    Args:
        reference: Reference distribution values
        current: Current distribution values
        feature_name: Name of the feature
        psi_score: Calculated PSI score
        save_path: Optional path to save the figure
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Histogram comparison
    axes[0].hist(reference, bins=30, alpha=0.5, label='Reference', density=True, color='blue')
    axes[0].hist(current, bins=30, alpha=0.5, label='Current', density=True, color='orange')
    axes[0].set_title(f'{feature_name} Distribution Comparison')
    axes[0].set_xlabel('Value')
    axes[0].set_ylabel('Density')
    axes[0].legend()
    
    # PSI gauge
    if psi_score < 0.1:
        color = 'green'
        status = 'Stable'
    elif psi_score < 0.2:
        color = 'orange'
        status = 'Moderate Drift'
    else:
        color = 'red'
        status = 'Significant Drift'
    
    axes[1].barh(['PSI'], [psi_score], color=color, height=0.3)
    axes[1].axvline(x=0.1, color='orange', linestyle='--', label='Warning (0.1)')
    axes[1].axvline(x=0.2, color='red', linestyle='--', label='Critical (0.2)')
    axes[1].set_xlim(0, max(0.35, psi_score * 1.2))
    axes[1].set_title(f'PSI: {psi_score:.4f} ({status})')
    axes[1].legend(loc='lower right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


# === EXPECTED OUTPUT ===
# When you run this lab successfully, you should see output similar to:
#
# === Drift Detection Lab ===
#
# Loading sample data...
# Reference: 5000 samples, Production: 2000 samples
#
# === Drift Analysis Results ===
#
# Feature Analysis:
# +-----------------+--------+----------+-----------+---------+
# | Feature         | PSI    | Severity | Importance| Weighted|
# +-----------------+--------+----------+-----------+---------+
# | age             | 0.24   | HIGH     | 0.15      | 3.60    |
# | income          | 0.12   | LOW      | 0.25      | 3.00    |
# | credit_score    | 0.02   | NONE     | 0.30      | 0.60    |
# | loan_amount     | 0.08   | NONE     | 0.12      | 0.96    |
# | employment_years| 0.03   | NONE     | 0.08      | 0.24    |
# | employment_type | 0.31   | CRITICAL | 0.06      | 1.86    |
# | loan_purpose    | 0.18   | MODERATE | 0.04      | 0.72    |
# +-----------------+--------+----------+-----------+---------+
#
# Top Concerns (by weighted score):
#   1. age: weighted_score=3.60, severity=HIGH
#   2. income: weighted_score=3.00, severity=LOW
#   3. employment_type: weighted_score=1.86, severity=CRITICAL
#
# Recommendation: consider_retraining
#
# ✅ All checks passed!


# === SELF-CHECK ===

def run_self_check():
    """Validate your implementation."""
    print("=" * 60)
    print("Running Self-Check...")
    print("=" * 60)
    
    errors = []
    reference, production = generate_sample_data()
    
    # Test 1: PSI Calculation
    print("\n[Test 1] PSI Calculation...")
    try:
        psi, details = calculate_psi(
            reference['age'].values,
            production['age'].values,
            n_bins=10
        )
        assert psi is not None, "calculate_psi returned None"
        assert isinstance(psi, float), "PSI should be a float"
        assert 0 <= psi <= 1, f"PSI {psi} out of expected range [0, 1]"
        assert 'bin_edges' in details, "Details missing 'bin_edges'"
        assert 'reference_pct' in details, "Details missing 'reference_pct'"
        # Age should show significant drift (PSI > 0.15)
        assert psi > 0.1, f"Age PSI {psi:.3f} should be > 0.1 (drift was induced)"
        print(f"   ✓ PSI for 'age': {psi:.4f}")
    except Exception as e:
        errors.append(f"PSI calculation failed: {e}")
        print(f"   ✗ Error: {e}")
    
    # Test 2: Categorical Drift
    print("\n[Test 2] Categorical Drift Detection...")
    try:
        score, details = calculate_categorical_drift(
            reference['employment_type'],
            production['employment_type']
        )
        assert score is not None, "calculate_categorical_drift returned None"
        assert isinstance(score, float), "Score should be a float"
        assert 'chi2_statistic' in details, "Details missing 'chi2_statistic'"
        assert 'p_value' in details, "Details missing 'p_value'"
        # Employment type should show drift
        assert score > 0.1, f"Employment type score {score:.3f} should indicate drift"
        print(f"   ✓ Categorical drift score for 'employment_type': {score:.4f}")
        print(f"   ✓ Chi-square p-value: {details['p_value']:.6f}")
    except Exception as e:
        errors.append(f"Categorical drift detection failed: {e}")
        print(f"   ✗ Error: {e}")
    
    # Test 3: Severity Classification
    print("\n[Test 3] Severity Classification...")
    try:
        test_cases = [
            (0.05, 'numerical', DriftSeverity.NONE),
            (0.12, 'numerical', DriftSeverity.LOW),
            (0.17, 'numerical', DriftSeverity.MODERATE),
            (0.22, 'numerical', DriftSeverity.HIGH),
            (0.30, 'numerical', DriftSeverity.CRITICAL),
        ]
        for score, ftype, expected in test_cases:
            result = classify_severity(score, ftype)
            assert result is not None, f"classify_severity returned None for score={score}"
            assert result == expected, f"For score {score}: expected {expected}, got {result}"
        print(f"   ✓ All severity classifications correct")
    except Exception as e:
        errors.append(f"Severity classification failed: {e}")
        print(f"   ✗ Error: {e}")
    
    # Test 4: Weighted Drift
    print("\n[Test 4] Importance-Weighted Drift...")
    try:
        weighted = calculate_weighted_drift(0.2, 0.3, FEATURE_IMPORTANCES)
        assert weighted is not None, "calculate_weighted_drift returned None"
        assert isinstance(weighted, float), "Weighted score should be a float"
        assert weighted > 0, "Weighted score should be positive for non-zero inputs"
        # Higher importance should give higher weighted score
        weighted_high = calculate_weighted_drift(0.2, 0.3, FEATURE_IMPORTANCES)
        weighted_low = calculate_weighted_drift(0.2, 0.05, FEATURE_IMPORTANCES)
        assert weighted_high > weighted_low, "Higher importance should give higher weighted score"
        print(f"   ✓ Weighted score (importance=0.3): {weighted_high:.2f}")
        print(f"   ✓ Weighted score (importance=0.05): {weighted_low:.2f}")
    except Exception as e:
        errors.append(f"Weighted drift calculation failed: {e}")
        print(f"   ✗ Error: {e}")
    
    # Test 5: Complete Drift Detector
    print("\n[Test 5] Complete Drift Detector...")
    try:
        detector = DriftDetector(
            reference_data=reference,
            feature_importances=FEATURE_IMPORTANCES,
            numerical_features=NUMERICAL_FEATURES,
            categorical_features=CATEGORICAL_FEATURES
        )
        results = detector.detect_all_drift(production)
        
        assert results is not None, "detect_all_drift returned None"
        assert len(results) == 7, f"Expected 7 features, got {len(results)}"
        
        # Check result structure
        sample_result = list(results.values())[0]
        assert isinstance(sample_result, DriftResult), "Results should be DriftResult objects"
        assert hasattr(sample_result, 'severity'), "DriftResult missing 'severity'"
        assert hasattr(sample_result, 'weighted_score'), "DriftResult missing 'weighted_score'"
        
        print(f"   ✓ Detected drift across {len(results)} features")
        
        # Test summary
        summary = detector.get_summary(production)
        assert summary is not None, "get_summary returned None"
        assert 'features_with_drift' in summary, "Summary missing 'features_with_drift'"
        assert 'recommendation' in summary, "Summary missing 'recommendation'"
        print(f"   ✓ Summary: {summary['features_with_drift']} features with drift")
        print(f"   ✓ Recommendation: {summary['recommendation']}")
    except Exception as e:
        errors.append(f"Drift detector failed: {e}")
        print(f"   ✗ Error: {e}")
    
    # Test 6: Diagnostic Report
    print("\n[Test 6] Diagnostic Report Generation...")
    try:
        detector = DriftDetector(
            reference_data=reference,
            feature_importances=FEATURE_IMPORTANCES,
            numerical_features=NUMERICAL_FEATURES,
            categorical_features=CATEGORICAL_FEATURES
        )
        report = generate_diagnostic_report(detector, production, reference)
        
        assert report is not None, "generate_diagnostic_report returned None"
        assert isinstance(report, str), "Report should be a string"
        assert len(report) > 200, "Report seems too short"
        assert "##" in report, "Report should use markdown headers"
        assert "age" in report.lower(), "Report should mention features"
        
        print(f"   ✓ Generated report with {len(report)} characters")
        print(f"   ✓ Report preview: {report[:100]}...")
    except Exception as e:
        errors.append(f"Diagnostic report generation failed: {e}")
        print(f"   ✗ Error: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    if errors:
        print(f"❌ {len(errors)} check(s) failed:")
        for err in errors:
            print(f"   - {err}")
        print("\nFix the errors above and run again.")
    else:
        print("✅ All checks passed!")
        print("\nYour drift detection implementation is working correctly.")
        print("\nNext steps for Final Project Component 4:")
        print("  1. Apply this to your actual model's training data")
        print("  2. Generate visualizations using plot_drift_comparison()")
        print("  3. Write a 1-2 page diagnostic report")
        print("  4. Include recommendations for retraining or intervention")
    print("=" * 60)
    
    return len(errors) == 0


# === MAIN ===

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Module 8 Lab: Drift Detection Fundamentals")
    print("=" * 60)
    
    # Load sample data
    print("\nLoading sample data...")
    reference, production = generate_sample_data()
    print(f"Reference: {len(reference)} samples, Production: {len(production)} samples")
    
    # Run the self-check
    success = run_self_check()
    
    if success:
        # Demo complete analysis
        print("\n" + "=" * 60)
        print("Running Full Demo...")
        print("=" * 60)
        
        detector = DriftDetector(
            reference_data=reference,
            feature_importances=FEATURE_IMPORTANCES,
            numerical_features=NUMERICAL_FEATURES,
            categorical_features=CATEGORICAL_FEATURES
        )
        
        results = detector.detect_all_drift(production)
        
        print("\n--- Feature-Level Drift Analysis ---")
        print(f"{'Feature':<18} {'PSI':<8} {'Severity':<10} {'Import.':<10} {'Weighted':<8}")
        print("-" * 54)
        
        # Sort by weighted score for display
        sorted_results = sorted(
            results.items(),
            key=lambda x: x[1].weighted_score,
            reverse=True
        )
        
        for feature, result in sorted_results:
            print(f"{feature:<18} {result.drift_score:<8.4f} "
                  f"{result.severity.value:<10} "
                  f"{result.importance_weight:<10.2f} "
                  f"{result.weighted_score:<8.2f}")
        
        print("\n--- Summary ---")
        summary = detector.get_summary(production)
        print(f"Features with drift: {summary['features_with_drift']} / {summary['total_features']}")
        print(f"Recommendation: {summary['recommendation']}")
        
        if summary.get('top_concerns'):
            print("\nTop 3 Concerns:")
            for i, (feature, score) in enumerate(summary['top_concerns'][:3], 1):
                print(f"  {i}. {feature}: weighted_score={score:.2f}")
        
        # Generate report
        print("\n--- Diagnostic Report ---")
        report = generate_diagnostic_report(detector, production, reference)
        print(report[:500] + "..." if len(report) > 500 else report)
        
        # Optional: Generate visualizations
        print("\n--- Generating Visualizations ---")
        try:
            for feature in ['age', 'income']:
                psi, _ = calculate_psi(
                    reference[feature].values,
                    production[feature].values
                )
                plot_drift_comparison(
                    reference[feature].values,
                    production[feature].values,
                    feature,
                    psi,
                    save_path=f"drift_{feature}.png"
                )
                print(f"   ✓ Saved drift_{feature}.png")
        except ImportError:
            print("   (matplotlib not available for visualizations)")
        except Exception as e:
            print(f"   Visualization skipped: {e}")