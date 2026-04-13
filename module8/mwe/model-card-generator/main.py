"""
Model Card Generator MWE - Programmatic Model Documentation

Demonstrates building model cards as structured data and exporting
them in multiple formats:

  1. ModelCard dataclass — structured representation of model metadata
  2. Subgroup metrics — disaggregated performance across segments
  3. Markdown export — human-readable documentation artifact
  4. JSON export — machine-readable format for governance pipelines
  5. Completeness validation — check that required sections are filled

No external dependencies required (stdlib only).

Run: python main.py
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import json


# =============================================================================
# Model Card Data Structure
# =============================================================================

@dataclass
class ModelCard:
    """Structured model card for ML governance."""

    # Model Details
    name: str
    version: str
    model_type: str
    framework: str
    owner: str
    contact: str = ""
    last_updated: datetime = field(default_factory=datetime.now)

    # Intended Use
    primary_uses: List[str] = field(default_factory=list)
    intended_users: List[str] = field(default_factory=list)
    out_of_scope_uses: List[str] = field(default_factory=list)

    # Training Data
    dataset_description: str = ""
    dataset_size: int = 0
    feature_count: int = 0
    label_description: str = ""
    class_balance: str = ""
    data_limitations: List[str] = field(default_factory=list)

    # Performance Metrics
    metrics: Dict[str, float] = field(default_factory=dict)
    subgroup_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    confidence_intervals: Dict[str, Tuple[float, float]] = field(default_factory=dict)

    # Limitations & Ethics
    limitations: List[str] = field(default_factory=list)
    potential_harms: List[str] = field(default_factory=list)
    mitigation_strategies: List[str] = field(default_factory=list)

    # Caveats
    recommendations: List[str] = field(default_factory=list)

    def to_markdown(self) -> str:
        """Export model card as a complete markdown document."""
        lines = [
            f"# Model Card: {self.name}",
            "",
            "## Model Details",
            "",
            f"- **Model Name:** {self.name}",
            f"- **Version:** {self.version}",
            f"- **Type:** {self.model_type}",
            f"- **Framework:** {self.framework}",
            f"- **Owner:** {self.owner}",
        ]

        if self.contact:
            lines.append(f"- **Contact:** {self.contact}")
        lines.append(
            f"- **Last Updated:** {self.last_updated.strftime('%Y-%m-%d')}"
        )

        # Intended Use
        lines.extend(["", "## Intended Use", ""])
        if self.primary_uses:
            lines.append("### Primary Use Cases")
            lines.append("")
            lines.extend(f"- {use}" for use in self.primary_uses)

        if self.intended_users:
            lines.extend(["", "### Intended Users", ""])
            lines.extend(f"- {user}" for user in self.intended_users)

        if self.out_of_scope_uses:
            lines.extend(["", "### Out-of-Scope Uses", ""])
            lines.extend(f"- {use}" for use in self.out_of_scope_uses)

        # Training Data
        if self.dataset_description:
            lines.extend(["", "## Training Data", ""])
            lines.append(
                "| Attribute | Value |"
            )
            lines.append("|-----------|-------|")
            lines.append(
                f"| Dataset | {self.dataset_description} |"
            )
            if self.dataset_size:
                lines.append(
                    f"| Size | {self.dataset_size:,} records |"
                )
            if self.feature_count:
                lines.append(
                    f"| Features | {self.feature_count} |"
                )
            if self.label_description:
                lines.append(
                    f"| Label | {self.label_description} |"
                )
            if self.class_balance:
                lines.append(
                    f"| Class Balance | {self.class_balance} |"
                )

        if self.data_limitations:
            lines.extend(["", "### Known Data Limitations", ""])
            lines.extend(
                f"- {lim}" for lim in self.data_limitations
            )

        # Performance Metrics
        if self.metrics:
            lines.extend(["", "## Performance Metrics", ""])

            if self.subgroup_metrics:
                # Build table with subgroup columns
                subgroups = list(self.subgroup_metrics.keys())
                header = "| Metric | Overall | " + " | ".join(subgroups) + " |"
                sep = "|--------|---------|" + "|".join(
                    "-" * (len(sg) + 2) for sg in subgroups
                ) + "|"
                lines.append(header)
                lines.append(sep)

                for metric, overall_val in self.metrics.items():
                    row = f"| {metric} | {overall_val:.3f} |"
                    for sg in subgroups:
                        sg_val = self.subgroup_metrics[sg].get(metric, 0)
                        row += f" {sg_val:.3f} |"
                    lines.append(row)
            else:
                lines.append("| Metric | Value |")
                lines.append("|--------|-------|")
                for metric, val in self.metrics.items():
                    lines.append(f"| {metric} | {val:.3f} |")

        if self.confidence_intervals:
            lines.extend(["", "### Confidence Intervals (95%)", ""])
            for metric, (lo, hi) in self.confidence_intervals.items():
                lines.append(f"- {metric}: [{lo:.3f}, {hi:.3f}]")

        # Limitations
        if self.limitations:
            lines.extend(["", "## Limitations", ""])
            lines.extend(f"- {lim}" for lim in self.limitations)

        # Ethics
        if self.potential_harms or self.mitigation_strategies:
            lines.extend(["", "## Ethical Considerations", ""])

        if self.potential_harms:
            lines.extend(["### Potential Harms", ""])
            lines.extend(f"- {harm}" for harm in self.potential_harms)

        if self.mitigation_strategies:
            lines.extend(["", "### Mitigation Strategies", ""])
            lines.extend(
                f"- {strategy}" for strategy in self.mitigation_strategies
            )

        # Recommendations
        if self.recommendations:
            lines.extend(["", "## Caveats and Recommendations", ""])
            for i, rec in enumerate(self.recommendations, 1):
                lines.append(f"{i}. {rec}")

        return "\n".join(lines)

    def to_json(self) -> str:
        """Export model card as JSON for programmatic use."""
        data = {
            "model_details": {
                "name": self.name,
                "version": self.version,
                "type": self.model_type,
                "framework": self.framework,
                "owner": self.owner,
                "contact": self.contact,
                "last_updated": self.last_updated.strftime("%Y-%m-%d"),
            },
            "intended_use": {
                "primary_uses": self.primary_uses,
                "intended_users": self.intended_users,
                "out_of_scope_uses": self.out_of_scope_uses,
            },
            "training_data": {
                "description": self.dataset_description,
                "size": self.dataset_size,
                "features": self.feature_count,
                "limitations": self.data_limitations,
            },
            "performance": {
                "overall_metrics": self.metrics,
                "subgroup_metrics": self.subgroup_metrics,
                "confidence_intervals": {
                    k: {"lower": v[0], "upper": v[1]}
                    for k, v in self.confidence_intervals.items()
                },
            },
            "limitations": self.limitations,
            "ethical_considerations": {
                "potential_harms": self.potential_harms,
                "mitigation_strategies": self.mitigation_strategies,
            },
            "recommendations": self.recommendations,
        }
        return json.dumps(data, indent=2)

    def validate_completeness(self) -> Dict[str, bool]:
        """Check that all required model card sections are filled."""
        checks = {
            "model_details": bool(
                self.name and self.version and self.model_type
            ),
            "intended_use": bool(self.primary_uses),
            "out_of_scope": bool(self.out_of_scope_uses),
            "training_data": bool(self.dataset_description),
            "performance_metrics": bool(self.metrics),
            "subgroup_analysis": bool(self.subgroup_metrics),
            "limitations": bool(self.limitations),
            "ethical_considerations": bool(
                self.potential_harms and self.mitigation_strategies
            ),
            "recommendations": bool(self.recommendations),
        }
        return checks


# =============================================================================
# Demonstrations
# =============================================================================

def build_example_card() -> ModelCard:
    """Create a fully populated example model card."""
    return ModelCard(
        name="Customer Churn Predictor",
        version="2.1.0",
        model_type="Binary Classification",
        framework="scikit-learn (Random Forest)",
        owner="Data Science Team",
        contact="ml-team@company.com",
        last_updated=datetime(2024, 1, 15),
        # Intended use
        primary_uses=[
            "Identify customers at risk of churning within 30 days",
            "Prioritize retention outreach by customer success team",
            "Segment marketing campaigns based on churn risk",
        ],
        intended_users=[
            "Customer Success Managers (via dashboard)",
            "Marketing team (via API integration)",
            "Business Intelligence for reporting",
        ],
        out_of_scope_uses=[
            "Individual credit decisions",
            "Automated account termination",
            "Use without human review for high-value accounts",
        ],
        # Training data
        dataset_description="Internal CRM data, Jan 2020 - Dec 2023",
        dataset_size=450000,
        feature_count=47,
        label_description="Churned within 30 days (binary)",
        class_balance="8.2% positive (churned)",
        data_limitations=[
            "Enterprise segment underrepresented (< 5% of training data)",
            "Missing data for customers acquired via partner channel",
            "Historical bias: churned customers from 2020 may differ from current",
        ],
        # Performance
        metrics={
            "AUC-ROC": 0.847,
            "Precision@50%recall": 0.680,
            "F1": 0.720,
        },
        subgroup_metrics={
            "Enterprise": {
                "AUC-ROC": 0.792,
                "Precision@50%recall": 0.540,
                "F1": 0.610,
            },
            "SMB": {
                "AUC-ROC": 0.861,
                "Precision@50%recall": 0.710,
                "F1": 0.740,
            },
            "Consumer": {
                "AUC-ROC": 0.853,
                "Precision@50%recall": 0.690,
                "F1": 0.730,
            },
        },
        confidence_intervals={
            "AUC-ROC": (0.841, 0.853),
            "Precision": (0.650, 0.710),
        },
        # Limitations
        limitations=[
            "Significantly lower precision for enterprise segment",
            "Temporal validity: trained on pre-2024 data; major product changes "
            "may affect predictions",
            "Requires 30+ days of usage history; inaccurate for new customers",
            "Probability outputs are not well-calibrated; use for ranking, not "
            "absolute probabilities",
        ],
        # Ethics
        potential_harms=[
            "Preferential treatment for low-risk customers may disadvantage "
            "at-risk users",
            "Demographic features (age, location) may proxy for protected "
            "characteristics",
            "Retention offers triggered by model could create perverse incentives",
        ],
        mitigation_strategies=[
            "Regular fairness audits across demographic segments",
            "Human review required for accounts over $50K ARR",
            "Quarterly recalibration using recent churn data",
            "Transparency: customers can request explanation of retention offers",
        ],
        # Recommendations
        recommendations=[
            "Do not use probability scores as absolute churn likelihood",
            "Retrain quarterly to account for behavioral drift",
            "Monitor prediction distribution weekly for anomalies",
            "Combine with qualitative signals from customer success team",
        ],
    )


def demo_build_card():
    """Demo 1: Build a model card from structured data."""
    print("=" * 60)
    print("Demo 1: Building a Model Card")
    print("=" * 60)

    card = build_example_card()

    print(f"\n  Model: {card.name} v{card.version}")
    print(f"  Type: {card.model_type}")
    print(f"  Framework: {card.framework}")
    print(f"  Owner: {card.owner}")
    print(f"  Training data: {card.dataset_size:,} records, {card.feature_count} features")
    print(f"  Primary uses: {len(card.primary_uses)}")
    print(f"  Limitations: {len(card.limitations)}")
    print(f"  Subgroups analyzed: {list(card.subgroup_metrics.keys())}")

    return card


def demo_markdown_export(card: ModelCard):
    """Demo 2: Export model card as markdown."""
    print("\n" + "=" * 60)
    print("Demo 2: Markdown Export")
    print("=" * 60)

    markdown = card.to_markdown()
    lines = markdown.split("\n")

    print(f"\n  Generated {len(lines)} lines of markdown documentation")
    print(f"  Document size: {len(markdown)} characters\n")

    # Show first 40 lines
    max_show = 40
    print("  --- Preview (first 40 lines) ---\n")
    for line in lines[:max_show]:
        print(f"  {line}")
    if len(lines) > max_show:
        print(f"\n  ... ({len(lines) - max_show} more lines)")


def demo_json_export(card: ModelCard):
    """Demo 3: Export model card as JSON."""
    print("\n" + "=" * 60)
    print("Demo 3: JSON Export")
    print("=" * 60)

    json_output = card.to_json()
    parsed = json.loads(json_output)

    print(f"\n  JSON output: {len(json_output)} characters")
    print(f"  Top-level keys: {list(parsed.keys())}")

    # Show performance section
    print(f"\n  Performance metrics (from JSON):")
    for metric, value in parsed["performance"]["overall_metrics"].items():
        print(f"    {metric}: {value}")

    print(f"\n  Subgroup metrics:")
    for group, metrics in parsed["performance"]["subgroup_metrics"].items():
        auc = metrics.get("AUC-ROC", 0)
        print(f"    {group}: AUC-ROC = {auc:.3f}")

    print(f"\n  Confidence intervals:")
    for metric, ci in parsed["performance"]["confidence_intervals"].items():
        print(f"    {metric}: [{ci['lower']:.3f}, {ci['upper']:.3f}]")


def demo_validate(card: ModelCard):
    """Demo 4: Validate model card completeness."""
    print("\n" + "=" * 60)
    print("Demo 4: Completeness Validation")
    print("=" * 60)

    checks = card.validate_completeness()

    print(f"\n  {'Section':<25} {'Status':>8}")
    print(f"  {'—' * 25} {'—' * 8}")

    all_pass = True
    for section, passed in checks.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {section:<25} {status:>8}")
        if not passed:
            all_pass = False

    total = len(checks)
    passed_count = sum(checks.values())
    print(f"\n  Result: {passed_count}/{total} sections complete")
    if all_pass:
        print("  Status: Model card is complete and ready for review")
    else:
        print("  Status: Model card has missing sections — fill before deployment")


def demo_incomplete_card():
    """Demo 5: Show validation failure for an incomplete card."""
    print("\n" + "=" * 60)
    print("Demo 5: Incomplete Card Validation")
    print("=" * 60)

    incomplete = ModelCard(
        name="Quick Prototype",
        version="0.1.0",
        model_type="Regression",
        framework="PyTorch",
        owner="Research Team",
        metrics={"MSE": 0.034, "R2": 0.91},
    )

    print(f"\n  Validating an incomplete model card for '{incomplete.name}':\n")

    checks = incomplete.validate_completeness()

    print(f"  {'Section':<25} {'Status':>8}")
    print(f"  {'—' * 25} {'—' * 8}")

    for section, passed in checks.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {section:<25} {status:>8}")

    missing = [s for s, p in checks.items() if not p]
    print(f"\n  Missing sections: {missing}")
    print("  Action: Fill these sections before deploying to production")


def demo_subgroup_analysis(card: ModelCard):
    """Demo 6: Highlight subgroup performance disparities."""
    print("\n" + "=" * 60)
    print("Demo 6: Subgroup Performance Analysis")
    print("=" * 60)

    print(f"\n  Analyzing performance disparities across subgroups:\n")

    if not card.subgroup_metrics:
        print("  No subgroup metrics available.")
        return

    # Find best and worst performing subgroups for each metric
    for metric in card.metrics:
        overall = card.metrics[metric]
        print(f"  {metric}:")
        print(f"    Overall: {overall:.3f}")

        subgroup_vals = {}
        for sg, sg_metrics in card.subgroup_metrics.items():
            val = sg_metrics.get(metric, 0)
            subgroup_vals[sg] = val
            gap = val - overall
            flag = "  ← BELOW" if gap < -0.05 else ""
            print(f"    {sg:>12}: {val:.3f} (gap: {gap:+.3f}){flag}")

        best = max(subgroup_vals, key=subgroup_vals.get)
        worst = min(subgroup_vals, key=subgroup_vals.get)
        spread = subgroup_vals[best] - subgroup_vals[worst]
        print(f"    Spread: {spread:.3f} (best: {best}, worst: {worst})")
        print()

    print("  Key insight: Disaggregated metrics reveal performance")
    print("  disparities hidden by overall averages.")


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("Model Card Generator MWE — Programmatic ML Documentation")
    print("=" * 60)

    card = demo_build_card()
    demo_markdown_export(card)
    demo_json_export(card)
    demo_validate(card)
    demo_incomplete_card()
    demo_subgroup_analysis(card)

    print("=" * 60)
    print("MWE Complete — Model card generation demonstrated!")
    print("=" * 60)


if __name__ == "__main__":
    main()
