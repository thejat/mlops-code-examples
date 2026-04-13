"""
Risk Scoring MWE - NIST AI RMF-Style Risk Assessment

Demonstrates AI risk management for ML and LLM/RAG systems:

  1. Risk entry definition — structured risk descriptions with scoring
  2. Risk matrix — likelihood × severity scoring with thresholds
  3. Risk register — complete register with owners and triggers
  4. Treatment recommendations — mitigate/accept/transfer/avoid decisions
  5. Deployment gate — go/no-go decision based on risk profile
  6. LLM/RAG risks — additional risk entries for generative AI systems

No external dependencies required (stdlib only).

Run: python main.py
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional
from datetime import datetime


# =============================================================================
# Risk Framework Data Structures
# =============================================================================

class Likelihood(Enum):
    """Likelihood of a risk materializing."""
    RARE = 1
    UNLIKELY = 2
    POSSIBLE = 3
    LIKELY = 4
    ALMOST_CERTAIN = 5


class Severity(Enum):
    """Severity of impact if risk materializes."""
    NEGLIGIBLE = 1
    MINOR = 2
    MODERATE = 3
    MAJOR = 4
    CRITICAL = 5


class Treatment(Enum):
    """Risk treatment strategy."""
    MITIGATE = "mitigate"
    ACCEPT = "accept"
    TRANSFER = "transfer"
    AVOID = "avoid"


class RiskLevel(Enum):
    """Overall risk level from the risk matrix."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class RiskEntry:
    """A single risk in the risk register."""
    risk_id: str
    title: str
    description: str
    category: str  # Bias, Security, Data Quality, Accountability, Compliance
    likelihood: Likelihood
    severity: Severity
    treatment: Treatment
    controls: List[str]
    owner: str
    review_trigger: str
    residual_likelihood: Optional[Likelihood] = None
    residual_severity: Optional[Severity] = None
    notes: str = ""


# =============================================================================
# Risk Matrix
# =============================================================================

def compute_risk_score(likelihood: Likelihood, severity: Severity) -> int:
    """Compute risk score as likelihood × severity."""
    return likelihood.value * severity.value


def classify_risk_level(score: int) -> RiskLevel:
    """Classify risk level from numeric score."""
    if score <= 4:
        return RiskLevel.LOW
    elif score <= 9:
        return RiskLevel.MEDIUM
    elif score <= 15:
        return RiskLevel.HIGH
    else:
        return RiskLevel.CRITICAL


def format_risk_matrix() -> str:
    """Generate a text-based risk matrix."""
    lines = [
        "                          SEVERITY",
        "              ┌──────┬──────┬──────┬──────┬──────┐",
        "              │  1   │  2   │  3   │  4   │  5   │",
        "              │Negl. │Minor │Mod.  │Major │Crit. │",
        "        ┌─────┼──────┼──────┼──────┼──────┼──────┤",
    ]

    likelihood_labels = [
        (5, "A.Cert"),
        (4, "Likely"),
        (3, "Poss. "),
        (2, "Unlik."),
        (1, "Rare  "),
    ]

    for lv, label in likelihood_labels:
        row = f"        │  {lv}  │"
        for sv in range(1, 6):
            score = lv * sv
            level = classify_risk_level(score)
            cell = f"{score:>2} {level.value[:3]:>3}"
            row += f"{cell:>6}│"

        if lv == 5:
            lines.append(f"  LIKE- {row}")
        elif lv == 4:
            lines.append(f"  LIHOOD{row}")
        else:
            lines.append(f"        {row}")

        if lv > 1:
            lines.append(
                "        ├─────┼──────┼──────┼──────┼──────┼──────┤"
            )
        else:
            lines.append(
                "        └─────┴──────┴──────┴──────┴──────┴──────┘"
            )

    return "\n".join(lines)


# =============================================================================
# Example Risk Registers
# =============================================================================

def build_ml_risk_register() -> List[RiskEntry]:
    """Build example risk register for a traditional ML system."""
    return [
        RiskEntry(
            risk_id="R001",
            title="Demographic bias in approvals",
            description=(
                "Model approval rates differ significantly across "
                "demographic groups, violating the 80% rule"
            ),
            category="Bias",
            likelihood=Likelihood.POSSIBLE,
            severity=Severity.MAJOR,
            treatment=Treatment.MITIGATE,
            controls=[
                "Quarterly fairness audits with disparate impact analysis",
                "Alert when demographic parity ratio drops below 0.8",
                "Human appeal path for denied applicants",
            ],
            owner="ML Team Lead",
            review_trigger="Fairness ratio < 0.8 on any protected group",
            residual_likelihood=Likelihood.UNLIKELY,
            residual_severity=Severity.MODERATE,
        ),
        RiskEntry(
            risk_id="R002",
            title="Feature drift degrades predictions",
            description=(
                "Input feature distributions shift from training data, "
                "causing silent model performance degradation"
            ),
            category="Data Quality",
            likelihood=Likelihood.LIKELY,
            severity=Severity.MODERATE,
            treatment=Treatment.MITIGATE,
            controls=[
                "PSI monitoring on all input features (threshold: 0.2)",
                "Automated retraining pipeline on drift detection",
                "Weekly drift report to model owners",
            ],
            owner="ML Platform Engineer",
            review_trigger="PSI > 0.2 on any feature with importance > 0.1",
            residual_likelihood=Likelihood.POSSIBLE,
            residual_severity=Severity.MINOR,
        ),
        RiskEntry(
            risk_id="R003",
            title="Model used for unintended purpose",
            description=(
                "Churn prediction model used for credit scoring or "
                "automated account termination without human review"
            ),
            category="Accountability",
            likelihood=Likelihood.UNLIKELY,
            severity=Severity.CRITICAL,
            treatment=Treatment.MITIGATE,
            controls=[
                "Model card documents out-of-scope uses explicitly",
                "API access controls restrict unauthorized consumers",
                "Audit trail logs all prediction requests with use context",
            ],
            owner="Product Manager",
            review_trigger="New API consumer onboarded or use case change request",
        ),
        RiskEntry(
            risk_id="R004",
            title="No manual override for high-stakes decisions",
            description=(
                "Automated decisions cannot be overridden by humans, "
                "removing accountability for edge cases"
            ),
            category="Accountability",
            likelihood=Likelihood.POSSIBLE,
            severity=Severity.MAJOR,
            treatment=Treatment.MITIGATE,
            controls=[
                "Human review required for accounts over $50K ARR",
                "Kill switch to disable model and fall back to rules",
                "Override audit trail with justification field",
            ],
            owner="Operations Director",
            review_trigger="Review queue misses SLA or kill switch activated",
        ),
        RiskEntry(
            risk_id="R005",
            title="Vendor changes schema or terms",
            description=(
                "External data vendor changes data format or usage "
                "terms, breaking pipeline or violating agreements"
            ),
            category="Compliance",
            likelihood=Likelihood.UNLIKELY,
            severity=Severity.MODERATE,
            treatment=Treatment.TRANSFER,
            controls=[
                "Vendor SLA with schema change notification clause",
                "Schema validation at data ingestion boundary",
                "Quarterly vendor compliance review",
            ],
            owner="Platform Eng Manager",
            review_trigger="Contract update notification or schema validation failure",
        ),
    ]


def build_llm_risk_register() -> List[RiskEntry]:
    """Build additional risk entries for LLM/RAG systems."""
    return [
        RiskEntry(
            risk_id="R006",
            title="Prompt injection bypasses safety controls",
            description=(
                "Adversarial user input manipulates the LLM into "
                "ignoring system prompt instructions or producing "
                "harmful outputs"
            ),
            category="Security",
            likelihood=Likelihood.POSSIBLE,
            severity=Severity.MAJOR,
            treatment=Treatment.MITIGATE,
            controls=[
                "Input validation and sanitization layer",
                "Output filtering for known harmful patterns",
                "Red-team testing before deployment and quarterly",
                "Log flagged interactions for review",
            ],
            owner="ML Platform Lead",
            review_trigger="Red-team finding or flagged interaction report",
            residual_likelihood=Likelihood.UNLIKELY,
            residual_severity=Severity.MODERATE,
        ),
        RiskEntry(
            risk_id="R007",
            title="Knowledge base staleness causes outdated responses",
            description=(
                "RAG knowledge base not refreshed after product or "
                "policy updates, leading to incorrect answers"
            ),
            category="Data Quality",
            likelihood=Likelihood.LIKELY,
            severity=Severity.MODERATE,
            treatment=Treatment.MITIGATE,
            controls=[
                "Automated ingestion schedule (weekly or on doc change)",
                "Document age distribution monitoring",
                "Alert when p95 document age exceeds 90 days",
            ],
            owner="Data Eng Lead",
            review_trigger="Document age p95 > 90 days or content update deployed",
            residual_likelihood=Likelihood.UNLIKELY,
            residual_severity=Severity.MINOR,
        ),
        RiskEntry(
            risk_id="R008",
            title="Sensitive data leaks through LLM context",
            description=(
                "PII or confidential information included in retrieved "
                "context is exposed in generated responses"
            ),
            category="Compliance",
            likelihood=Likelihood.POSSIBLE,
            severity=Severity.CRITICAL,
            treatment=Treatment.MITIGATE,
            controls=[
                "PII scrubbing at document ingestion time",
                "Access control on retrieval index by sensitivity level",
                "Output scanning for PII patterns before response",
                "Data classification for all indexed documents",
            ],
            owner="Security Lead",
            review_trigger="PII pattern detected in output log or user report",
            residual_likelihood=Likelihood.RARE,
            residual_severity=Severity.MAJOR,
        ),
    ]


# =============================================================================
# Demonstrations
# =============================================================================

def demo_risk_matrix():
    """Demo 1: Display the risk matrix."""
    print("=" * 60)
    print("Demo 1: Risk Matrix (Likelihood × Severity)")
    print("=" * 60)

    print(f"\n{format_risk_matrix()}")

    print(f"\n  Risk Level Thresholds:")
    print(f"    LOW:      1-4   (acceptable, monitor)")
    print(f"    MEDIUM:   5-9   (investigate, plan mitigation)")
    print(f"    HIGH:     10-15 (active mitigation required)")
    print(f"    CRITICAL: 16-25 (block deployment until addressed)")


def demo_risk_register(risks: List[RiskEntry], title: str):
    """Demo 2/3: Display a risk register table."""
    print("\n" + "=" * 60)
    print(f"Demo: {title}")
    print("=" * 60)

    header = (
        f"  {'ID':<5} {'Risk':<35} {'L×S':>5} {'Score':>6} "
        f"{'Level':>8} {'Treatment':>10}"
    )
    print(f"\n{header}")
    print(
        f"  {'—' * 5} {'—' * 35} {'—' * 5} {'—' * 6} {'—' * 8} {'—' * 10}"
    )

    for risk in risks:
        score = compute_risk_score(risk.likelihood, risk.severity)
        level = classify_risk_level(score)
        ls = f"{risk.likelihood.value}×{risk.severity.value}"
        print(
            f"  {risk.risk_id:<5} {risk.title[:35]:<35} {ls:>5} "
            f"{score:>6} {level.value:>8} {risk.treatment.value:>10}"
        )

    # Show details for highest-risk entry
    highest = max(
        risks,
        key=lambda r: compute_risk_score(r.likelihood, r.severity),
    )
    score = compute_risk_score(highest.likelihood, highest.severity)
    level = classify_risk_level(score)

    print(f"\n  Highest risk: {highest.risk_id} — {highest.title}")
    print(f"  Score: {score} ({level.value})")
    print(f"  Category: {highest.category}")
    print(f"  Owner: {highest.owner}")
    print(f"  Review trigger: {highest.review_trigger}")
    print(f"  Controls:")
    for ctrl in highest.controls:
        print(f"    • {ctrl}")


def demo_residual_risk(risks: List[RiskEntry]):
    """Demo 4: Show inherent vs residual risk after controls."""
    print("\n" + "=" * 60)
    print("Demo 4: Inherent vs Residual Risk")
    print("=" * 60)

    print(f"\n  After applying controls, residual risk should decrease:\n")

    header = (
        f"  {'ID':<5} {'Risk':<30} "
        f"{'Inherent':>9} {'Residual':>9} {'Reduction':>10}"
    )
    print(header)
    print(
        f"  {'—' * 5} {'—' * 30} {'—' * 9} {'—' * 9} {'—' * 10}"
    )

    for risk in risks:
        inherent = compute_risk_score(risk.likelihood, risk.severity)
        inherent_level = classify_risk_level(inherent)

        if risk.residual_likelihood and risk.residual_severity:
            residual = compute_risk_score(
                risk.residual_likelihood, risk.residual_severity
            )
            residual_level = classify_risk_level(residual)
            reduction = inherent - residual
            reduction_pct = (
                f"-{reduction} ({reduction / inherent:.0%})"
                if inherent > 0 else "N/A"
            )
        else:
            residual_level = inherent_level
            residual = inherent
            reduction_pct = "no data"

        print(
            f"  {risk.risk_id:<5} {risk.title[:30]:<30} "
            f"{inherent:>4} {inherent_level.value:>4} "
            f"{residual:>4} {residual_level.value:>4} "
            f"{reduction_pct:>10}"
        )


def demo_deployment_gate(risks: List[RiskEntry]):
    """Demo 5: Deployment gate decision based on risk profile."""
    print("\n" + "=" * 60)
    print("Demo 5: Deployment Gate Decision")
    print("=" * 60)

    print(f"\n  Evaluating {len(risks)} risks for deployment readiness:\n")

    # Classify all risks
    critical_risks = []
    high_risks = []
    unowned_risks = []
    no_controls = []

    for risk in risks:
        score = compute_risk_score(risk.likelihood, risk.severity)
        level = classify_risk_level(score)

        if level == RiskLevel.CRITICAL:
            critical_risks.append(risk)
        elif level == RiskLevel.HIGH:
            high_risks.append(risk)

        if not risk.owner:
            unowned_risks.append(risk)
        if not risk.controls:
            no_controls.append(risk)

    # Apply decision rules
    print(f"  {'Check':<45} {'Result':>10}")
    print(f"  {'—' * 45} {'—' * 10}")

    has_critical = len(critical_risks) > 0
    has_unowned = len(unowned_risks) > 0
    has_no_controls = len(no_controls) > 0
    high_without_residual = [
        r for r in high_risks
        if not (r.residual_likelihood and r.residual_severity)
    ]

    print(
        f"  {'No CRITICAL risks without mitigation':<45} "
        f"{'✗ FAIL' if has_critical else '✓ PASS':>10}"
    )
    print(
        f"  {'All risks have assigned owners':<45} "
        f"{'✗ FAIL' if has_unowned else '✓ PASS':>10}"
    )
    print(
        f"  {'All risks have defined controls':<45} "
        f"{'✗ FAIL' if has_no_controls else '✓ PASS':>10}"
    )
    print(
        f"  {'HIGH risks have residual assessment':<45} "
        f"{'✗ FAIL' if high_without_residual else '✓ PASS':>10}"
    )

    # Decision
    if has_critical or has_unowned or has_no_controls:
        decision = "BLOCK LAUNCH"
        reason = "Unacceptable risk profile — resolve blockers before deployment"
    elif high_without_residual:
        decision = "PROCEED WITH CONDITIONS"
        reason = "Complete residual risk assessment for HIGH risks before launch"
    else:
        decision = "PROCEED"
        reason = "All risks documented with owners, controls, and acceptable residual levels"

    print(f"\n  ┌{'─' * 50}┐")
    print(f"  │ Decision: {decision:<39}│")
    print(f"  │ {reason[:50]:<50}│")
    print(f"  └{'─' * 50}┘")

    if critical_risks:
        print(f"\n  Blocking risks:")
        for r in critical_risks:
            print(f"    • {r.risk_id}: {r.title}")


def demo_treatment_summary(risks: List[RiskEntry]):
    """Demo 6: Treatment strategy summary."""
    print("\n" + "=" * 60)
    print("Demo 6: Treatment Strategy Summary")
    print("=" * 60)

    # Count by treatment
    treatment_counts = {}
    for risk in risks:
        t = risk.treatment.value
        treatment_counts[t] = treatment_counts.get(t, 0) + 1

    print(f"\n  Treatment distribution across {len(risks)} risks:\n")
    for treatment, count in sorted(treatment_counts.items()):
        bar = "█" * (count * 4)
        print(f"  {treatment:<10} {bar} {count}")

    print(f"\n  Treatment decision guide:")
    print(f"    MITIGATE — Add controls to reduce likelihood or severity")
    print(f"    ACCEPT   — Document low residual risk; sign off explicitly")
    print(f"    TRANSFER — Use contracts, insurance, or vendor SLAs")
    print(f"    AVOID    — Remove or redesign the feature entirely")


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("Risk Scoring MWE — NIST AI RMF Risk Assessment")
    print("=" * 60)

    demo_risk_matrix()

    ml_risks = build_ml_risk_register()
    demo_risk_register(ml_risks, "Traditional ML Risk Register")

    llm_risks = build_llm_risk_register()
    demo_risk_register(llm_risks, "LLM/RAG Additional Risks")

    all_risks = ml_risks + llm_risks
    demo_residual_risk(all_risks)
    demo_deployment_gate(all_risks)
    demo_treatment_summary(all_risks)

    print("\n" + "=" * 60)
    print("MWE Complete — AI risk scoring demonstrated!")
    print("=" * 60)


if __name__ == "__main__":
    main()
