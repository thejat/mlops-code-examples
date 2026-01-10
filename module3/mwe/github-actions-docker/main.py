#!/usr/bin/env python3
"""
GitHub Actions Docker CI/CD MWE - Demonstrates workflow validation and pipeline simulation.

This script parses GitHub Actions workflow files and simulates the CI/CD pipeline
execution, teaching the core concepts of Docker-based CI/CD automation.
"""

import yaml
from pathlib import Path


def load_workflow(workflow_path: str) -> dict:
    """Load and parse a GitHub Actions workflow YAML file."""
    with open(workflow_path, 'r') as f:
        return yaml.safe_load(f)


def validate_workflow(workflow: dict) -> list[str]:
    """Validate workflow structure and return any issues found."""
    issues = []
    
    # Check required top-level keys
    if 'name' not in workflow:
        issues.append("⚠️  Missing 'name' field - workflow should have a descriptive name")
    if 'on' not in workflow:
        issues.append("❌ Missing 'on' trigger - workflow won't run without triggers")
    if 'jobs' not in workflow:
        issues.append("❌ Missing 'jobs' - workflow has no jobs to execute")
    
    # Validate jobs
    for job_name, job in workflow.get('jobs', {}).items():
        if 'runs-on' not in job:
            issues.append(f"❌ Job '{job_name}' missing 'runs-on' - no runner specified")
        if 'steps' not in job:
            issues.append(f"❌ Job '{job_name}' has no steps to execute")
        
        # Check for checkout step (common mistake)
        steps = job.get('steps', [])
        has_checkout = any('actions/checkout' in str(s.get('uses', '')) for s in steps)
        if not has_checkout and len(steps) > 0:
            issues.append(f"⚠️  Job '{job_name}' may need 'actions/checkout' to access code")
    
    return issues


def analyze_triggers(workflow: dict) -> dict:
    """Analyze workflow triggers and return summary."""
    triggers = workflow.get('on', {})
    analysis = {"types": [], "details": []}
    
    if isinstance(triggers, str):
        triggers = {triggers: None}
    elif isinstance(triggers, list):
        triggers = {t: None for t in triggers}
    
    for trigger, config in triggers.items():
        analysis["types"].append(trigger)
        if trigger == 'push' and config:
            if 'tags' in config:
                analysis["details"].append(f"  - push.tags: {config['tags']} (release builds)")
            if 'branches' in config:
                analysis["details"].append(f"  - push.branches: {config['branches']}")
        elif trigger == 'pull_request':
            analysis["details"].append("  - pull_request: runs on PR events")
    
    return analysis


def simulate_pipeline(workflow: dict) -> None:
    """Simulate workflow execution and print step-by-step output."""
    jobs = workflow.get('jobs', {})
    
    # Build dependency graph
    job_order = []
    remaining = set(jobs.keys())
    
    while remaining:
        for job_name in list(remaining):
            needs = jobs[job_name].get('needs', [])
            if isinstance(needs, str):
                needs = [needs]
            if all(n in job_order for n in needs):
                job_order.append(job_name)
                remaining.remove(job_name)
    
    print("\n🔄 Simulated Pipeline Execution:")
    print("-" * 50)
    
    for job_name in job_order:
        job = jobs[job_name]
        needs = job.get('needs', [])
        runner = job.get('runs-on', 'unknown')
        
        print(f"\n📦 Job: {job_name}")
        print(f"   Runner: {runner}")
        if needs:
            print(f"   Depends on: {needs}")
        
        for i, step in enumerate(job.get('steps', []), 1):
            step_name = step.get('name', step.get('uses', step.get('run', 'unnamed')))
            if len(step_name) > 50:
                step_name = step_name[:47] + "..."
            print(f"   Step {i}: {step_name}")


def main():
    """Main demonstration of GitHub Actions Docker CI/CD concepts."""
    print("=" * 60)
    print("GITHUB ACTIONS DOCKER - CI/CD Pipeline Demonstration")
    print("=" * 60)
    
    workflow_path = Path(__file__).parent / "sample_workflow.yml"
    
    if not workflow_path.exists():
        print(f"❌ Workflow file not found: {workflow_path}")
        return
    
    # Load and validate
    print(f"\n📄 Loading: {workflow_path.name}")
    workflow = load_workflow(workflow_path)
    print(f"   Workflow name: {workflow.get('name', 'unnamed')}")
    
    # Analyze triggers
    print("\n🎯 Trigger Analysis:")
    triggers = analyze_triggers(workflow)
    print(f"   Triggers: {', '.join(triggers['types'])}")
    for detail in triggers["details"]:
        print(detail)
    
    # Validate
    print("\n✅ Validation Results:")
    issues = validate_workflow(workflow)
    if issues:
        for issue in issues:
            print(f"   {issue}")
    else:
        print("   All checks passed!")
    
    # Simulate execution
    simulate_pipeline(workflow)
    
    print("\n" + "=" * 60)
    print("💡 Key Takeaways:")
    print("   1. 'needs' creates job dependencies (test must pass first)")
    print("   2. Secrets are referenced via ${{ secrets.NAME }}")
    print("   3. Tags like 'v*' trigger release builds only")
    print("=" * 60)


if __name__ == "__main__":
    main()