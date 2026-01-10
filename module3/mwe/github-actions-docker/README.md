# GitHub Actions Docker CI/CD MWE

**Pattern:** Automate Docker image builds with GitHub Actions using test-first pipelines and version-tagged releases.

## Prerequisites

- Python 3.9+
- Linux, macOS, or Windows
- No Docker installation required (this MWE simulates the pipeline)

## Quick Start (3 Steps)

```bash
# 1. Clone and navigate
git clone <repo-url>
cd src/module3/learning-activities/mwe/github-actions-docker

# 2. Install dependencies
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Run the demonstration
python main.py
```

## Expected Output

```
============================================================
GITHUB ACTIONS DOCKER - CI/CD Pipeline Demonstration
============================================================

📄 Loading: sample_workflow.yml
   Workflow name: Build and Push Docker Image

 Trigger Analysis:
   Triggers: push, pull_request
  - push.tags: ['v*'] (release builds)
  - pull_request: runs on PR events

 Validation Results:
   All checks passed!

🔄 Simulated Pipeline Execution:
--------------------------------------------------

 Job: test
   Runner: ubuntu-latest
   Step 1: Checkout code
   Step 2: Set up Python
   Step 3: Install dependencies
   Step 4: Run unit tests

 Job: build-and-push
   Runner: ubuntu-latest
   Depends on: ['test']
   Step 1: Checkout code
   Step 2: Authenticate to Google Cloud
   Step 3: Configure Docker for Artifact Registry
   Step 4: Extract version from tag
   Step 5: Build Docker image
   Step 6: Push to Artifact Registry

============================================================
 Key Takeaways:
   1. 'needs' creates job dependencies (test must pass first)
   2. Secrets are referenced via ${{ secrets.NAME }}
   3. Tags like 'v*' trigger release builds only
============================================================
```

## Key Concepts Demonstrated

| Concept | Implementation |
|---------|----------------|
| Job dependencies | `needs: test` ensures tests pass before build |
| Tag-based triggers | `push.tags: ['v*']` for semantic versioning |
| Secret management | `${{ secrets.GCP_SA_KEY }}` for credentials |
| Multi-job workflow | Separate test and build-push jobs |

## Files Included

| File | Purpose |
|------|---------|
| [`main.py`](main.py) | Workflow validator and pipeline simulator |
| [`sample_workflow.yml`](sample_workflow.yml) | Complete CI/CD workflow example |
| [`sample_data/Dockerfile`](sample_data/Dockerfile) | Example Dockerfile for context |
| [`requirements.txt`](requirements.txt) | Single dependency (PyYAML) |

## Extension Challenge

 **Add a validation check for security best practices:**

1. Modify `validate_workflow()` in `main.py` to detect:
   - Hardcoded secrets (strings containing "password", "key", "token")
   - Missing `if:` conditions on push jobs
   - Use of `latest` tags instead of pinned versions

2. Create a `sample_workflow_insecure.yml` with intentional issues

3. Run your validator against both files and compare the output

Example addition to `validate_workflow()`:
```python
# Check for hardcoded secrets
for job_name, job in workflow.get('jobs', {}).items():
    for step in job.get('steps', []):
        run_cmd = step.get('run', '')
        if any(s in run_cmd.lower() for s in ['password', 'api_key', 'token=']):
            issues.append(f" SECURITY: Job '{job_name}' may contain hardcoded secrets")
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'yaml'` | Install PyYAML: `pip install PyYAML` |
| `FileNotFoundError: sample_workflow.yml` | Run from the MWE directory |
| Validation shows unexpected warnings | Review the sample workflow for intentional examples |

## Related Materials

- [GitHub Actions for Docker CI/CD](../../../instructional-materials/github-actions-docker.md)
- [Container Registries & Image Tagging](../../../instructional-materials/container-registries.md)
- [Semantic Versioning for Containers](../../../instructional-materials/semantic-versioning.md)