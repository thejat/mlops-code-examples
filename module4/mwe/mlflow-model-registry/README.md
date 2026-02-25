# MLflow Model Registry - Minimal Working Example

This example demonstrates MLflow's Model Registry: registering models, managing versions, transitioning stages, and loading models for inference.

## What You'll Learn

- How to register a trained model in the Model Registry
- How model versioning works automatically
- How to transition models through stages (None → Staging → Production)
- How to load models by name, version, or stage
- How to manage model metadata and descriptions

## Files

| File | Purpose |
|------|---------|
| `train_and_register.py` | Train a model and register it in Model Registry |
| `manage_model.py` | Manage versions, stages, and metadata |
| `load_model.py` | Load registered models for inference |
| `requirements.txt` | Python dependencies |
| `expected_output/sample_output.txt` | Expected console output |

## Prerequisites

- Python 3.9+
- pip

## Quick Start

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# OR: venv\Scripts\activate  # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train and register a model (creates mlflow.db and version 1)
python train_and_register.py

# 4. Run again to create version 2
python train_and_register.py

# 5. Manage model versions and stages
python manage_model.py

# 6. Load and use registered models
python load_model.py

# 7. View in MLflow UI
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
# Open http://localhost:5000 and click "Models" tab
```

## Expected Behavior

1. `train_and_register.py` trains a RandomForest model and registers it
2. Running it twice creates version 1 and version 2
3. `manage_model.py` transitions the latest version to Production
4. `load_model.py` loads models by version number, stage, and latest
5. The MLflow UI shows your model in the "Models" tab

## Code Walkthrough

### 1. Setting Up the Tracking URI (Required!)

```python
import mlflow

# CRITICAL: Model Registry requires a database backend
# File-based mlruns/ does NOT support Model Registry
DB_URI = "sqlite:///mlflow.db"
mlflow.set_tracking_uri(DB_URI)
```

### 2. Registering a Model

```python
# Register during logging (recommended)
mlflow.sklearn.log_model(
    model,
    artifact_path="model",
    registered_model_name="iris-classifier"  # This triggers registration
)

# OR register an existing run's model
mlflow.register_model(
    model_uri=f"runs:/{run_id}/model",
    name="iris-classifier"
)
```

### 3. Managing Stages

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Transition to Staging
client.transition_model_version_stage(
    name="iris-classifier",
    version="1",
    stage="Staging"
)

# Promote to Production
client.transition_model_version_stage(
    name="iris-classifier",
    version="1",
    stage="Production",
    archive_existing_versions=True  # Archive old Production models
)
```

### 4. Loading Registered Models

```python
# Load by version number
model = mlflow.sklearn.load_model("models:/iris-classifier/1")

# Load by stage
model = mlflow.sklearn.load_model("models:/iris-classifier/Production")

# Load latest version (must query first - no "latest" URI!)
client = MlflowClient()
versions = client.search_model_versions("name='iris-classifier'")
latest = max(int(v.version) for v in versions)
model = mlflow.sklearn.load_model(f"models:/iris-classifier/{latest}")
```

## Key Concepts

### Model Registry vs Tracking

| Feature | MLflow Tracking | Model Registry |
|---------|-----------------|----------------|
| Purpose | Log experiments | Manage model lifecycle |
| Storage | Runs, params, metrics | Named models, versions |
| URI | `runs:/{run_id}/model` | `models:/{name}/{version}` |
| Stages | N/A | None, Staging, Production, Archived |

### Stage Workflow

```
None → Staging → Production → Archived
         ↓
    (validation)
```

### Valid Model URIs

| URI Format | Example | Description |
|------------|---------|-------------|
| `models:/{name}/{version}` | `models:/iris-classifier/1` | Specific version |
| `models:/{name}/{stage}` | `models:/iris-classifier/Production` | Latest in stage |
| ~~`models:/{name}/latest`~~ | ❌ NOT VALID | Use MlflowClient instead |

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "Model registry functionality is not supported" | Use SQLite backend: `mlflow.set_tracking_uri("sqlite:///mlflow.db")` |
| "No module named mlflow" | Run `pip install -r requirements.txt` |
| Model not found in registry | Run `train_and_register.py` first |
| Multiple mlflow.db files | Run all scripts from the same directory |
| UI doesn't show Models tab | Use `--backend-store-uri sqlite:///mlflow.db` when starting UI |

## Related Materials

- [mlflow-tracking MWE](../mlflow-tracking/) - MLflow Tracking basics
- [MLflow Model Registry Docs](https://mlflow.org/docs/latest/model-registry.html)

## Next Steps

- Try different model architectures and compare versions
- Set up a remote tracking server for team collaboration
- Integrate model registry with CI/CD pipelines