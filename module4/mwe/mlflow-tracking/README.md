# MLflow Tracking - Minimal Working Example

This example demonstrates basic MLflow tracking: logging parameters, metrics, and artifacts during a model training run.

## What You'll Learn

- How to set up MLflow tracking
- How to log parameters (hyperparameters, configuration)
- How to log metrics (accuracy, loss, training time)
- How to log artifacts (model files, plots)
- How to view results in the MLflow UI

## Files

| File | Purpose |
|------|---------|
| `train.py` | Main training script with MLflow tracking |
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

# 3. Run the training script
python train.py

# 4. View results in MLflow UI
mlflow ui --port 5000
# Open http://localhost:5000 in your browser
```

## Expected Behavior

1. Script trains a simple RandomForest classifier on the Iris dataset
2. MLflow logs parameters, metrics, and the trained model
3. A `mlruns/` directory is created with experiment data
4. The MLflow UI shows your experiment with all logged data

## Code Walkthrough

### 1. Setting Up the Experiment

```python
import mlflow

mlflow.set_experiment("iris-classification")  # Create/select experiment
```

### 2. Starting a Run and Logging Parameters

```python
with mlflow.start_run():
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 5)
```

### 3. Logging Metrics

```python
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("training_time_seconds", elapsed_time)
```

### 4. Logging the Model

```python
    mlflow.sklearn.log_model(model, "model")
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "No module named mlflow" | Run `pip install -r requirements.txt` |
| MLflow UI won't start | Check if port 5000 is in use: `lsof -i :5000` |
| No experiments visible | Make sure you ran `train.py` first |

## Next Steps

- Try changing hyperparameters and running multiple times
- Compare runs in the MLflow UI
- Explore the `mlflow-model-registry/` MWE for model registration

## Related Materials

- MLflow Tracking Fundamentals
- Experiment Comparison