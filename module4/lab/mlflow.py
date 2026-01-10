"""
Lab: Module 4 - MLflow Experiment Tracking and Model Registry
Time: ~30 minutes
Prerequisites: Read the following instructional materials first:
  - mlflow-tracking.md
  - mlflow-model-registry.md
  - experiment-comparison.md
  - artifact-versioning-hashing.md

Learning Objectives:
- LO1: Log parameters, metrics, and artifacts to MLflow tracking
- LO2: Run multiple experiments with varying hyperparameters
- LO3: Compare experiments programmatically and identify best model
- LO4: Register models to MLflow Model Registry with proper versioning
- LO5: Implement artifact hashing for reproducibility verification

Setup Instructions:
    pip install mlflow scikit-learn pandas matplotlib

To run this lab:
    1. Start MLflow tracking server (optional, uses local file store by default):
       mlflow ui --port 5000
    2. Run this script:
       python mlflow.py
    3. View results at http://localhost:5000
"""

import hashlib
import json
import os
import tempfile
from datetime import datetime
from typing import Dict, Any, Tuple

import mlflow
from mlflow import MlflowClient
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# === CONFIGURATION ===
EXPERIMENT_NAME = "module4-lab-hyperparameter-search"
MODEL_NAME = "module4-lab-classifier"
RANDOM_STATE = 42

# === SAMPLE DATA GENERATION (provided) ===
def generate_sample_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Generate synthetic classification data for the lab.
    
    Returns:
        X_train, X_test, y_train, y_test: Train/test split of synthetic data
    """
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        n_classes=2,
        random_state=RANDOM_STATE
    )
    
    # Convert to DataFrame for realistic handling
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=feature_names)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y, test_size=0.2, random_state=RANDOM_STATE
    )
    
    return X_train, X_test, y_train, y_test


# === HELPER FUNCTIONS (provided) ===
def compute_data_hash(df: pd.DataFrame) -> str:
    """Compute SHA-256 hash of DataFrame contents for versioning."""
    content = pd.util.hash_pandas_object(df, index=True).values
    return hashlib.sha256(content.tobytes()).hexdigest()


def compute_file_hash(filepath: str) -> str:
    """Compute SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest()


# ============================================================
# TODO 1: Initialize MLflow Experiment
# ============================================================
# Set up MLflow tracking for this lab. You need to:
# 1. Set the experiment name using mlflow.set_experiment()
# 2. The experiment name should be EXPERIMENT_NAME defined above
#
# HINT: See mlflow-tracking.md section "Basic Tracking Pattern"
# ============================================================
def setup_mlflow_experiment() -> str:
    """Set up MLflow experiment and return experiment ID.
    
    Returns:
        experiment_id: The ID of the created/existing experiment
    """
    # TODO: Set the experiment using mlflow.set_experiment()
    # Your code here (1-2 lines)
    
    pass  # Remove this line after implementing
    
    # Get and return the experiment ID
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    return experiment.experiment_id


# ============================================================
# TODO 2: Log a Training Run with Parameters, Metrics, and Artifacts
# ============================================================
# Implement a function that:
# 1. Starts an MLflow run using mlflow.start_run() as a context manager
# 2. Logs hyperparameters using mlflow.log_params()
# 3. Trains a RandomForestClassifier with the given parameters
# 4. Evaluates and logs metrics using mlflow.log_metrics()
# 5. Logs the data hash as a tag using mlflow.set_tag()
# 6. Logs the trained model using mlflow.sklearn.log_model()
# 7. Returns the run_id and metrics
#
# HINT: See mlflow-tracking.md sections on logging parameters, metrics, artifacts
# ============================================================
def train_and_log_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    params: Dict[str, Any],
    data_hash: str
) -> Dict[str, Any]:
    """Train model and log everything to MLflow.
    
    Args:
        X_train, X_test: Feature DataFrames
        y_train, y_test: Target Series
        params: Hyperparameters dict with keys: n_estimators, max_depth, min_samples_split
        data_hash: Hash of training data for lineage tracking
    
    Returns:
        Dict with run_id, accuracy, f1_score, precision, recall
    """
    # TODO: Start an MLflow run using context manager
    # Uncomment and use the with statement below:
    # with mlflow.start_run():
    #     # Your logging code goes inside this block
    #     pass
    
    # ---- BEGIN: Code that should be INSIDE the with block ----
    # (After uncommenting the with statement above, indent this code)
    
    # TODO: Log all hyperparameters
    # Your code here (1 line using mlflow.log_params)
    # Example: mlflow.log_params(params)
    
    # TODO: Log data lineage information as tags
    # Log: 'data_hash', 'train_samples', 'test_samples'
    # Your code here (3 lines using mlflow.set_tag)
    # Example: mlflow.set_tag('data_hash', data_hash)
    
    # Train model (provided - keep this code)
    model = RandomForestClassifier(
        n_estimators=params['n_estimators'],
        max_depth=params['max_depth'],
        min_samples_split=params['min_samples_split'],
        random_state=RANDOM_STATE
    )
    model.fit(X_train, y_train)
    
    # Evaluate model (provided - keep this code)
    y_pred = model.predict(X_test)
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred)
    }
    
    # TODO: Log all metrics
    # Your code here (1 line using mlflow.log_metrics)
    # Example: mlflow.log_metrics(metrics)
    
    # TODO: Log the sklearn model as an artifact
    # Use artifact_path="model"
    # Your code here (1 line using mlflow.sklearn.log_model)
    # Example: mlflow.sklearn.log_model(model, "model")
    
    # TODO: Get run ID - uncomment when inside with block
    # run_id = mlflow.active_run().info.run_id
    
    # ---- END: Code that should be INSIDE the with block ----
    
    # TODO: Return results - update run_id when implemented
    # Placeholder return - replace 'placeholder' with actual run_id
    return {
        'run_id': 'placeholder',  # Change to: run_id
        **metrics
    }


# ============================================================
# TODO 3: Run Multiple Experiments with Different Hyperparameters
# ============================================================
# Create a list of 5 different hyperparameter configurations and
# run experiments for each one. This is required for Milestone 3.
#
# Each configuration should be a dict with:
#   - n_estimators: int (try values like 50, 100, 150, 200)
#   - max_depth: int or None (try values like 5, 10, 15, 20, None)
#   - min_samples_split: int (try values like 2, 5, 10)
#
# HINT: Vary one parameter at a time to understand its impact
# ============================================================
def get_hyperparameter_configs() -> list:
    """Return list of 5 hyperparameter configurations to try.
    
    Returns:
        List of dicts, each with n_estimators, max_depth, min_samples_split
    """
    # TODO: Define 5 different hyperparameter configurations
    # Your code here - create a list of 5 dicts
    
    configs = [
        # Configuration 1: Baseline
        {'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 2},
        # TODO: Add 4 more configurations below
        # Configuration 2: ...
        # Configuration 3: ...
        # Configuration 4: ...
        # Configuration 5: ...
    ]
    
    return configs


# ============================================================
# TODO 4: Compare Experiments and Find Best Model
# ============================================================
# Query MLflow to find all runs in the experiment and identify
# the best performing model based on accuracy.
#
# You need to:
# 1. Use MlflowClient to search runs
# 2. Filter/sort by accuracy metric
# 3. Return information about the best run
#
# HINT: See experiment-comparison.md section "Programmatic Comparison"
# ============================================================
def find_best_run(experiment_id: str) -> Dict[str, Any]:
    """Find the best performing run in the experiment.
    
    Args:
        experiment_id: MLflow experiment ID
        
    Returns:
        Dict with run_id, accuracy, and parameters of best run
    """
    client = MlflowClient()
    
    # TODO: Search for all runs, ordered by accuracy descending
    # Use client.search_runs() with:
    #   - experiment_ids=[experiment_id]
    #   - order_by=["metrics.accuracy DESC"]
    #   - max_results=10
    # Your code here (1-3 lines)
    
    runs = []  # Replace with your search_runs call
    
    if not runs:
        return {'run_id': None, 'accuracy': 0.0, 'params': {}}
    
    # Get best run (first in sorted results)
    best_run = runs[0]
    
    # TODO: Extract run_id, accuracy, and parameters from best_run
    # best_run.info.run_id gives the run ID
    # best_run.data.metrics gives dict of metrics
    # best_run.data.params gives dict of parameters
    # Your code here (3-5 lines)
    
    return {
        'run_id': None,  # Replace with actual run_id
        'accuracy': 0.0,  # Replace with actual accuracy
        'params': {}  # Replace with actual params
    }


# ============================================================
# TODO 5: Register Model to MLflow Registry
# ============================================================
# Register the best model to the MLflow Model Registry and
# transition it through stages: None -> Staging -> Production
#
# You need to:
# 1. Register model using mlflow.register_model()
# 2. Add a description to the model version
# 3. Transition to Staging stage
# 4. (Bonus) Transition to Production if accuracy > 0.9
#
# HINT: See mlflow-model-registry.md section "Registering a Model"
# ============================================================
def register_best_model(run_id: str, accuracy: float) -> Dict[str, Any]:
    """Register the best model to MLflow Model Registry.
    
    Args:
        run_id: Run ID of the best model
        accuracy: Accuracy of the best model
        
    Returns:
        Dict with model_name, version, and final_stage
    """
    client = MlflowClient()
    
    # TODO: Register the model from the run
    # model_uri should be f"runs:/{run_id}/model"
    # Use mlflow.register_model(model_uri, MODEL_NAME)
    # Your code here (2 lines)
    
    model_uri = f"runs:/{run_id}/model"
    # result = mlflow.register_model(...)  # Uncomment and complete
    
    # TODO: Get the new version number from result.version
    # Your code here (1 line)
    
    new_version = 1  # Replace with result.version
    
    # TODO: Update model version description
    # Use client.update_model_version() with:
    #   - name=MODEL_NAME
    #   - version=new_version  
    #   - description=f"Accuracy: {accuracy:.4f}, Run: {run_id}"
    # Your code here (1-3 lines)
    
    # TODO: Transition model to Staging
    # Use client.transition_model_version_stage() with:
    #   - name=MODEL_NAME
    #   - version=new_version
    #   - stage="Staging"
    # Your code here (1-3 lines)
    
    final_stage = "Staging"
    
    # TODO (BONUS): If accuracy > 0.9, transition to Production
    # Your code here (3-5 lines)
    
    return {
        'model_name': MODEL_NAME,
        'version': new_version,
        'final_stage': final_stage
    }


# ============================================================
# TODO 6: Create Lineage Report
# ============================================================
# Generate a comparison table of all runs for the lineage report.
# This satisfies Milestone 3's requirement for experiment comparison.
#
# Create a pandas DataFrame with columns:
#   - run_id (truncated to 8 chars)
#   - n_estimators, max_depth, min_samples_split
#   - accuracy, f1_score
#   - is_best (boolean)
#
# HINT: See experiment-comparison.md section "Comparison Table"
# ============================================================
def generate_comparison_report(experiment_id: str, best_run_id: str) -> pd.DataFrame:
    """Generate comparison report of all experiment runs.
    
    Args:
        experiment_id: MLflow experiment ID
        best_run_id: Run ID of the best model
        
    Returns:
        DataFrame with run comparisons
    """
    client = MlflowClient()
    
    # Get all runs
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        order_by=["metrics.accuracy DESC"]
    )
    
    # TODO: Build a list of dicts, one per run, with the required fields
    # For each run, extract:
    #   - run_id (use run.info.run_id[:8] for truncated version)
    #   - params: n_estimators, max_depth, min_samples_split
    #   - metrics: accuracy, f1_score
    #   - is_best: True if run.info.run_id == best_run_id
    # Your code here (10-15 lines)
    
    runs_data = []
    for run in runs:
        # TODO: Extract data from each run and append to runs_data
        pass
    
    # Convert to DataFrame
    df = pd.DataFrame(runs_data)
    
    return df


# ============================================================
# SELF-CHECK SECTION (provided)
# ============================================================
def run_self_checks(results: Dict[str, Any]) -> bool:
    """Run self-check assertions to verify lab completion."""
    all_passed = True
    
    print("\n" + "="*60)
    print("SELF-CHECK RESULTS")
    print("="*60)
    
    # Check 1: Experiment was created
    try:
        assert results.get('experiment_id') is not None, \
            "Experiment ID should not be None"
        print("✅ Check 1: MLflow experiment created successfully")
    except AssertionError as e:
        print(f"❌ Check 1 FAILED: {e}")
        all_passed = False
    
    # Check 2: At least 5 runs were created
    try:
        assert results.get('num_runs', 0) >= 5, \
            f"Expected at least 5 runs, got {results.get('num_runs', 0)}"
        print(f"✅ Check 2: Created {results['num_runs']} experiment runs (≥5 required)")
    except AssertionError as e:
        print(f"❌ Check 2 FAILED: {e}")
        all_passed = False
    
    # Check 3: Best model was identified
    try:
        assert results.get('best_run_id') is not None, \
            "Best run ID should not be None"
        assert results.get('best_accuracy', 0) > 0.5, \
            f"Best accuracy should be > 0.5, got {results.get('best_accuracy', 0)}"
        print(f"✅ Check 3: Best model identified (accuracy: {results['best_accuracy']:.4f})")
    except AssertionError as e:
        print(f"❌ Check 3 FAILED: {e}")
        all_passed = False
    
    # Check 4: Model was registered
    try:
        assert results.get('registered_version') is not None, \
            "Model should be registered to registry"
        print(f"✅ Check 4: Model registered as {MODEL_NAME} v{results['registered_version']}")
    except AssertionError as e:
        print(f"❌ Check 4 FAILED: {e}")
        all_passed = False
    
    # Check 5: Comparison report was generated
    try:
        report = results.get('comparison_report')
        assert report is not None and len(report) >= 5, \
            "Comparison report should have at least 5 rows"
        print(f"✅ Check 5: Comparison report generated with {len(report)} runs")
    except AssertionError as e:
        print(f"❌ Check 5 FAILED: {e}")
        all_passed = False
    
    # Check 6: Data hash was logged for lineage
    try:
        assert results.get('data_hash') is not None, \
            "Data hash should be computed for lineage"
        print(f"✅ Check 6: Data hash computed for reproducibility: {results['data_hash'][:16]}...")
    except AssertionError as e:
        print(f"❌ Check 6 FAILED: {e}")
        all_passed = False
    
    print("="*60)
    if all_passed:
        print("🎉 All checks passed! Lab completed successfully.")
        print("\nNext steps for Milestone 3:")
        print("  1. Integrate this tracking into your Airflow DAG")
        print("  2. Add CI/CD validation in GitHub Actions")
        print("  3. Write your lineage report using the comparison table")
    else:
        print("⚠️  Some checks failed. Review the TODOs above.")
    print("="*60)
    
    return all_passed


# ============================================================
# MAIN EXECUTION (provided)
# ============================================================
def main():
    """Main lab execution flow."""
    print("="*60)
    print("Module 4 Lab: MLflow Experiment Tracking")
    print("="*60)
    
    results = {}
    
    # Step 1: Generate sample data
    print("\n📊 Generating sample data...")
    X_train, X_test, y_train, y_test = generate_sample_data()
    data_hash = compute_data_hash(X_train)
    results['data_hash'] = data_hash
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    print(f"   Data hash: {data_hash[:16]}...")
    
    # Step 2: Set up MLflow experiment
    print("\n🔧 Setting up MLflow experiment...")
    experiment_id = setup_mlflow_experiment()
    results['experiment_id'] = experiment_id
    if experiment_id:
        print(f"   Experiment: {EXPERIMENT_NAME}")
        print(f"   Experiment ID: {experiment_id}")
    else:
        print("   ⚠️  Experiment setup incomplete - check TODO 1")
    
    # Step 3: Get hyperparameter configurations
    print("\n📋 Loading hyperparameter configurations...")
    configs = get_hyperparameter_configs()
    print(f"   Configurations to try: {len(configs)}")
    for i, cfg in enumerate(configs, 1):
        print(f"   Config {i}: {cfg}")
    
    # Step 4: Run experiments
    print("\n🚀 Running experiments...")
    run_results = []
    for i, params in enumerate(configs, 1):
        print(f"   Running experiment {i}/{len(configs)}...", end=" ")
        try:
            result = train_and_log_model(
                X_train, X_test, y_train, y_test,
                params, data_hash
            )
            run_results.append(result)
            print(f"accuracy: {result['accuracy']:.4f}")
        except Exception as e:
            print(f"FAILED: {e}")
            run_results.append({'run_id': None, 'accuracy': 0.0})
    
    results['num_runs'] = sum(1 for r in run_results if r.get('run_id') and r['run_id'] != 'placeholder')
    
    # Step 5: Find best model
    print("\n🔍 Finding best model...")
    if experiment_id:
        best = find_best_run(experiment_id)
        results['best_run_id'] = best['run_id']
        results['best_accuracy'] = best['accuracy']
        if best['run_id']:
            print(f"   Best run: {best['run_id'][:8]}...")
            print(f"   Best accuracy: {best['accuracy']:.4f}")
            print(f"   Best params: {best['params']}")
        else:
            print("   ⚠️  No best run found - check TODO 4")
    else:
        print("   ⚠️  Skipped - experiment not set up")
        results['best_run_id'] = None
        results['best_accuracy'] = 0.0
    
    # Step 6: Register model
    print("\n📦 Registering best model...")
    if results.get('best_run_id'):
        try:
            reg_result = register_best_model(
                results['best_run_id'],
                results['best_accuracy']
            )
            results['registered_version'] = reg_result['version']
            print(f"   Model: {reg_result['model_name']}")
            print(f"   Version: {reg_result['version']}")
            print(f"   Stage: {reg_result['final_stage']}")
        except Exception as e:
            print(f"   ⚠️  Registration failed: {e}")
            results['registered_version'] = None
    else:
        print("   ⚠️  Skipped - no best run to register")
        results['registered_version'] = None
    
    # Step 7: Generate comparison report
    print("\n📈 Generating comparison report...")
    if experiment_id and results.get('best_run_id'):
        try:
            report_df = generate_comparison_report(experiment_id, results['best_run_id'])
            results['comparison_report'] = report_df
            if len(report_df) > 0:
                print("\n" + report_df.to_string(index=False))
            else:
                print("   ⚠️  Empty report - check TODO 6")
        except Exception as e:
            print(f"   ⚠️  Report generation failed: {e}")
            results['comparison_report'] = pd.DataFrame()
    else:
        print("   ⚠️  Skipped - missing experiment or runs")
        results['comparison_report'] = pd.DataFrame()
    
    # Step 8: Run self-checks
    run_self_checks(results)
    
    # Expected output guide
    print("\n" + "="*60)
    print("EXPECTED OUTPUT (when all TODOs are complete)")
    print("="*60)
    print("""
📊 Generating sample data...
   Training samples: 800
   Test samples: 200
   Data hash: a1b2c3d4e5f6...

🔧 Setting up MLflow experiment...
   Experiment: module4-lab-hyperparameter-search
   Experiment ID: 1

📋 Loading hyperparameter configurations...
   Configurations to try: 5

🚀 Running experiments...
   Running experiment 1/5... accuracy: 0.8950
   Running experiment 2/5... accuracy: 0.9100
   Running experiment 3/5... accuracy: 0.8850
   Running experiment 4/5... accuracy: 0.9050
   Running experiment 5/5... accuracy: 0.8900

🔍 Finding best model...
   Best run: abc123...
   Best accuracy: 0.9100
   Best params: {'n_estimators': '100', 'max_depth': '15', ...}

📦 Registering best model...
   Model: module4-lab-classifier
   Version: 1
   Stage: Staging

📈 Generating comparison report...
   run_id  n_estimators max_depth min_samples_split accuracy  f1_score is_best
   abc123           100        15                 2   0.9100    0.9050    True
   def456           150        10                 5   0.9050    0.9000   False
   ...

SELF-CHECK RESULTS
✅ Check 1: MLflow experiment created successfully
✅ Check 2: Created 5 experiment runs (≥5 required)
✅ Check 3: Best model identified (accuracy: 0.9100)
✅ Check 4: Model registered as module4-lab-classifier v1
✅ Check 5: Comparison report generated with 5 runs
✅ Check 6: Data hash computed for reproducibility
🎉 All checks passed! Lab completed successfully.
""")


if __name__ == "__main__":
    main()