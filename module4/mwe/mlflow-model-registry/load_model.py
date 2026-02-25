"""
MLflow Model Registry - Load Registered Models

This script demonstrates:
- Loading a model by version number (models:/{name}/{version})
- Loading a model by stage (models:/{name}/Production)
- Finding and loading the latest version (via MlflowClient)
- Making predictions with loaded models

Run after: python manage_model.py
Run with: python load_model.py

NOTE: There is NO "models:/{name}/latest" URI - you must query for the
latest version using MlflowClient, then load by version number.
"""

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.datasets import load_iris
import numpy as np
import sys

# Configuration - must match other scripts
MODEL_NAME = "iris-classifier"
DB_URI = "sqlite:///mlflow.db"


def load_and_predict():
    """Demonstrate loading models from the Model Registry."""
    
    # 1. Configure MLflow - must use same URI as other scripts
    mlflow.set_tracking_uri(DB_URI)
    client = MlflowClient()
    
    print("=" * 60)
    print("MLflow Model Registry - Load Models")
    print("=" * 60)
    print(f"\n🔧 Using tracking URI: {DB_URI}")
    
    # 2. Check if model exists
    try:
        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        if not versions:
            raise Exception("No versions found")
    except Exception:
        print(f"\n❌ Error: Model '{MODEL_NAME}' not found in registry.")
        print("   Run 'python train_and_register.py' first!")
        sys.exit(1)
    
    # 3. Prepare sample data for predictions
    print("\n📊 Loading sample data...")
    iris = load_iris()
    sample = iris.data[:5]  # Use first 5 samples
    actual_labels = iris.target[:5]
    print(f"   Sample shape: {sample.shape}")
    print(f"   Actual labels: {list(actual_labels)}")
    
    print("\n🔍 Loading models from registry...")
    
    # 4. Method 1: Load by version number
    print(f"\n   Method 1: Load by version")
    print(f"   Loading models:/{MODEL_NAME}/1...")
    try:
        model_v1 = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/1")
        predictions_v1 = model_v1.predict(sample)
        print(f"   ✓ Loaded version 1")
    except Exception as e:
        print(f"   ✗ Could not load version 1: {e}")
        predictions_v1 = None
    
    # 5. Method 2: Load by stage (Production)
    print(f"\n   Method 2: Load by stage")
    print(f"   Loading models:/{MODEL_NAME}/Production...")
    try:
        model_prod = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/Production")
        predictions_prod = model_prod.predict(sample)
        print(f"   ✓ Loaded Production model")
    except Exception as e:
        print(f"   ✗ Could not load Production model: {e}")
        print("   (Run manage_model.py to promote a model to Production)")
        predictions_prod = None
    
    # 6. Method 3: Load latest version (must query first)
    # NOTE: "models:/{name}/latest" does NOT work!
    print(f"\n   Method 3: Load latest version")
    print(f"   Querying latest version via MlflowClient...")
    latest_version = max(int(v.version) for v in versions)
    print(f"   Latest version is: {latest_version}")
    print(f"   Loading models:/{MODEL_NAME}/{latest_version}...")
    model_latest = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{latest_version}")
    predictions_latest = model_latest.predict(sample)
    print(f"   ✓ Loaded version {latest_version}")
    
    # 7. Display predictions
    print("\n" + "=" * 60)
    print("🎯 Predictions (5 samples):")
    print("=" * 60)
    print(f"\n   Actual labels:  {list(actual_labels)}")
    if predictions_v1 is not None:
        print(f"   Version 1:      {list(predictions_v1)}")
    if predictions_prod is not None:
        print(f"   Production:     {list(predictions_prod)}")
    print(f"   Latest (v{latest_version}):   {list(predictions_latest)}")
    
    # 8. Map predictions to class names
    print("\n📋 Class names:")
    for i, name in enumerate(iris.target_names):
        print(f"   {i}: {name}")
    
    print("\n" + "=" * 60)
    print("✅ All models loaded and predictions complete!")
    print("=" * 60)
    
    return {
        "model_name": MODEL_NAME,
        "latest_version": latest_version,
        "predictions": list(predictions_latest),
    }


if __name__ == "__main__":
    result = load_and_predict()