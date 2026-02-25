"""
MLflow Model Registry - Train and Register

This script demonstrates:
- Training a model with MLflow tracking
- Registering the model in MLflow Model Registry
- Automatic model versioning

Run with: python train_and_register.py
Run twice to create multiple versions.

IMPORTANT: Model Registry requires a database backend (SQLite or PostgreSQL).
File-based mlruns/ does NOT support Model Registry.
"""

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import time
import os

# Configuration
MODEL_NAME = "iris-classifier"
# Use SQLite backend - Model Registry requires a database, not file store
# Note: This is a relative path. Run all scripts from the same directory.
DB_URI = "sqlite:///mlflow.db"


def train_and_register():
    """Train a model and register it in the Model Registry."""
    
    # 1. Configure MLflow with database backend
    # CRITICAL: Model Registry only works with database-backed tracking stores
    mlflow.set_tracking_uri(DB_URI)
    mlflow.set_experiment("iris-model-registry")
    
    print("=" * 60)
    print("MLflow Model Registry - Train and Register")
    print("=" * 60)
    print(f"\n🔧 Using tracking URI: {DB_URI}")
    
    # 2. Load and split data
    print("\n📊 Loading Iris dataset...")
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    
    # 3. Define hyperparameters
    params = {
        "n_estimators": 100,
        "max_depth": 5,
        "random_state": 42,
    }
    
    # 4. Start MLflow run
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"\n🔵 Started MLflow run: {run_id[:8]}...")
        
        # 5. Log parameters
        print("\n📝 Logging parameters...")
        mlflow.log_params(params)
        for key, value in params.items():
            print(f"   {key}: {value}")
        
        # 6. Train model
        print("\n🏋️ Training RandomForest classifier...")
        start_time = time.time()
        
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        
        training_time = time.time() - start_time
        print(f"   Training completed in {training_time:.2f} seconds")
        
        # 7. Evaluate
        print("\n📈 Evaluating model...")
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   F1 Score: {f1:.4f}")
        
        # 8. Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("training_time_seconds", training_time)
        
        # 9. Log and REGISTER the model in one step
        # The registered_model_name parameter registers the model in Model Registry
        print("\n📝 Registering model in Model Registry...")
        model_info = mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name=MODEL_NAME  # This triggers registration
        )
        
        # 10. Get version information
        client = MlflowClient()
        
        # Find the version that was just created
        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        latest_version = max(int(v.version) for v in versions)
        
        print(f"\n✅ Model registered successfully!")
        print(f"   Model name: {MODEL_NAME}")
        print(f"   Model version: {latest_version}")
        print(f"   Run ID: {run_id}")
        print(f"   Model URI: models:/{MODEL_NAME}/{latest_version}")
        
        # 11. Add tags
        mlflow.set_tag("model_type", "RandomForest")
        mlflow.set_tag("dataset", "iris")
        mlflow.set_tag("registered", "true")
        
        print("\n" + "=" * 60)
        print("✅ Training and registration complete!")
        print("=" * 60)
        print("\n👀 To view in MLflow UI, run:")
        print(f"   mlflow ui --backend-store-uri {DB_URI} --port 5000")
        print("   Then open http://localhost:5000")
        print("\n💡 Run this script again to create version 2!")
        
        return {
            "run_id": run_id,
            "model_name": MODEL_NAME,
            "model_version": latest_version,
            "accuracy": accuracy,
        }


if __name__ == "__main__":
    result = train_and_register()