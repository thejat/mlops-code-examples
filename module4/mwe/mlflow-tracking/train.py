"""
MLflow Tracking - Minimal Working Example

This script demonstrates basic MLflow tracking:
- Logging parameters (hyperparameters)
- Logging metrics (model performance)
- Logging artifacts (trained model)

Run with: python train.py
View results: mlflow ui --port 5000
"""

import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import time


def train_and_track():
    """Train a model with full MLflow tracking."""
    
    # 1. Set up the experiment
    # This creates a new experiment or selects an existing one
    mlflow.set_experiment("iris-classification")
    
    print("=" * 50)
    print("MLflow Tracking - Minimal Working Example")
    print("=" * 50)
    
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
    
    # 4. Start an MLflow run
    # Everything inside this block is logged to MLflow
    with mlflow.start_run():
        
        # Get the run ID for reference
        run_id = mlflow.active_run().info.run_id
        print(f"\n🔵 Started MLflow run: {run_id[:8]}...")
        
        # 5. Log parameters (hyperparameters)
        print("\n📝 Logging parameters...")
        mlflow.log_params(params)
        for key, value in params.items():
            print(f"   {key}: {value}")
        
        # Also log data-related parameters
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))
        mlflow.log_param("n_features", iris.data.shape[1])
        
        # 6. Train the model
        print("\n🏋️ Training RandomForest classifier...")
        start_time = time.time()
        
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        
        training_time = time.time() - start_time
        print(f"   Training completed in {training_time:.2f} seconds")
        
        # 7. Evaluate the model
        print("\n📈 Evaluating model...")
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   F1 Score: {f1:.4f}")
        
        # 8. Log metrics
        print("\n📊 Logging metrics...")
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("training_time_seconds", training_time)
        
        # 9. Log the model as an artifact
        print("\n💾 Logging model artifact...")
        mlflow.sklearn.log_model(model, "model")
        print("   Model saved to: model/")
        
        # 10. Add tags for organization
        mlflow.set_tag("model_type", "RandomForest")
        mlflow.set_tag("dataset", "iris")
        mlflow.set_tag("status", "completed")
        
        print("\n" + "=" * 50)
        print("✅ Training complete! Run logged to MLflow.")
        print("=" * 50)
        print(f"\n🔗 Run ID: {run_id}")
        print("\n👀 To view results, run:")
        print("   mlflow ui --port 5000")
        print("   Then open http://localhost:5000")
        
        return {
            "run_id": run_id,
            "accuracy": accuracy,
            "f1_score": f1,
        }


if __name__ == "__main__":
    result = train_and_track()