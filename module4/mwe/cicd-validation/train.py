"""
Simple training script that saves metrics to a file.

This simulates what would happen in a real training pipeline,
where metrics are saved for later validation.
"""

import json
import time
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score


def train_and_save_metrics():
    """Train a model and save metrics for validation."""
    
    print("=" * 50)
    print("Training Model for Validation Demo")
    print("=" * 50)
    
    # Load data
    print("\n📊 Loading Iris dataset...")
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )
    
    # Train model
    print("🏋️ Training RandomForest classifier...")
    start_time = time.time()
    
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    
    # Evaluate
    print("📈 Evaluating model...")
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Save metrics to file (for validation script to read)
    metrics = {
        "accuracy": round(accuracy, 4),
        "f1_score": round(f1, 4),
        "training_time_seconds": round(training_time, 2),
        "n_estimators": 100,
        "max_depth": 5,
    }
    
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    print("\n✅ Training complete!")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   F1 Score: {f1:.4f}")
    print(f"\n💾 Metrics saved to: metrics.json")
    
    return metrics


if __name__ == "__main__":
    train_and_save_metrics()