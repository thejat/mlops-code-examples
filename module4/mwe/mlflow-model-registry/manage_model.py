"""
MLflow Model Registry - Manage Model Versions and Stages

This script demonstrates:
- Listing registered model versions
- Updating model descriptions
- Transitioning models through stages (None → Staging → Production)
- Archiving old versions

Run after: python train_and_register.py (at least twice for multiple versions)
Run with: python manage_model.py
"""

import mlflow
from mlflow.tracking import MlflowClient
import sys

# Configuration - must match train_and_register.py
MODEL_NAME = "iris-classifier"
DB_URI = "sqlite:///mlflow.db"


def manage_model():
    """Demonstrate model version management and stage transitions."""
    
    # 1. Configure MLflow - must use same URI as train_and_register.py
    mlflow.set_tracking_uri(DB_URI)
    client = MlflowClient()
    
    print("=" * 60)
    print("MLflow Model Registry - Stage Management")
    print("=" * 60)
    print(f"\n🔧 Using tracking URI: {DB_URI}")
    
    # 2. Check if model exists
    try:
        model = client.get_registered_model(MODEL_NAME)
    except mlflow.exceptions.MlflowException:
        print(f"\n❌ Error: Model '{MODEL_NAME}' not found in registry.")
        print("   Run 'python train_and_register.py' first!")
        sys.exit(1)
    
    # 3. Display model info
    print(f"\n📋 Registered Model: {model.name}")
    print(f"   Description: {model.description or '(none)'}")
    print(f"   Creation time: {model.creation_timestamp}")
    
    # 4. List all versions
    print("\n📦 Model Versions:")
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    
    if not versions:
        print("   No versions found!")
        return
    
    for v in sorted(versions, key=lambda x: int(x.version)):
        print(f"   Version {v.version}: stage={v.current_stage}, run_id={v.run_id[:8]}...")
    
    # 5. Update model description
    print("\n📝 Updating model description...")
    client.update_registered_model(
        name=MODEL_NAME,
        description="Iris flower classifier using RandomForest. Trained on the classic Iris dataset."
    )
    print("   ✓ Description updated")
    
    # 6. Get the latest version
    latest_version = max(int(v.version) for v in versions)
    print(f"\n🔄 Transitioning version {latest_version}...")
    
    # 7. Transition to Staging
    print(f"   None → Staging...", end=" ")
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=str(latest_version),
        stage="Staging",
        archive_existing_versions=False
    )
    print("✓")
    
    # 8. Transition to Production
    print(f"   Staging → Production...", end=" ")
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=str(latest_version),
        stage="Production",
        archive_existing_versions=True  # Archive any existing Production versions
    )
    print("✓")
    
    # 9. Add version description
    client.update_model_version(
        name=MODEL_NAME,
        version=str(latest_version),
        description=f"Production model - accuracy validated"
    )
    
    # 10. Display final state
    print(f"\n✅ Version {latest_version} is now in Production!")
    
    print("\n📦 Final Model Versions:")
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    for v in sorted(versions, key=lambda x: int(x.version)):
        print(f"   Version {v.version}: stage={v.current_stage}")
    
    print("\n" + "=" * 60)
    print("✅ Model management complete!")
    print("=" * 60)
    print("\n👀 View in MLflow UI:")
    print(f"   mlflow ui --backend-store-uri {DB_URI} --port 5000")
    
    return {
        "model_name": MODEL_NAME,
        "production_version": latest_version,
    }


if __name__ == "__main__":
    result = manage_model()