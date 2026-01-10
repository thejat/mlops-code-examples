"""
Lab: Module 1 - Environment Setup & ML Lifecycle Foundations
Time: ~30 minutes
Prerequisites: Read the following instructional materials first:
  - ml-lifecycle-stages.md
  - python-virtual-environments.md
  - dependency-pinning.md
  - project-organization-mlops.md
  - github-actions-basics.md

Learning Objectives:
- LO1: Identify the six stages of the ML/AI lifecycle and classify code snippets
- LO2: Create properly pinned dependency specifications
- LO3: Understand project organization principles for MLOps
- LO4: Write a basic smoke test for CI validation
"""

# === SETUP (provided) ===
# No external dependencies required - this lab uses only Python standard library
# Run with: python environment-setup.py

from typing import Dict, List, Tuple
import re

# =============================================================================
# PART 1: ML LIFECYCLE STAGES (10 minutes)
# =============================================================================

# The six stages of the ML lifecycle are:
ML_LIFECYCLE_STAGES = [
    "data_collection",
    "data_preprocessing", 
    "model_training",
    "model_evaluation",
    "deployment",
    "monitoring"
]

# Sample code snippets representing different lifecycle stages
CODE_SNIPPETS = {
    "snippet_a": """
    response = requests.get("https://api.example.com/sensor-data")
    raw_data = response.json()
    save_to_database(raw_data)
    """,
    
    "snippet_b": """
    df = df.dropna()
    df['category'] = LabelEncoder().fit_transform(df['category'])
    X_train, X_test = train_test_split(df, test_size=0.2)
    """,
    
    "snippet_c": """
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    joblib.dump(model, 'model.pkl')
    """,
    
    "snippet_d": """
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Model accuracy: {accuracy:.2%}")
    """,
    
    "snippet_e": """
    @app.post("/predict")
    def predict(data: InputSchema):
        model = load_model("model.pkl")
        return {"prediction": model.predict(data.features)}
    """,
    
    "snippet_f": """
    drift_score = calculate_data_drift(production_data, training_data)
    if drift_score > THRESHOLD:
        alert_team("Data drift detected!")
    """
}

# === TODO 1: Classify each code snippet to its lifecycle stage ===
# Map each snippet key to the correct stage from ML_LIFECYCLE_STAGES
# Example: "snippet_a": "data_collection"

def classify_lifecycle_stages() -> Dict[str, str]:
    """
    Classify each code snippet to its corresponding ML lifecycle stage.
    
    Returns:
        Dict mapping snippet keys to lifecycle stage names
    """
    # TODO: Fill in the correct stage for each snippet
    classifications = {
        "snippet_a": "",  # TODO: Which stage collects data from APIs?
        "snippet_b": "",  # TODO: Which stage cleans and transforms data?
        "snippet_c": "",  # TODO: Which stage fits the model?
        "snippet_d": "",  # TODO: Which stage measures performance?
        "snippet_e": "",  # TODO: Which stage serves predictions via API?
        "snippet_f": "",  # TODO: Which stage detects drift in production?
    }
    return classifications


# =============================================================================
# PART 2: DEPENDENCY PINNING (10 minutes)
# =============================================================================

# Sample unpinned dependencies (what NOT to do)
UNPINNED_DEPS = """
pandas
numpy>=1.20
scikit-learn
pytest
requests>=2.0,<3.0
"""

# === TODO 2: Convert unpinned dependencies to properly pinned versions ===
# Each dependency should have an exact version using ==

def create_pinned_requirements() -> str:
    """
    Create a properly pinned requirements.txt content.
    
    Rules:
    - Use exact versions with == (not >= or ranges)
    - Include all dependencies needed for an ML project
    - Use realistic version numbers
    
    Returns:
        String content for requirements.txt
    """
    # TODO: Replace each dependency with a pinned version
    # Use realistic version numbers (e.g., pandas==2.0.3, numpy==1.24.3)
    pinned_requirements = """
# TODO: Pin each dependency with exact versions
# Example format: package==X.Y.Z

pandas
numpy
scikit-learn
pytest
requests
"""
    return pinned_requirements.strip()


def validate_pinned_requirements(requirements: str) -> Tuple[bool, List[str]]:
    """
    Validate that all dependencies are properly pinned.
    
    Returns:
        Tuple of (is_valid, list of error messages)
    """
    errors = []
    lines = [line.strip() for line in requirements.split('\n') 
             if line.strip() and not line.strip().startswith('#')]
    
    # Pattern for properly pinned dependency: package==version
    pinned_pattern = r'^[a-zA-Z][a-zA-Z0-9_-]*==\d+\.\d+(\.\d+)?$'
    
    for line in lines:
        if not re.match(pinned_pattern, line):
            errors.append(f"Not properly pinned: '{line}'")
    
    if len(lines) < 5:
        errors.append(f"Expected at least 5 dependencies, found {len(lines)}")
    
    return len(errors) == 0, errors


# =============================================================================
# PART 3: PROJECT ORGANIZATION (5 minutes)
# =============================================================================

# Files that SHOULD be in .gitignore for an ML project
GITIGNORE_CANDIDATES = [
    "requirements.txt",
    "__pycache__/",
    "venv/",
    "README.md",
    "data/raw/training.csv",
    ".github/workflows/ci.yml",
    "model.pkl",
    "src/train.py",
    ".env",
    "tests/test_model.py",
]

# === TODO 3: Identify which files should be git-ignored ===

def identify_gitignore_files() -> List[str]:
    """
    Identify which files/directories from GITIGNORE_CANDIDATES should be 
    added to .gitignore.
    
    Criteria for git-ignoring:
    - Virtual environments (large, system-specific)
    - Python cache files (generated, not source)
    - Data files (too large for Git)
    - Model artifacts (binary, large)
    - Environment variables (secrets)
    
    Returns:
        List of items that should be in .gitignore
    """
    # TODO: Select the items that should be git-ignored
    # Remove items that should NOT be ignored (like source code, README, etc.)
    should_ignore = [
        # TODO: Add items from GITIGNORE_CANDIDATES that should be ignored
        # Hint: 5 items should be ignored, 5 should NOT be ignored
    ]
    return should_ignore


# =============================================================================
# PART 4: SMOKE TEST WRITING (5 minutes)
# =============================================================================

# === TODO 4: Complete this smoke test function ===

def smoke_test_imports() -> bool:
    """
    A smoke test that verifies core standard library imports work.
    This simulates what you'd do for ML packages in a real project.
    
    Returns:
        True if all imports succeed, False otherwise
    """
    try:
        # TODO: Add import statements for these standard library modules:
        # - json (for data serialization)
        # - pathlib (for file path handling)
        # - typing (for type hints)
        # - os (for environment variables)
        
        # Uncomment and complete the imports below:
        # import json
        # import pathlib
        # import typing
        # import os
        
        # TODO: Set this to True after adding all imports
        imports_completed = False
        
        if not imports_completed:
            return False
        
        return True
    except ImportError as e:
        print(f"Import failed: {e}")
        return False


def smoke_test_basic_operations() -> bool:
    """
    A smoke test that verifies basic operations work.
    
    Returns:
        True if all operations succeed, False otherwise
    """
    try:
        # TODO: Implement these basic checks:
        # 1. Create a sample config dictionary with keys: "model_type", "n_estimators"
        # 2. Verify "model_type" key exists in the config
        # 3. Create a Path object using pathlib.Path(__file__)
        
        # Uncomment and complete:
        # import json
        # from pathlib import Path
        #
        # config = {"model_type": "random_forest", "n_estimators": 100}
        # assert "model_type" in config, "Config missing model_type"
        #
        # current_file = Path(__file__)
        # assert current_file.exists(), "Path check failed"
        
        # TODO: Set this to True after implementing the checks
        operations_completed = False
        
        if not operations_completed:
            return False
        
        return True
    except Exception as e:
        print(f"Operation failed: {e}")
        return False


# =============================================================================
# EXPECTED OUTPUT
# =============================================================================
"""
When completed correctly, running this script should output:

=== Part 1: ML Lifecycle Classification ===
✅ snippet_a -> data_collection
✅ snippet_b -> data_preprocessing
✅ snippet_c -> model_training
✅ snippet_d -> model_evaluation
✅ snippet_e -> deployment
✅ snippet_f -> monitoring
All 6 snippets classified correctly!

=== Part 2: Dependency Pinning ===
✅ All dependencies properly pinned!
Sample of your pinned requirements:
  - pandas==2.0.3
  - numpy==1.24.3
  ...

=== Part 3: Project Organization ===
✅ Correctly identified 5 items to git-ignore:
  - __pycache__/
  - venv/
  - data/raw/training.csv
  - model.pkl
  - .env

=== Part 4: Smoke Tests ===
✅ Import smoke test passed!
✅ Basic operations smoke test passed!

=== FINAL RESULT ===
✅ All checks passed! You're ready for Milestone 0!
"""


# =============================================================================
# SELF-CHECK (Do not modify below this line)
# =============================================================================

def run_all_checks():
    """Run all validation checks and report results."""
    all_passed = True
    
    print("\n" + "="*60)
    print("=== Part 1: ML Lifecycle Classification ===")
    print("="*60)
    
    # Check Part 1
    expected_classifications = {
        "snippet_a": "data_collection",
        "snippet_b": "data_preprocessing",
        "snippet_c": "model_training",
        "snippet_d": "model_evaluation",
        "snippet_e": "deployment",
        "snippet_f": "monitoring"
    }
    
    student_classifications = classify_lifecycle_stages()
    part1_correct = 0
    
    for snippet, expected in expected_classifications.items():
        actual = student_classifications.get(snippet, "")
        if actual == expected:
            print(f"✅ {snippet} -> {expected}")
            part1_correct += 1
        else:
            print(f"❌ {snippet}: expected '{expected}', got '{actual}'")
            all_passed = False
    
    if part1_correct == 6:
        print("All 6 snippets classified correctly!")
    else:
        print(f"Classified {part1_correct}/6 correctly. Review the instructional materials.")
    
    print("\n" + "="*60)
    print("=== Part 2: Dependency Pinning ===")
    print("="*60)
    
    # Check Part 2
    pinned = create_pinned_requirements()
    is_valid, errors = validate_pinned_requirements(pinned)
    
    if is_valid:
        print("✅ All dependencies properly pinned!")
        print("Sample of your pinned requirements:")
        for line in pinned.split('\n')[:5]:
            if line.strip() and not line.startswith('#'):
                print(f"  - {line.strip()}")
    else:
        print("❌ Dependency pinning issues found:")
        for error in errors:
            print(f"  - {error}")
        all_passed = False
    
    print("\n" + "="*60)
    print("=== Part 3: Project Organization ===")
    print("="*60)
    
    # Check Part 3
    expected_ignore = {"__pycache__/", "venv/", "data/raw/training.csv", "model.pkl", ".env"}
    student_ignore = set(identify_gitignore_files())
    
    if student_ignore == expected_ignore:
        print(f"✅ Correctly identified {len(expected_ignore)} items to git-ignore:")
        for item in sorted(expected_ignore):
            print(f"  - {item}")
    else:
        missing = expected_ignore - student_ignore
        extra = student_ignore - expected_ignore
        if missing:
            print(f"❌ Missing from your list: {missing}")
        if extra:
            print(f"❌ Should NOT be ignored: {extra}")
        all_passed = False
    
    print("\n" + "="*60)
    print("=== Part 4: Smoke Tests ===")
    print("="*60)
    
    # Check Part 4
    if smoke_test_imports():
        print("✅ Import smoke test passed!")
    else:
        print("❌ Import smoke test failed!")
        all_passed = False
    
    if smoke_test_basic_operations():
        print("✅ Basic operations smoke test passed!")
    else:
        print("❌ Basic operations smoke test failed!")
        all_passed = False
    
    print("\n" + "="*60)
    print("=== FINAL RESULT ===")
    print("="*60)
    
    if all_passed:
        print("✅ All checks passed! You're ready for Milestone 0!")
    else:
        print("❌ Some checks failed. Review the sections above and try again.")
    
    return all_passed


if __name__ == "__main__":
    success = run_all_checks()
    # Exit with appropriate code for CI integration
    exit(0 if success else 1)
