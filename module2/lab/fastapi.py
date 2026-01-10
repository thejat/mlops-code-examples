"""
Lab: Module 2 - FastAPI Inference Service
Time: ~30 minutes
Prerequisites: Read the following instructional materials first:
  - rest-api-design.md
  - fastapi-fundamentals.md
  - pydantic-schemas.md
  - model-artifact-loading.md
  - ml-serving-lifecycle.md

Learning Objectives:
- LO1: Define Pydantic request/response schemas for ML inference
- LO2: Implement a FastAPI prediction endpoint with proper validation
- LO3: Apply correct model loading patterns (eager vs. per-request)
- LO4: Understand HTTP status codes and error handling in ML APIs
"""

# === SETUP (provided) ===
# To run this lab, install minimal dependencies:
#   pip install fastapi pydantic
#
# This lab uses a simulated model to avoid sklearn dependency.
# The concepts apply directly to real ML models.
#
# Run validation with: python fastapi.py
# After completing TODOs, test your API with: uvicorn fastapi:app --reload

from typing import Dict, List, Optional, Tuple
import json

# =============================================================================
# PART 1: PYDANTIC SCHEMAS (10 minutes)
# =============================================================================

# We'll use Pydantic's BaseModel for request/response validation.
# In a real FastAPI app, this provides automatic validation and docs.

try:
    from pydantic import BaseModel, Field, field_validator
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    # Mock BaseModel for validation purposes when pydantic not installed
    class BaseModel:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
        def model_dump(self):
            return self.__dict__
    def Field(*args, **kwargs):
        return None
    def field_validator(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

# Sample feature data for iris classification
SAMPLE_FEATURES = {
    "setosa": {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2},
    "versicolor": {"sepal_length": 6.0, "sepal_width": 2.7, "petal_length": 4.5, "petal_width": 1.5},
    "virginica": {"sepal_length": 6.9, "sepal_width": 3.1, "petal_length": 5.4, "petal_width": 2.1},
}


# === TODO 1: Complete the PredictionRequest schema ===
# Define a Pydantic model for incoming prediction requests.
# 
# Requirements:
# - All four iris features: sepal_length, sepal_width, petal_length, petal_width
# - All fields should be float type
# - All fields should be required (no defaults)
# - Add Field() with gt=0 constraint (values must be greater than 0)
# - Add description for at least one field

class PredictionRequest(BaseModel):
    """Input schema for iris prediction endpoint."""
    
    # TODO: Define sepal_length as a required float field > 0
    # Example: field_name: float = Field(..., gt=0, description="...")
    sepal_length: float = None  # TODO: Replace None with proper Field()
    
    # TODO: Define sepal_width as a required float field > 0
    sepal_width: float = None  # TODO: Replace None with proper Field()
    
    # TODO: Define petal_length as a required float field > 0
    petal_length: float = None  # TODO: Replace None with proper Field()
    
    # TODO: Define petal_width as a required float field > 0
    petal_width: float = None  # TODO: Replace None with proper Field()


# === TODO 2: Complete the PredictionResponse schema ===
# Define a Pydantic model for outgoing prediction responses.
#
# Requirements:
# - prediction: str (the predicted class name)
# - confidence: float between 0.0 and 1.0
# - model_version: str (e.g., "1.0.0")
# - Optional: request_id with default None

class PredictionResponse(BaseModel):
    """Output schema for iris prediction endpoint."""
    
    # TODO: Define prediction as a required string field
    prediction: str = None  # TODO: Replace with proper definition
    
    # TODO: Define confidence as a float between 0 and 1
    # Hint: Use Field(..., ge=0.0, le=1.0)
    confidence: float = None  # TODO: Replace with proper Field()
    
    # TODO: Define model_version as a required string
    model_version: str = None  # TODO: Replace with proper definition
    
    # TODO: Define request_id as an optional string with default None
    request_id: Optional[str] = None


# =============================================================================
# PART 2: MODEL LOADING PATTERNS (5 minutes)
# =============================================================================

# This simulates a trained ML model. In production, you'd use joblib.load()
class SimulatedIrisModel:
    """Simulates a trained scikit-learn classifier for testing purposes."""
    
    def __init__(self):
        self.version = "1.0.0"
        self._is_loaded = True
        # Simulate model loading time
        print("Model initialized (simulated)")
    
    def predict(self, features: List[List[float]]) -> List[str]:
        """Simulate prediction based on petal length."""
        predictions = []
        for f in features:
            petal_length = f[2]  # Third feature
            if petal_length < 2.5:
                predictions.append("setosa")
            elif petal_length < 4.9:
                predictions.append("versicolor")
            else:
                predictions.append("virginica")
        return predictions
    
    def predict_proba(self, features: List[List[float]]) -> List[List[float]]:
        """Simulate probability scores."""
        # Returns a list of [setosa_prob, versicolor_prob, virginica_prob]
        probas = []
        for f in features:
            petal_length = f[2]
            if petal_length < 2.5:
                probas.append([0.95, 0.03, 0.02])
            elif petal_length < 4.9:
                probas.append([0.05, 0.85, 0.10])
            else:
                probas.append([0.02, 0.08, 0.90])
        return probas


# === TODO 3: Implement eager model loading ===
# The model should be loaded ONCE at module level, not inside functions.
#
# Why? Loading inside a function means re-loading on every request.
# This adds 100ms-2s latency per request for real models.

# BAD PATTERN (don't do this):
# def get_model():
#     return SimulatedIrisModel()  # Loads on EVERY call!

# GOOD PATTERN:
# TODO: Create a module-level variable called 'model' that holds a SimulatedIrisModel instance
# This ensures the model loads once when the module imports

model = None  # TODO: Replace None with SimulatedIrisModel()


# =============================================================================
# PART 3: PREDICTION ENDPOINT LOGIC (10 minutes)
# =============================================================================

# === TODO 4: Implement the prediction function ===
# This simulates what would be inside your FastAPI @app.post("/predict") handler.

def make_prediction(request_data: Dict[str, float]) -> Dict[str, any]:
    """
    Process a prediction request and return the response.
    
    This function simulates the core logic of a FastAPI /predict endpoint.
    In a real FastAPI app, this would be decorated with @app.post("/predict").
    
    Args:
        request_data: Dictionary with sepal_length, sepal_width, petal_length, petal_width
    
    Returns:
        Dictionary matching PredictionResponse schema
    
    Raises:
        ValueError: If model is not loaded or input is invalid
    """
    # TODO Step 1: Check if model is loaded, raise ValueError if not
    # Hint: Check if 'model' is None
    if model is None:
        pass  # TODO: Raise ValueError with message "Model not loaded"
    
    # TODO Step 2: Extract features from request_data into a list
    # Format: [[sepal_length, sepal_width, petal_length, petal_width]]
    # Hint: features = [[request_data["sepal_length"], ...]]
    features = None  # TODO: Extract features as nested list
    
    # TODO Step 3: Get prediction from model
    # Hint: prediction = model.predict(features)[0]
    prediction = None  # TODO: Call model.predict()
    
    # TODO Step 4: Get confidence from model.predict_proba()
    # Hint: Get max probability from predict_proba result
    confidence = None  # TODO: Get max probability
    
    # TODO Step 5: Return response dictionary matching PredictionResponse schema
    # Include: prediction, confidence, model_version
    response = {
        # TODO: Fill in the response fields
    }
    
    return response


# =============================================================================
# PART 4: HTTP STATUS CODES & ERROR HANDLING (5 minutes)
# =============================================================================

# HTTP status codes are crucial for API consumers to understand what happened.
# Match each scenario to its correct status code.

HTTP_STATUS_SCENARIOS = {
    "successful_prediction": {
        "description": "Model returns a valid prediction",
        "correct_status": None,  # TODO: What status code?
    },
    "missing_required_field": {
        "description": "Client sends JSON without 'sepal_length'",
        "correct_status": None,  # TODO: What status code?
    },
    "invalid_field_type": {
        "description": "Client sends sepal_length: 'five' instead of 5.0",
        "correct_status": None,  # TODO: What status code?
    },
    "model_not_loaded": {
        "description": "Model artifact failed to load at startup",
        "correct_status": None,  # TODO: What status code?
    },
    "inference_exception": {
        "description": "Model.predict() throws an unexpected error",
        "correct_status": None,  # TODO: What status code?
    },
}

# === TODO 5: Fill in the correct HTTP status codes ===
# Reference the rest-api-design.md for the status code table:
# - 200: Successful response
# - 400: Bad Request (malformed syntax)
# - 422: Unprocessable Entity (valid JSON but fails validation)
# - 500: Internal Server Error (server-side failure)
# - 503: Service Unavailable (temporary inability to handle request)

def get_status_codes() -> Dict[str, int]:
    """
    Map scenarios to their correct HTTP status codes.
    
    Returns:
        Dictionary mapping scenario names to status codes
    """
    return {
        "successful_prediction": 0,      # TODO: Replace with correct code
        "missing_required_field": 0,     # TODO: Replace with correct code
        "invalid_field_type": 0,         # TODO: Replace with correct code
        "model_not_loaded": 0,           # TODO: Replace with correct code
        "inference_exception": 0,        # TODO: Replace with correct code
    }


# =============================================================================
# BONUS: COMPLETE FASTAPI APPLICATION (Optional - if you have FastAPI installed)
# =============================================================================

# This is a complete working FastAPI app you can run after completing the TODOs.
# Uncomment and run with: uvicorn fastapi:app --reload

"""
from fastapi import FastAPI, HTTPException

app = FastAPI(
    title="Iris Classifier API",
    description="Lab: Module 2 - FastAPI Inference",
    version="1.0.0"
)

# Eager load model at startup
model = SimulatedIrisModel()

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    features = [[
        request.sepal_length,
        request.sepal_width,
        request.petal_length,
        request.petal_width
    ]]
    
    prediction = model.predict(features)[0]
    probas = model.predict_proba(features)[0]
    confidence = max(probas)
    
    return PredictionResponse(
        prediction=prediction,
        confidence=confidence,
        model_version=model.version
    )
"""


# =============================================================================
# EXPECTED OUTPUT
# =============================================================================
"""
When completed correctly, running this script should output:

=============================================================
=== Part 1: Pydantic Schemas ===
=============================================================
✅ PredictionRequest has all required fields
✅ PredictionRequest fields are properly typed
✅ PredictionResponse has all required fields
✅ PredictionResponse fields are properly typed

=============================================================
=== Part 2: Model Loading Pattern ===
=============================================================
✅ Model loaded at module level (eager loading)
✅ Model is a SimulatedIrisModel instance

=============================================================
=== Part 3: Prediction Endpoint ===
=============================================================
Testing prediction for setosa sample...
✅ Prediction: setosa (confidence: 0.95)
Testing prediction for versicolor sample...
✅ Prediction: versicolor (confidence: 0.85)
Testing prediction for virginica sample...
✅ Prediction: virginica (confidence: 0.90)

=============================================================
=== Part 4: HTTP Status Codes ===
=============================================================
✅ successful_prediction -> 200
✅ missing_required_field -> 422
✅ invalid_field_type -> 422
✅ model_not_loaded -> 503
✅ inference_exception -> 500

=============================================================
=== FINAL RESULT ===
=============================================================
✅ All checks passed! You're ready for Milestone 1!
"""


# =============================================================================
# SELF-CHECK (Do not modify below this line)
# =============================================================================

def validate_pydantic_schemas() -> Tuple[bool, List[str]]:
    """Validate Part 1: Pydantic schemas are correctly defined."""
    errors = []
    
    # Check PredictionRequest
    request_fields = {"sepal_length", "sepal_width", "petal_length", "petal_width"}
    if PYDANTIC_AVAILABLE:
        actual_fields = set(PredictionRequest.model_fields.keys())
    else:
        # Check class annotations for non-pydantic case
        actual_fields = set(getattr(PredictionRequest, '__annotations__', {}).keys())
    
    if not request_fields.issubset(actual_fields):
        missing = request_fields - actual_fields
        errors.append(f"PredictionRequest missing fields: {missing}")
    
    # Check PredictionResponse
    response_fields = {"prediction", "confidence", "model_version"}
    if PYDANTIC_AVAILABLE:
        actual_response_fields = set(PredictionResponse.model_fields.keys())
    else:
        actual_response_fields = set(getattr(PredictionResponse, '__annotations__', {}).keys())
    
    if not response_fields.issubset(actual_response_fields):
        missing = response_fields - actual_response_fields
        errors.append(f"PredictionResponse missing fields: {missing}")
    
    return len(errors) == 0, errors


def validate_model_loading() -> Tuple[bool, List[str]]:
    """Validate Part 2: Model is eagerly loaded."""
    errors = []
    
    if model is None:
        errors.append("Model not loaded at module level (model is None)")
    elif not isinstance(model, SimulatedIrisModel):
        errors.append(f"Model is wrong type: {type(model)}, expected SimulatedIrisModel")
    
    return len(errors) == 0, errors


def validate_prediction_endpoint() -> Tuple[bool, List[str]]:
    """Validate Part 3: Prediction function works correctly."""
    errors = []
    
    if model is None:
        errors.append("Cannot test prediction - model not loaded")
        return False, errors
    
    test_cases = [
        ("setosa", SAMPLE_FEATURES["setosa"], "setosa", 0.95),
        ("versicolor", SAMPLE_FEATURES["versicolor"], "versicolor", 0.85),
        ("virginica", SAMPLE_FEATURES["virginica"], "virginica", 0.90),
    ]
    
    for name, features, expected_pred, expected_conf in test_cases:
        try:
            result = make_prediction(features)
            
            if result is None or not isinstance(result, dict):
                errors.append(f"{name}: make_prediction returned {type(result)}, expected dict")
                continue
            
            if "prediction" not in result:
                errors.append(f"{name}: Response missing 'prediction' field")
            elif result["prediction"] != expected_pred:
                errors.append(f"{name}: Expected prediction '{expected_pred}', got '{result.get('prediction')}'")
            
            if "confidence" not in result:
                errors.append(f"{name}: Response missing 'confidence' field")
            elif result.get("confidence") is None:
                errors.append(f"{name}: Confidence is None")
            
            if "model_version" not in result:
                errors.append(f"{name}: Response missing 'model_version' field")
                
        except Exception as e:
            errors.append(f"{name}: Exception during prediction - {e}")
    
    return len(errors) == 0, errors


def validate_status_codes() -> Tuple[bool, List[str]]:
    """Validate Part 4: HTTP status codes are correct."""
    errors = []
    
    expected = {
        "successful_prediction": 200,
        "missing_required_field": 422,
        "invalid_field_type": 422,
        "model_not_loaded": 503,
        "inference_exception": 500,
    }
    
    student_codes = get_status_codes()
    
    for scenario, expected_code in expected.items():
        actual_code = student_codes.get(scenario, 0)
        if actual_code != expected_code:
            errors.append(f"{scenario}: Expected {expected_code}, got {actual_code}")
    
    return len(errors) == 0, errors


def run_all_checks():
    """Run all validation checks and report results."""
    all_passed = True
    
    print("\n" + "="*60)
    print("=== Part 1: Pydantic Schemas ===")
    print("="*60)
    
    passed, errors = validate_pydantic_schemas()
    if passed:
        print("✅ PredictionRequest has all required fields")
        print("✅ PredictionRequest fields are properly typed")
        print("✅ PredictionResponse has all required fields")
        print("✅ PredictionResponse fields are properly typed")
    else:
        for error in errors:
            print(f"❌ {error}")
        all_passed = False
    
    print("\n" + "="*60)
    print("=== Part 2: Model Loading Pattern ===")
    print("="*60)
    
    passed, errors = validate_model_loading()
    if passed:
        print("✅ Model loaded at module level (eager loading)")
        print("✅ Model is a SimulatedIrisModel instance")
    else:
        for error in errors:
            print(f"❌ {error}")
        all_passed = False
    
    print("\n" + "="*60)
    print("=== Part 3: Prediction Endpoint ===")
    print("="*60)
    
    passed, errors = validate_prediction_endpoint()
    if passed:
        for name in ["setosa", "versicolor", "virginica"]:
            features = SAMPLE_FEATURES[name]
            try:
                result = make_prediction(features)
                print(f"Testing prediction for {name} sample...")
                print(f"✅ Prediction: {result['prediction']} (confidence: {result['confidence']:.2f})")
            except Exception as e:
                print(f"❌ {name}: {e}")
    else:
        for error in errors:
            print(f"❌ {error}")
        all_passed = False
    
    print("\n" + "="*60)
    print("=== Part 4: HTTP Status Codes ===")
    print("="*60)
    
    passed, errors = validate_status_codes()
    student_codes = get_status_codes()
    expected = {
        "successful_prediction": 200,
        "missing_required_field": 422,
        "invalid_field_type": 422,
        "model_not_loaded": 503,
        "inference_exception": 500,
    }
    
    if passed:
        for scenario, code in expected.items():
            print(f"✅ {scenario} -> {code}")
    else:
        for scenario, expected_code in expected.items():
            actual = student_codes.get(scenario, 0)
            if actual == expected_code:
                print(f"✅ {scenario} -> {expected_code}")
            else:
                print(f"❌ {scenario}: Expected {expected_code}, got {actual}")
        all_passed = False
    
    print("\n" + "="*60)
    print("=== FINAL RESULT ===")
    print("="*60)
    
    if all_passed:
        print("✅ All checks passed! You're ready for Milestone 1!")
    else:
        print("❌ Some checks failed. Review the TODOs and try again.")
        print("\nHints:")
        print("  - Part 1: Use Field(..., gt=0) for required positive floats")
        print("  - Part 2: Initialize model = SimulatedIrisModel() at module level")
        print("  - Part 3: Extract features as [[f1, f2, f3, f4]] and call model.predict()")
        print("  - Part 4: Review rest-api-design.md status code table")
    
    return all_passed


if __name__ == "__main__":
    success = run_all_checks()
    # Exit with appropriate code for CI integration
    exit(0 if success else 1)