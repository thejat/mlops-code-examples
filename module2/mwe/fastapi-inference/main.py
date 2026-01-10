"""
FastAPI Inference MWE - Demonstrates ML model serving pattern.

This minimal example shows:
1. Health check endpoint (GET /health)
2. Prediction endpoint (POST /predict)
3. Pydantic request/response validation
4. Model loading at startup (simulated)

Run with: uvicorn main:app --reload --port 8000
Test at: http://localhost:8000/docs
"""

from fastapi import FastAPI
from pydantic import BaseModel, Field

# Initialize FastAPI application
app = FastAPI(
    title="Iris Classifier API",
    description="MWE: FastAPI ML serving pattern",
    version="1.0.0",
)


# --- Pydantic Schemas ---
class PredictionRequest(BaseModel):
    """Input schema for prediction endpoint."""

    sepal_length: float = Field(..., ge=0, le=10, example=5.1)
    sepal_width: float = Field(..., ge=0, le=10, example=3.5)
    petal_length: float = Field(..., ge=0, le=10, example=1.4)
    petal_width: float = Field(..., ge=0, le=10, example=0.2)


class PredictionResponse(BaseModel):
    """Output schema for prediction endpoint."""

    prediction: str
    confidence: float
    model_version: str


# --- Simulated Model (replace with joblib.load in production) ---
class MockIrisClassifier:
    """Mock classifier for demonstration - no sklearn dependency."""

    def predict(self, features: list) -> str:
        """Simple rule-based classification for demo."""
        petal_length = features[0][2]
        if petal_length < 2.5:
            return "setosa"
        elif petal_length < 4.9:
            return "versicolor"
        return "virginica"

    def predict_proba(self, features: list) -> float:
        """Return mock confidence score."""
        return 0.95


# Load model at module level (loaded once at startup)
model = MockIrisClassifier()


# --- Endpoints ---
@app.get("/health")
def health_check():
    """Health check endpoint for container orchestration."""
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """
    Generate prediction from input features.

    - Validates input via Pydantic schema
    - Returns prediction with confidence score
    """
    features = [
        [
            request.sepal_length,
            request.sepal_width,
            request.petal_length,
            request.petal_width,
        ]
    ]

    prediction = model.predict(features)
    confidence = model.predict_proba(features)

    return PredictionResponse(
        prediction=prediction,
        confidence=confidence,
        model_version="1.0.0",
    )


# --- Entry point for direct execution ---
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)