"""
Simple FastAPI Application for Docker Multistage Build Demonstration.

This minimal API service demonstrates containerization patterns:
- Health check endpoint for orchestration
- Prediction endpoint simulating ML inference
- Proper request/response models with Pydantic
"""

from fastapi import FastAPI
from pydantic import BaseModel, Field
import os

app = FastAPI(
    title="ML Inference Service",
    description="Containerized ML inference endpoint",
    version="1.0.0"
)


class PredictionInput(BaseModel):
    """Input schema for prediction requests."""
    
    features: list[float] = Field(
        ..., 
        min_length=1,
        description="List of numeric features for prediction"
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [{"features": [1.0, 2.0, 3.0, 4.0]}]
        }
    }


class PredictionResponse(BaseModel):
    """Output schema for prediction results."""
    
    prediction: str = Field(..., description="Predicted class label")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence")
    container_id: str = Field(..., description="Container hostname for debugging")


class HealthResponse(BaseModel):
    """Health check response schema."""
    
    status: str
    version: str
    container_id: str


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """
    Health check endpoint for container orchestration.
    
    Used by Docker HEALTHCHECK, Kubernetes probes, and load balancers
    to verify the service is running and ready to accept requests.
    """
    return HealthResponse(
        status="healthy",
        version=app.version,
        container_id=os.environ.get("HOSTNAME", "local")
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(input_data: PredictionInput) -> PredictionResponse:
    """
    Simulate ML model prediction.
    
    In production, this would load a trained model and run inference.
    For demonstration, we use a simple rule-based prediction.
    """
    features = input_data.features
    
    # Simple mock prediction logic
    feature_sum = sum(features)
    if feature_sum < 10:
        prediction, confidence = "class_a", 0.85
    elif feature_sum < 20:
        prediction, confidence = "class_b", 0.78
    else:
        prediction, confidence = "class_c", 0.92
    
    return PredictionResponse(
        prediction=prediction,
        confidence=confidence,
        container_id=os.environ.get("HOSTNAME", "local")
    )


@app.get("/")
def root():
    """Root endpoint with API information."""
    return {
        "service": "ML Inference API",
        "version": app.version,
        "docs": "/docs",
        "health": "/health"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)