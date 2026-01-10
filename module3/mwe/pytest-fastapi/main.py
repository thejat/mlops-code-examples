"""
Sample FastAPI ML Inference Application for Testing Demonstration.

This module provides a minimal FastAPI application that simulates
an ML inference service. It's designed to demonstrate pytest testing
patterns without requiring an actual trained model.
"""

from fastapi import FastAPI
from pydantic import BaseModel, Field
import random

app = FastAPI(
    title="Iris Classification API",
    description="Sample ML inference endpoint for testing demonstration",
    version="1.0.0"
)


class IrisInput(BaseModel):
    """Input schema for iris classification prediction."""
    
    sepal_length: float = Field(..., gt=0, description="Sepal length in cm")
    sepal_width: float = Field(..., gt=0, description="Sepal width in cm")
    petal_length: float = Field(..., gt=0, description="Petal length in cm")
    petal_width: float = Field(..., gt=0, description="Petal width in cm")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "sepal_length": 5.1,
                    "sepal_width": 3.5,
                    "petal_length": 1.4,
                    "petal_width": 0.2
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    """Output schema for prediction results."""
    
    prediction: str = Field(..., description="Predicted iris species")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence")


def mock_predict(features: list[float]) -> tuple[str, float]:
    """
    Simulate model prediction.
    
    In a real application, this would load and call an actual model.
    For testing purposes, we use a simple rule-based heuristic.
    
    Args:
        features: List of [sepal_length, sepal_width, petal_length, petal_width]
        
    Returns:
        Tuple of (predicted_class, confidence_score)
    """
    sepal_length, sepal_width, petal_length, petal_width = features
    
    # Simple heuristic based on petal measurements
    if petal_length < 2.5:
        prediction = "setosa"
        confidence = 0.95
    elif petal_width < 1.75:
        prediction = "versicolor"
        confidence = 0.85
    else:
        prediction = "virginica"
        confidence = 0.80
    
    # Add small random variation to confidence for realism
    confidence = min(1.0, max(0.0, confidence + random.uniform(-0.05, 0.05)))
    
    return prediction, round(confidence, 4)


@app.get("/health")
def health_check() -> dict:
    """
    Health check endpoint for container orchestration.
    
    Returns:
        Dictionary with status field indicating service health.
    """
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionResponse)
def predict(input_data: IrisInput) -> PredictionResponse:
    """
    Predict iris species from flower measurements.
    
    Args:
        input_data: Iris flower measurements (sepal and petal dimensions)
        
    Returns:
        Prediction with species name and confidence score.
    """
    features = [
        input_data.sepal_length,
        input_data.sepal_width,
        input_data.petal_length,
        input_data.petal_width
    ]
    
    prediction, confidence = mock_predict(features)
    
    return PredictionResponse(
        prediction=prediction,
        confidence=confidence
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)