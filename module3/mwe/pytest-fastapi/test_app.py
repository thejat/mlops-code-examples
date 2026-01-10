"""
Test Suite for FastAPI ML Inference Application.

This module demonstrates pytest testing patterns for FastAPI endpoints,
including fixtures, parametrized tests, and validation testing.
"""

import pytest
from fastapi.testclient import TestClient
from main import app


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def valid_iris_input():
    """Sample valid input for iris classification."""
    return {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }


@pytest.fixture
def setosa_sample():
    """Known setosa sample (small petal length)."""
    return {
        "sepal_length": 4.9,
        "sepal_width": 3.0,
        "petal_length": 1.4,
        "petal_width": 0.2
    }


@pytest.fixture
def versicolor_sample():
    """Known versicolor sample (medium petal measurements)."""
    return {
        "sepal_length": 6.0,
        "sepal_width": 2.7,
        "petal_length": 4.5,
        "petal_width": 1.3
    }


@pytest.fixture
def virginica_sample():
    """Known virginica sample (large petal measurements)."""
    return {
        "sepal_length": 6.3,
        "sepal_width": 2.9,
        "petal_length": 5.6,
        "petal_width": 2.1
    }


# =============================================================================
# Health Endpoint Tests
# =============================================================================

def test_health_endpoint(client):
    """Test that health check returns 200 OK with expected response."""
    response = client.get("/health")
    
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


# =============================================================================
# Prediction Endpoint Tests - Happy Path
# =============================================================================

def test_predict_valid_input(client, valid_iris_input):
    """Test prediction with valid input returns 200 and expected structure."""
    response = client.post("/predict", json=valid_iris_input)
    
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "confidence" in data


def test_predict_returns_valid_class(client, valid_iris_input):
    """Test that prediction is one of the expected iris species."""
    response = client.post("/predict", json=valid_iris_input)
    
    valid_classes = ["setosa", "versicolor", "virginica"]
    prediction = response.json()["prediction"]
    
    assert prediction in valid_classes


def test_predict_returns_confidence(client, valid_iris_input):
    """Test that confidence score is a float between 0 and 1."""
    response = client.post("/predict", json=valid_iris_input)
    
    data = response.json()
    confidence = data["confidence"]
    
    assert isinstance(confidence, float)
    assert 0.0 <= confidence <= 1.0


# =============================================================================
# Prediction Endpoint Tests - Error Cases
# =============================================================================

def test_predict_invalid_type(client):
    """Test that invalid input types return 422 Unprocessable Entity."""
    payload = {
        "sepal_length": "not a number",  # Invalid: should be float
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    
    response = client.post("/predict", json=payload)
    
    assert response.status_code == 422


def test_predict_missing_field(client):
    """Test that missing required fields return 422 Unprocessable Entity."""
    payload = {
        "sepal_length": 5.1,
        # Missing: sepal_width, petal_length, petal_width
    }
    
    response = client.post("/predict", json=payload)
    
    assert response.status_code == 422


def test_predict_empty_body(client):
    """Test that empty request body returns 422 Unprocessable Entity."""
    response = client.post("/predict", json={})
    
    assert response.status_code == 422


def test_predict_negative_value(client):
    """Test that negative values (invalid per schema) return 422."""
    payload = {
        "sepal_length": -5.1,  # Invalid: must be > 0
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    
    response = client.post("/predict", json=payload)
    
    assert response.status_code == 422


# =============================================================================
# Parametrized Tests
# =============================================================================

@pytest.mark.parametrize("expected_class", ["setosa", "versicolor", "virginica"])
def test_predict_parametrized(client, valid_iris_input, expected_class):
    """
    Demonstrate parametrized testing.
    
    Note: This test verifies the response contains a valid class,
    not that it matches the parameter. The parameter shows how
    pytest.mark.parametrize works for running the same test multiple times.
    """
    response = client.post("/predict", json=valid_iris_input)
    prediction = response.json()["prediction"]
    
    # Verify the prediction is one of the valid classes
    assert prediction in ["setosa", "versicolor", "virginica"]


@pytest.mark.parametrize("input_fixture,expected_class", [
    ("setosa_sample", "setosa"),
    ("versicolor_sample", "versicolor"),
    ("virginica_sample", "virginica"),
])
def test_specific_predictions(client, input_fixture, expected_class, request):
    """
    Test that specific samples are classified correctly.
    
    This demonstrates using pytest's request fixture to access
    other fixtures dynamically.
    """
    # Get the actual fixture value using request.getfixturevalue
    sample = request.getfixturevalue(input_fixture)
    
    response = client.post("/predict", json=sample)
    prediction = response.json()["prediction"]
    
    assert prediction == expected_class


# =============================================================================
# Response Schema Tests
# =============================================================================

def test_response_schema(client, valid_iris_input):
    """Test that response matches expected JSON schema."""
    response = client.post("/predict", json=valid_iris_input)
    data = response.json()
    
    # Check all expected keys are present
    assert set(data.keys()) == {"prediction", "confidence"}
    
    # Check types
    assert isinstance(data["prediction"], str)
    assert isinstance(data["confidence"], (int, float))


def test_content_type(client, valid_iris_input):
    """Test that response has correct content type."""
    response = client.post("/predict", json=valid_iris_input)
    
    assert response.headers["content-type"] == "application/json"


# =============================================================================
# Edge Case Tests
# =============================================================================

def test_predict_with_zero_values(client):
    """Test handling of zero values (boundary case)."""
    payload = {
        "sepal_length": 0.001,  # Near zero but valid (> 0)
        "sepal_width": 0.001,
        "petal_length": 0.001,
        "petal_width": 0.001
    }
    
    response = client.post("/predict", json=payload)
    
    # Should still return a valid prediction
    assert response.status_code == 200
    assert response.json()["prediction"] in ["setosa", "versicolor", "virginica"]


def test_predict_with_large_values(client):
    """Test handling of large input values."""
    payload = {
        "sepal_length": 100.0,
        "sepal_width": 50.0,
        "petal_length": 75.0,
        "petal_width": 40.0
    }
    
    response = client.post("/predict", json=payload)
    
    # Should still return a valid prediction (model handles extrapolation)
    assert response.status_code == 200
    assert response.json()["prediction"] in ["setosa", "versicolor", "virginica"]