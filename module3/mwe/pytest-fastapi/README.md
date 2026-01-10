# pytest FastAPI Testing MWE

**Pattern:** Test FastAPI ML inference endpoints with pytest using fixtures, parametrization, and the TestClient.

## Prerequisites

- Python 3.9+
- No external services required

## Quick Start (3 Steps)

```bash
# 1. Clone and navigate
git clone <repo-url>
cd src/module3/learning-activities/mwe/pytest-fastapi

# 2. Install dependencies
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Run the tests
pytest -v
```

## Expected Output

```
========================= test session starts =========================
platform linux -- Python 3.11.x, pytest-8.x.x
collected 8 items

test_app.py::test_health_endpoint PASSED                         [ 12%]
test_app.py::test_predict_valid_input PASSED                     [ 25%]
test_app.py::test_predict_returns_valid_class PASSED             [ 37%]
test_app.py::test_predict_returns_confidence PASSED              [ 50%]
test_app.py::test_predict_invalid_type PASSED                    [ 62%]
test_app.py::test_predict_missing_field PASSED                   [ 75%]
test_app.py::test_predict_empty_body PASSED                      [ 87%]
test_app.py::test_predict_parametrized[setosa] PASSED            [100%]

========================== 8 passed in 0.45s ==========================
```

## Key Concepts Demonstrated

| Concept | Implementation |
|---------|----------------|
| Test fixtures | `@pytest.fixture` for reusable test client and sample data |
| FastAPI TestClient | `TestClient(app)` for HTTP request simulation |
| Response validation | Checking status codes, JSON structure, and data types |
| Parametrized tests | `@pytest.mark.parametrize` for testing multiple inputs |
| Error case testing | Validating 422 responses for invalid inputs |

## Files Included

| File | Purpose |
|------|---------|
| [`main.py`](main.py) | Sample FastAPI ML inference application |
| [`test_app.py`](test_app.py) | Comprehensive pytest test suite |
| [`requirements.txt`](requirements.txt) | Dependencies (FastAPI, pytest, httpx) |

## Project Structure

```
pytest-fastapi/
├── main.py           # FastAPI application
├── test_app.py       # Test suite
├── requirements.txt  # Dependencies
├── expected_output/
│   └── sample_output.txt
└── README.md
```

## Understanding the Tests

### Fixture: Test Client

```python
@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)
```

The `TestClient` wraps the FastAPI app, allowing you to make HTTP requests without running a server. The fixture runs before each test that requests it.

### Fixture: Sample Input Data

```python
@pytest.fixture
def valid_iris_input():
    """Sample valid input for iris classification."""
    return {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
```

Fixtures keep test data DRY (Don't Repeat Yourself) and make tests more readable.

### Basic Request Test

```python
def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}
```

### Response Structure Validation

```python
def test_predict_returns_confidence(client, valid_iris_input):
    response = client.post("/predict", json=valid_iris_input)
    data = response.json()
    
    assert "confidence" in data
    assert isinstance(data["confidence"], float)
    assert 0.0 <= data["confidence"] <= 1.0
```

### Parametrized Testing

```python
@pytest.mark.parametrize("expected_class", ["setosa", "versicolor", "virginica"])
def test_predict_parametrized(client, valid_iris_input, expected_class):
    response = client.post("/predict", json=valid_iris_input)
    prediction = response.json()["prediction"]
    assert prediction in ["setosa", "versicolor", "virginica"]
```

## Extension Challenge

 **Add mock model testing:**

1. Create a mock model that returns predictable results
2. Use `app.dependency_overrides` to inject the mock in tests
3. Test that specific inputs return expected predictions

Example pattern:

```python
from unittest.mock import MagicMock

@pytest.fixture
def mock_model():
    model = MagicMock()
    model.predict.return_value = ["setosa"]
    model.predict_proba.return_value = [[0.9, 0.05, 0.05]]
    return model

@pytest.fixture
def client_with_mock(mock_model):
    app.dependency_overrides[get_model] = lambda: mock_model
    yield TestClient(app)
    app.dependency_overrides.clear()
```

## Running with Coverage

```bash
# Install coverage
pip install pytest-cov

# Run with coverage report
pytest --cov=. --cov-report=term-missing

# Generate HTML report
pytest --cov=. --cov-report=html
# Open htmlcov/index.html in browser
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'main'` | Run pytest from the MWE directory |
| `ImportError: cannot import name 'app'` | Check that main.py exists and defines `app` |
| Tests pass locally but fail in CI | Verify Python version matches (3.11) |

## Related Materials

- Testing ML Services for CI/CD
- FastAPI Fundamentals
- Pydantic Request/Response Models