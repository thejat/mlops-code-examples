# FastAPI Inference MWE

**Pattern:** Serve ML model predictions via REST API with automatic input validation and interactive documentation.

## Prerequisites

- Python 3.9+
- Linux, macOS, or Windows
- ~50MB disk space

## Quick Start (3 Steps)

```bash
# 1. Clone and navigate
git clone <repo-url>
cd src/module2/learning-activities/mwe/fastapi-inference

# 2. Install dependencies
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Run the server
uvicorn main:app --reload --port 8000
```

## Expected Output

### Server Startup

```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started reloader process [12345] using StatReload
INFO:     Started server process [12346]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

### Test the API

**Health Check:**
```bash
curl http://localhost:8000/health
```
```json
{"status": "healthy", "model_loaded": true}
```

**Prediction Request:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'
```
```json
{"prediction": "setosa", "confidence": 0.95, "model_version": "1.0.0"}
```

### Interactive Documentation

Open [http://localhost:8000/docs](http://localhost:8000/docs) for Swagger UI.

![Swagger UI Screenshot](expected_output/swagger_ui.png)

## Key Concepts Demonstrated

| Concept | Implementation |
|---------|----------------|
| Pydantic validation | `PredictionRequest` with `Field` constraints |
| Response schema | `response_model=PredictionResponse` |
| Health endpoint | `GET /health` for container orchestration |
| Model loading | Module-level (loaded once at startup) |

## Extension Challenge

**Add a `/batch` endpoint** that accepts a list of prediction requests and returns a list of predictions:

```python
@app.post("/batch", response_model=list[PredictionResponse])
def batch_predict(requests: list[PredictionRequest]):
    # Your implementation here
    pass
```

Test it with:
```bash
curl -X POST http://localhost:8000/batch \
  -H "Content-Type: application/json" \
  -d '[{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2},
       {"sepal_length": 6.7, "sepal_width": 3.0, "petal_length": 5.2, "petal_width": 2.3}]'
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'fastapi'` | Activate venv: `source .venv/bin/activate` |
| Port 8000 already in use | Use different port: `--port 8001` |
| Connection refused | Check server is running and firewall allows port |

## Related Materials

- [FastAPI Fundamentals](../../instructional-materials/fastapi-fundamentals.md)
- [Pydantic Request/Response Models](../../instructional-materials/pydantic-schemas.md)
- [REST API Design Principles](../../instructional-materials/rest-api-design.md)