# CI/CD Model Validation - Minimal Working Example

This example demonstrates model validation with quality gates in a CI/CD context. The validation script checks if a trained model meets minimum performance thresholds.

## What You'll Learn

- How to implement threshold-based model validation
- How to generate validation reports
- How to fail a CI/CD pipeline when quality gates aren't met
- How to retrieve metrics from MLflow for validation

## Files

| File | Purpose |
|------|---------|
| `train.py` | Trains a model and saves metrics |
| `validate.py` | Validates model against thresholds |
| `sample_workflow.yml` | Example GitHub Actions workflow |
| `requirements.txt` | Python dependencies |
| `expected_output/` | Expected console output |

## Prerequisites

- Python 3.9+
- pip
- (Optional) MLflow tracking server running

## Quick Start

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# OR: venv\Scripts\activate  # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train a model (creates metrics.json)
python train.py

# 4. Validate with passing thresholds
python validate.py --min-accuracy 0.90 --min-f1 0.85
# Expected: PASSED

# 5. Validate with failing thresholds (too strict)
python validate.py --min-accuracy 0.99 --min-f1 0.99
# Expected: FAILED (exit code 1)
```

## Expected Behavior

### Passing Validation

```
$ python validate.py --min-accuracy 0.90 --min-f1 0.85

=== Model Validation Report ===
✓ PASS: accuracy = 0.9667 (threshold: 0.9000)
✓ PASS: f1_score = 0.9665 (threshold: 0.8500)

Overall: PASSED
Validation report saved to: validation_report.json
```

### Failing Validation

```
$ python validate.py --min-accuracy 0.99 --min-f1 0.99

=== Model Validation Report ===
✗ FAIL: accuracy = 0.9667 (threshold: 0.9900)
✗ FAIL: f1_score = 0.9665 (threshold: 0.9900)

Overall: FAILED
Validation report saved to: validation_report.json
$ echo $?
1
```

The exit code `1` causes the CI/CD pipeline to fail.

## How It Works

### 1. Train and Save Metrics

```python
# train.py saves metrics to a file
metrics = {
    "accuracy": 0.9667,
    "f1_score": 0.9665,
    "training_time": 0.15
}
with open("metrics.json", "w") as f:
    json.dump(metrics, f)
```

### 2. Validate Against Thresholds

```python
# validate.py checks thresholds and exits with appropriate code
if accuracy >= min_accuracy:
    print("✓ PASS")
else:
    print("✗ FAIL")
    passed = False

if not passed:
    sys.exit(1)  # Fails the CI/CD pipeline
```

### 3. GitHub Actions Integration

```yaml
- name: Validate model
  run: |
    python validate.py \
      --min-accuracy 0.90 \
      --min-f1 0.85
# If validation fails, the workflow stops here
```

## Customizing Thresholds

You can adjust thresholds via command line:

```bash
# Strict thresholds
python validate.py --min-accuracy 0.95 --min-f1 0.93

# Lenient thresholds
python validate.py --min-accuracy 0.80 --min-f1 0.75
```

Or via environment variables in CI/CD:

```yaml
env:
  MIN_ACCURACY: 0.90
  MIN_F1: 0.85
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "metrics.json not found" | Run `train.py` first |
| Validation always fails | Lower the thresholds |
| Exit code not captured | Use `echo $?` (bash) or `echo %ERRORLEVEL%` (Windows) |

## Next Steps

- Integrate with your GitHub Actions workflow
- Add additional metrics (latency, model size)
- Implement relative validation (compare to production model)

## Related Materials

- CI/CD for ML Validation
- GitHub Actions for ML Pipelines