# Airflow DAG - Minimum Working Example

**Pattern:** Define and execute an Airflow DAG with task dependencies and XCom data passing.

---

## Prerequisites

- Python 3.9+ (tested on 3.9, 3.10, 3.11)
- Linux or macOS (Windows requires WSL2)
- ~2GB disk space for Airflow installation

---

## Setup (3 Steps)

### 1. Clone and Navigate

```bash
git clone <repository-url>
cd src/module4/learning-activities/mwe/airflow-dag
```

### 2. Install Dependencies

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install Airflow (constraint file ensures compatible dependencies)
pip install "apache-airflow==2.8.1" \
  --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.8.1/constraints-3.10.txt"
```

### 3. Initialize and Run

```bash
# Set Airflow home to current directory
export AIRFLOW_HOME=$(pwd)

# Initialize the database
airflow db init

# Copy DAG to dags folder
mkdir -p dags
cp main.py dags/train_pipeline.py

# Test the DAG (runs all tasks without scheduler)
airflow dags test train_pipeline 2024-01-01
```

---

## Expected Output

When running `airflow dags test train_pipeline 2024-01-01`, you should see:

```
==================================================
PREPROCESS_DATA: Starting data preprocessing...
  - Loaded 1000 samples
  - Cleaned missing values
  - Saved to /tmp/processed_data.csv
PREPROCESS_DATA: Complete!
==================================================
==================================================
TRAIN_MODEL: Starting model training...
  - Received from preprocess: {'data_path': '/tmp/processed_data.csv', 'n_samples': 1000}
  - Training on 1000 samples
  - Model accuracy: 0.92
  - Saved model to /tmp/model.pkl
TRAIN_MODEL: Complete!
==================================================
==================================================
REGISTER_MODEL: Starting model registration...
  - Received from train: {'model_path': '/tmp/model.pkl', 'accuracy': 0.92}
  - Model accuracy threshold check: 0.92 >= 0.8 ✓
  - Registered as version: v1.0
REGISTER_MODEL: Complete!
==================================================
```

---

## Key Concepts Demonstrated

| Concept | Where in Code |
|---------|---------------|
| DAG Definition | `with DAG(...) as dag:` block |
| PythonOperator | `PythonOperator(task_id='...', python_callable=...)` |
| Task Dependencies | `preprocess >> train >> register` |
| XCom Push | `return {'key': 'value'}` (automatic) |
| XCom Pull | `ti.xcom_pull(task_ids='preprocess_data')` |
| Default Args | `default_args` dictionary with retries, owner |

---

## Extension Challenge

**Add Parallel Preprocessing:** Modify the DAG to have two parallel preprocessing tasks (`preprocess_features` and `preprocess_labels`) that both feed into `train_model`.

```
preprocess_features ──┐
                      ├──▶ train_model ──▶ register_model
preprocess_labels ────┘
```

Hints:
- Create two separate `PythonOperator` tasks
- Both should use `>>` to point to `train_model`
- Modify `train_model` to pull from both using `ti.xcom_pull(task_ids=['task1', 'task2'])`

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: airflow` | Activate your virtual environment |
| DAG not found | Ensure `main.py` is copied to `dags/` folder |
| XCom returns None | Check `task_ids` parameter matches exact task name |
| Permission errors | Ensure `AIRFLOW_HOME` is set to a writable directory |

---

## Resources

- [Airflow DAG Tutorial](https://airflow.apache.org/docs/apache-airflow/stable/tutorial/fundamentals.html)
- [PythonOperator Reference](https://airflow.apache.org/docs/apache-airflow/stable/howto/operator/python.html)
- [XCom Documentation](https://airflow.apache.org/docs/apache-airflow/stable/core-concepts/xcoms.html)