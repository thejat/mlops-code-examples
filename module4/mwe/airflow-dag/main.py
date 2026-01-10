#!/usr/bin/env python3
"""
Airflow DAG Minimum Working Example

Demonstrates: DAG definition, PythonOperator, task dependencies, and XCom.
This DAG simulates a simple ML training pipeline with 3 stages.

Run with: airflow dags test train_pipeline 2024-01-01
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta


# Default arguments applied to all tasks
default_args = {
    'owner': 'mlops',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}


def preprocess_data(**context):
    """Task 1: Simulate data preprocessing."""
    print("=" * 50)
    print("PREPROCESS_DATA: Starting data preprocessing...")
    
    # Simulate preprocessing work
    n_samples = 1000
    data_path = "/tmp/processed_data.csv"
    
    print(f"  - Loaded {n_samples} samples")
    print(f"  - Cleaned missing values")
    print(f"  - Saved to {data_path}")
    print("PREPROCESS_DATA: Complete!")
    print("=" * 50)
    
    # Return value is automatically pushed to XCom
    return {'data_path': data_path, 'n_samples': n_samples}


def train_model(**context):
    """Task 2: Simulate model training using data from Task 1."""
    print("=" * 50)
    print("TRAIN_MODEL: Starting model training...")
    
    # Pull results from upstream task via XCom
    ti = context['ti']
    preprocess_result = ti.xcom_pull(task_ids='preprocess_data')
    
    if preprocess_result:
        print(f"  - Received from preprocess: {preprocess_result}")
        data_path = preprocess_result['data_path']
        n_samples = preprocess_result['n_samples']
    else:
        # Fallback for standalone testing
        data_path = "/tmp/processed_data.csv"
        n_samples = 1000
    
    # Simulate training
    model_path = "/tmp/model.pkl"
    accuracy = 0.92
    
    print(f"  - Training on {n_samples} samples")
    print(f"  - Model accuracy: {accuracy}")
    print(f"  - Saved model to {model_path}")
    print("TRAIN_MODEL: Complete!")
    print("=" * 50)
    
    return {'model_path': model_path, 'accuracy': accuracy}


def register_model(**context):
    """Task 3: Simulate model registration using results from Task 2."""
    print("=" * 50)
    print("REGISTER_MODEL: Starting model registration...")
    
    # Pull results from upstream task via XCom
    ti = context['ti']
    train_result = ti.xcom_pull(task_ids='train_model')
    
    if train_result:
        print(f"  - Received from train: {train_result}")
        accuracy = train_result['accuracy']
        model_path = train_result['model_path']
    else:
        accuracy = 0.92
        model_path = "/tmp/model.pkl"
    
    # Simulate registration
    model_version = "v1.0"
    print(f"  - Model accuracy threshold check: {accuracy} >= 0.8 ✓")
    print(f"  - Registered as version: {model_version}")
    print("REGISTER_MODEL: Complete!")
    print("=" * 50)
    
    return {'version': model_version, 'registered': True}


# Define the DAG
with DAG(
    dag_id='train_pipeline',
    default_args=default_args,
    description='Simple ML training pipeline demonstrating DAG concepts',
    schedule_interval='@daily',
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['mwe', 'ml', 'training'],
) as dag:
    
    # Define tasks
    preprocess = PythonOperator(
        task_id='preprocess_data',
        python_callable=preprocess_data,
    )
    
    train = PythonOperator(
        task_id='train_model',
        python_callable=train_model,
    )
    
    register = PythonOperator(
        task_id='register_model',
        python_callable=register_model,
    )
    
    # Set dependencies using bitshift operator
    # This creates: preprocess -> train -> register
    preprocess >> train >> register