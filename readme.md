# ❤️ Heart Disease Prediction MLOps Pipeline

## Overview

This project demonstrates an end-to-end Machine Learning Operations (MLOps) workflow for predicting heart disease risk using the UCI Heart Disease Dataset.

The pipeline covers:

* Data ingestion and preprocessing
* Feature engineering
* Model training and evaluation
* MLflow experiment tracking
* Automated testing with Pytest
* CI/CD with GitHub Actions
* FastAPI model serving
* Docker containerization
* Kubernetes deployment
* Prometheus monitoring and logging

---

# Project Architecture

```text
Dataset
   │
   ▼
Data Ingestion
   │
   ▼
Preprocessing
   │
   ▼
Feature Engineering
   │
   ▼
Model Training
   │
   ▼
Model Evaluation
   │
   ▼
MLflow Tracking
   │
   ▼
Model Artifacts (.pkl)
   │
   ▼
FastAPI Application
   │
   ▼
Docker Container
   │
   ▼
Kubernetes Deployment
   │
   ▼
Monitoring (Prometheus)
```

---

# Project Structure

```text
heart-disease-mlops/
│
├── app.py
├── Dockerfile
├── docker-compose.yml
├── deployment.yaml
├── service.yaml
├── prometheus.yml
├── requirements.txt
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── notebooks/
│   └── MLOps_assignment_1.ipynb
│
├── src/
│   ├── data_ingestion.py
│   ├── preprocess.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── mlflow_tracking.py
│   └── train_pipeline.py
│
├── models/
│   ├── heart_disease_model.pkl
│   ├── scaler.pkl
│   ├── feature_columns.pkl
│   └── imputer_values.pkl
│
├── tests/
│   ├── test_app.py
│   ├── test_data_ingestion.py
│   ├── test_preprocess.py
│   ├── test_feature_engineering.py
│   ├── test_train.py
│   ├── test_evaluate.py
│   ├── test_predict.py
│   ├── test_mlflow_tracking.py
│   ├── test_model.py
│   └── test_train_pipeline.py
│
└── mlruns/
    └── MLflow experiment artifacts
```

---

# Dataset

Dataset: UCI Heart Disease Dataset

Target Variable:

* 0 → No Heart Disease
* 1 → Heart Disease Present

Features include:

* Age
* Sex
* Chest Pain Type
* Blood Pressure
* Cholesterol
* Fasting Blood Sugar
* ECG Results
* Maximum Heart Rate
* Exercise Induced Angina
* ST Depression
* Thalassemia
* And other clinical measurements

---

# Setup Instructions

## Clone Repository

```bash
git clone https://github.com/4chaksu/heart-disease-mlops.git

cd heart-disease-mlops
```

## Create Virtual Environment

### Windows

```bash
python -m venv venv

venv\Scripts\activate
```

### Linux / Mac

```bash
python -m venv venv

source venv/bin/activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Train Model

Run the complete pipeline:

```bash
python -m src.train_pipeline
```

Pipeline Steps:

1. Data Ingestion
2. Data Preprocessing
3. Feature Engineering
4. Model Training
5. Model Evaluation
6. MLflow Logging
7. Model Serialization

Generated Artifacts:

```text
models/
├── heart_disease_model.pkl
├── scaler.pkl
├── feature_columns.pkl
└── imputer_values.pkl
```

---

# MLflow Experiment Tracking

Launch MLflow UI:

```bash
mlflow ui
```

Open:

```text
http://localhost:5000
```

Tracked Information:

* Hyperparameters
* Evaluation Metrics
* Model Artifacts
* Experiment Runs

---

# Running Tests

Run all tests:

```bash
pytest
```

Run with coverage:

```bash
pytest --cov=src --cov-report=term
```

Current test suite includes:

* Data ingestion tests
* Preprocessing tests
* Feature engineering tests
* Model training tests
* Evaluation tests
* Prediction tests
* MLflow tests
* API tests
* End-to-end pipeline tests

---

# FastAPI Model Serving

Start API:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

Swagger Documentation:

```text
http://localhost:8000/docs
```

Health Check:

```text
GET /
```

Prediction Endpoint:

```text
POST /predict
```

Sample Request:

```json
{
  "age": 63,
  "sex": 1,
  "cp": 3,
  "trestbps": 145,
  "chol": 233,
  "fbs": 1,
  "restecg": 0,
  "thalach": 150,
  "exang": 0,
  "oldpeak": 2.3,
  "slope": 0,
  "ca": 0,
  "thal": 1
}
```

Sample Response:

```json
{
  "prediction": 1,
  "confidence": 0.94
}
```

---

# Docker

Build Image:

```bash
docker build -t heart-api .
```

Run Container:

```bash
docker run -p 8000:8000 heart-api
```

Verify:

```text
http://localhost:8000/docs
```

---

# Docker Compose

Start Application Stack:

```bash
docker-compose up --build
```

Stop Stack:

```bash
docker-compose down
```

---

# Kubernetes Deployment

Deploy Application:

```bash
kubectl apply -f deployment.yaml

kubectl apply -f service.yaml
```

Verify:

```bash
kubectl get pods

kubectl get services
```

---

# Monitoring

Prometheus configuration:

```text
prometheus.yml
```

Run Prometheus:

```bash
docker run -p 9090:9090 \
-v ${PWD}/prometheus.yml:/etc/prometheus/prometheus.yml \
prom/prometheus
```

Prometheus Dashboard:

```text
http://localhost:9090
```

Monitoring Features:

* API request count
* Request latency
* Error tracking
* Service availability

---

# CI/CD Pipeline

GitHub Actions workflow:

```text
.github/workflows/ci.yml
```

Pipeline Stages:

1. Checkout Repository
2. Install Dependencies
3. Code Linting
4. Unit Testing
5. Coverage Validation
6. Model Training
7. Build Verification

Triggered On:

* Push
* Pull Request

---

# Technologies Used

* Python
* Scikit-Learn
* Pandas
* NumPy
* FastAPI
* MLflow
* Pytest
* Docker
* Kubernetes
* Prometheus
* GitHub Actions

---

# Author

**Vinay Bora**

Data Scientist | M.Tech AI & ML

BITS WILP | Fractal Analytics

---
