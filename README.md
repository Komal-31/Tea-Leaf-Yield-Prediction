# 🍃 Tea Leaf Yield Prediction

Predict tea leaf yield (kg per worker) based on plantation conditions, worker experience, and environmental features.

**Course:** SCSE3040 — MLOps | **Group:** G055 | **Problem ID:** MP148

---

## Quick Start

### Prerequisites

- Python 3.12+
- Docker Desktop (for containerized deployment)
- kubectl + Kubernetes cluster (for orchestrated deployment)

### Local Development

```bash
python -m pip install -r requirements.txt

# Run FastAPI server
uvicorn app:app --host 0.0.0.0 --port 8000

# Run Streamlit frontend
streamlit run streamlit_app.py

# Run tests
python -m pytest tests/test_model_artifact.py tests/test_api.py tests/test_deployment.py -v
```

### Docker Deployment

```bash
docker build -t tea-leaf-api:1.0.0 .
docker run -d --name tea-leaf-api -p 8000:8000 tea-leaf-api:1.0.0
```

### Kubernetes Deployment

```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl rollout status deployment/tea-leaf-api --timeout=180s
kubectl port-forward service/tea-leaf-api-service 8080:80
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Home — service status |
| GET | `/health` | Health check |
| GET | `/ready` | Readiness probe |
| POST | `/predict` | Predict tea leaf yield |
| GET | `/docs` | Swagger documentation |

### Sample Prediction Request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "worker_exp_yr": 48,
    "rain_mm_week": 285,
    "slope_deg": 21,
    "temp_c": 15.9,
    "bush_age_yr": 10,
    "hours_worked": 5.4,
    "flush": "monsoon",
    "section": "River Side"
  }'
```

**Response:**

```json
{"predicted_yield_kg_per_worker": 25.28}
```

---

## Project Structure

```
├── app.py                      # FastAPI application with logging
├── streamlit_app.py            # Streamlit frontend
├── Dockerfile                  # Production Docker image
├── .dockerignore               # Docker build exclusions
├── requirements.txt            # Python dependencies
├── render.yaml                 # Render deployment config
├── models/tea_leaf_v1/         # Trained MLflow model artifacts
├── k8s/
│   ├── deployment.yaml         # Kubernetes Deployment (3 replicas)
│   └── service.yaml            # Kubernetes Service (ClusterIP)
├── tests/
│   ├── test_api.py             # API endpoint tests
│   ├── test_model_artifact.py  # Model artifact tests
│   ├── test_deployment.py      # Docker & Kubernetes manifest tests
│   ├── test_data_quality.py    # Data quality tests (requires datasets)
│   └── test_model.py           # Model tests (requires MLflow DB)
├── docs/
│   └── shubham_deployment.md   # Deployment documentation
├── src/                        # Data cleaning and training scripts
├── data/                       # DVC-tracked datasets
├── notebooks/                  # Exploratory analysis
└── reports/                    # Model comparison reports
```

---

## Team Contributions

| Member | Responsibilities |
|--------|-----------------|
| Ravish | Repository, DVC, data cleaning, model training, MLflow tracking |
| Komal | Model artifact testing, CI, FastAPI serving, Render, Streamlit |
| Shubham | Docker, Kubernetes, logging, CI deployment tests, documentation |
| Anushka | Drift monitoring, retraining, holdout predictions |
