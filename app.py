from pathlib import Path

import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "tea_leaf_v1"

FEATURES = [
    "worker_exp_yr",
    "rain_mm_week",
    "slope_deg",
    "temp_c",
    "bush_age_yr",
    "hours_worked",
    "flush",
    "section",
]

# Load the trained MLflow model
model = mlflow.sklearn.load_model(str(MODEL_PATH))

# Create FastAPI application
app = FastAPI(
    title="Tea Leaf Yield Prediction API",
    description="Predict tea leaf yield in kg per worker.",
    version="1.0.0",
)


# Define input data format
class PredictionInput(BaseModel):
    worker_exp_yr: float
    rain_mm_week: float
    slope_deg: float
    temp_c: float
    bush_age_yr: float
    hours_worked: float
    flush: str
    section: str


# Homepage
@app.get("/")
def home():
    return {
        "message": "Tea Leaf Yield Prediction API is running"
    }


# Health check
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


# Prediction endpoint
@app.post("/predict")
def predict(data: PredictionInput):

    input_data = pd.DataFrame(
        [[getattr(data, feature) for feature in FEATURES]],
        columns=FEATURES,
    )

    prediction = model.predict(input_data)[0]

    return {
        "predicted_yield_kg_per_worker": round(float(prediction), 2)
    }