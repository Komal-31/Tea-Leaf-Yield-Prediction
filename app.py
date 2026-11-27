"""Tea Leaf Yield Prediction API — FastAPI application with production logging."""

import logging
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("tea_leaf_api")

# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
logger.info("Loading model from %s", MODEL_PATH)
try:
    model = mlflow.sklearn.load_model(str(MODEL_PATH))
    logger.info("Model loaded successfully")
except Exception:
    logger.exception("Failed to load model")
    model = None

@asynccontextmanager
async def lifespan(application: FastAPI):
    """Application lifespan manager for startup and shutdown logging."""
    logger.info("Application starting up")
    yield
    logger.info("Application shutting down")


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Tea Leaf Yield Prediction API",
    description="Predict tea leaf yield in kg per worker.",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------
class PredictionInput(BaseModel):
    worker_exp_yr: float
    rain_mm_week: float
    slope_deg: float
    temp_c: float
    bush_age_yr: float
    hours_worked: float
    flush: str
    section: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/")
def home():
    return {"message": "Tea Leaf Yield Prediction API is running"}


@app.get("/health")
def health():
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
    }


@app.get("/ready")
def readiness():
    """Kubernetes readiness probe endpoint."""
    if model is None:
        logger.warning("Readiness check failed — model not loaded")
        return {"status": "not ready", "model_loaded": False}
    return {"status": "ready", "model_loaded": True}


@app.post("/predict")
def predict(data: PredictionInput):
    if model is None:
        logger.error("Prediction attempted but model is not loaded")
        return {"error": "Model not loaded"}

    start_time = time.time()

    input_data = pd.DataFrame(
        [[getattr(data, feature) for feature in FEATURES]],
        columns=FEATURES,
    )

    logger.info(
        "Prediction request — flush=%s section=%s",
        data.flush,
        data.section,
    )

    try:
        prediction = model.predict(input_data)[0]
        elapsed = time.time() - start_time

        logger.info(
            "Prediction successful — result=%.2f elapsed=%.4fs",
            prediction,
            elapsed,
        )

        return {
            "predicted_yield_kg_per_worker": round(float(prediction), 2)
        }
    except Exception:
        elapsed = time.time() - start_time
        logger.exception("Prediction failed after %.4fs", elapsed)
        return {"error": "Prediction failed"}
