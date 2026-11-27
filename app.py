"""Tea Leaf Yield Prediction API — FastAPI application with production logging."""

import logging
import sys
import time
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from serving import (
    build_feature_dataframe,
    load_serving_config,
    load_serving_model,
)

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
SERVING_CONFIG = load_serving_config()
MODEL_PATH = SERVING_CONFIG["model_path"]
MODEL_VERSION = SERVING_CONFIG["model_version"]
FEATURES = SERVING_CONFIG["features"]
MAX_HOURS_WORKED = SERVING_CONFIG["max_hours_worked"]
MIN_RAIN_MM_WEEK = SERVING_CONFIG["min_rain_mm_week"]

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
logger.info("Loading model v%s from %s", MODEL_VERSION, MODEL_PATH)
try:
    model = load_serving_model(SERVING_CONFIG)
    logger.info("Model v%s loaded successfully", MODEL_VERSION)
except Exception:
    logger.exception("Failed to load model v%s", MODEL_VERSION)
    model = None


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Application lifespan manager for startup and shutdown logging."""
    logger.info("Application starting up (model_version=v%s)", MODEL_VERSION)
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
    rain_mm_week: float = Field(
        ..., ge=MIN_RAIN_MM_WEEK, description="Weekly rainfall in mm (must be >= 0)"
    )
    slope_deg: float
    temp_c: float
    bush_age_yr: float
    hours_worked: float = Field(
        ..., ge=0, le=MAX_HOURS_WORKED, description="Hours worked (must be <= 15)"
    )
    flush: str
    section: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/")
def home():
    return {
        "message": "Tea Leaf Yield Prediction API is running",
        "model_version": MODEL_VERSION,
    }


@app.get("/health")
def health():
    if model is None:
        return JSONResponse(
            status_code=503,
            content={
                "status": "error",
                "model_loaded": False,
                "model_version": MODEL_VERSION,
            },
        )
    return {
        "status": "ok",
        "model_loaded": True,
        "model_version": MODEL_VERSION,
    }


@app.get("/ready")
def readiness():
    """Kubernetes readiness probe endpoint."""
    if model is None:
        logger.warning("Readiness check failed — model not loaded")
        return JSONResponse(
            status_code=503,
            content={
                "status": "not ready",
                "model_loaded": False,
                "model_version": MODEL_VERSION,
            },
        )
    return {
        "status": "ready",
        "model_loaded": True,
        "model_version": MODEL_VERSION,
    }


@app.post("/predict")
def predict(data: PredictionInput):
    if model is None:
        logger.error("Prediction attempted but model is not loaded")
        return JSONResponse(
            status_code=503,
            content={
                "detail": "Model not loaded",
                "model_version": MODEL_VERSION,
            },
        )

    start_time = time.time()

    input_data = build_feature_dataframe(data.model_dump(), FEATURES)

    logger.info(
        "Prediction request — flush=%s section=%s",
        data.flush,
        data.section,
    )

    try:
        prediction = model.predict(input_data)[0]
        elapsed = time.time() - start_time

        logger.info(
            "Prediction successful — model_version=%s result=%.2f elapsed=%.4fs",
            MODEL_VERSION,
            prediction,
            elapsed,
        )

        return {
            "predicted_yield_kg_per_worker": round(float(prediction), 2),
            "model_version": MODEL_VERSION,
        }
    except Exception:
        elapsed = time.time() - start_time
        logger.exception("Prediction failed after %.4fs", elapsed)
        return JSONResponse(
            status_code=500,
            content={
                "detail": "Prediction failed",
                "model_version": MODEL_VERSION,
            },
        )
