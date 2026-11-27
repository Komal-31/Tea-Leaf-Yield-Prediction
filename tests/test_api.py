from pathlib import Path

import mlflow.sklearn
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

import app as app_module
from app import PredictionInput, app, health, home, predict

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent

VALID_EXAMPLE_PAYLOAD = {
    "worker_exp_yr": 12,
    "rain_mm_week": 150,
    "slope_deg": 17,
    "temp_c": 25,
    "bush_age_yr": 10,
    "hours_worked": 6,
    "flush": "monsoon",
    "section": "River Side",
}

client = TestClient(app)


def test_home():
    response = home()
    assert response["message"] == "Tea Leaf Yield Prediction API is running"
    assert response["model_version"] == 4


def test_health():
    response = health()
    assert response["status"] == "ok"
    assert response["model_loaded"] is True
    assert response["model_version"] == 4


def test_prediction():
    data = PredictionInput(**VALID_EXAMPLE_PAYLOAD)
    response = predict(data)

    v4_model = mlflow.sklearn.load_model(str(PROJECT_ROOT / "models" / "tea_leaf_v4"))
    expected = round(float(v4_model.predict(pd.DataFrame([VALID_EXAMPLE_PAYLOAD]))[0]), 2)

    assert "predicted_yield_kg_per_worker" in response
    assert response["predicted_yield_kg_per_worker"] == expected
    assert response["model_version"] == 4


def test_prediction_missing_field():
    with pytest.raises(ValidationError):
        PredictionInput(
            worker_exp_yr=48,
            rain_mm_week=285,
            slope_deg=21,
            temp_c=15.9,
            bush_age_yr=10,
            hours_worked=5.4,
            flush="monsoon",
        )


def test_prediction_invalid_numeric_type():
    with pytest.raises(ValidationError):
        PredictionInput(
            worker_exp_yr="invalid",
            rain_mm_week=285,
            slope_deg=21,
            temp_c=15.9,
            bush_age_yr=10,
            hours_worked=5.4,
            flush="monsoon",
            section="River Side",
        )


def test_prediction_negative_experience():
    data = PredictionInput(
        worker_exp_yr=-5,
        rain_mm_week=285,
        slope_deg=21,
        temp_c=15.9,
        bush_age_yr=10,
        hours_worked=5.4,
        flush="monsoon",
        section="River Side",
    )

    response = predict(data)

    assert response["predicted_yield_kg_per_worker"] >= 0
    assert response["model_version"] == 4


def test_health_http():
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["model_loaded"] is True
    assert payload["model_version"] == 4


def test_health_unloaded_model_not_healthy(monkeypatch):
    monkeypatch.setattr(app_module, "model", None)

    health_resp = client.get("/health")
    assert health_resp.status_code == 503
    assert health_resp.json()["status"] != "ok"
    assert health_resp.json()["model_loaded"] is False

    predict_resp = client.post("/predict", json=VALID_EXAMPLE_PAYLOAD)
    assert predict_resp.status_code == 503


def test_prediction_http_matches_direct_v4():
    response = client.post("/predict", json=VALID_EXAMPLE_PAYLOAD)
    assert response.status_code == 200

    body = response.json()
    v4_model = mlflow.sklearn.load_model(str(PROJECT_ROOT / "models" / "tea_leaf_v4"))
    expected = round(float(v4_model.predict(pd.DataFrame([VALID_EXAMPLE_PAYLOAD]))[0]), 2)

    assert body["predicted_yield_kg_per_worker"] == expected
    assert body["model_version"] == 4


def test_prediction_boundary_values_http():
    boundary_payload = {
        **VALID_EXAMPLE_PAYLOAD,
        "hours_worked": 15,
        "rain_mm_week": 0,
    }
    response = client.post("/predict", json=boundary_payload)

    assert response.status_code == 200
    body = response.json()
    assert isinstance(body["predicted_yield_kg_per_worker"], float)
    assert body["model_version"] == 4


@pytest.mark.parametrize("invalid_hours", [15.01, 16, 24])
def test_prediction_hours_worked_above_15_rejected(invalid_hours):
    payload = {**VALID_EXAMPLE_PAYLOAD, "hours_worked": invalid_hours}
    response = client.post("/predict", json=payload)

    assert response.status_code == 422


@pytest.mark.parametrize("invalid_rain", [-0.1, -1, -50])
def test_prediction_negative_rainfall_rejected(invalid_rain):
    payload = {**VALID_EXAMPLE_PAYLOAD, "rain_mm_week": invalid_rain}
    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_prediction_negative_hours_worked_rejected():
    payload = {**VALID_EXAMPLE_PAYLOAD, "hours_worked": -1}
    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_prediction_missing_field_http():
    response = client.post(
        "/predict",
        json={
            "worker_exp_yr": 48,
            "rain_mm_week": 285,
            "slope_deg": 21,
            "temp_c": 15.9,
            "bush_age_yr": 10,
            "hours_worked": 5.4,
            "flush": "monsoon",
        },
    )

    assert response.status_code == 422
