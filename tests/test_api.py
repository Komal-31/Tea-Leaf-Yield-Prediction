import pytest
from pydantic import ValidationError
from app import app, home, health, predict, PredictionInput
from fastapi.testclient import TestClient

def test_home():
    response = home()
    assert response["message"] == "Tea Leaf Yield Prediction API is running"


def test_health():
    response = health()
    assert response["status"] == "healthy"
    assert response["model_loaded"] is True


def test_prediction():
    data = PredictionInput(
        worker_exp_yr=48,
        rain_mm_week=285,
        slope_deg=21,
        temp_c=15.9,
        bush_age_yr=10,
        hours_worked=5.4,
        flush="monsoon",
        section="River Side",
    )

    response = predict(data)

    assert "predicted_yield_kg_per_worker" in response
    assert response["predicted_yield_kg_per_worker"] == 25.28
    
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
    
client = TestClient(app)


def test_health_http():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


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
            "flush": "monsoon"
        }
    )

    assert response.status_code == 422