from app import home, health, predict, PredictionInput


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