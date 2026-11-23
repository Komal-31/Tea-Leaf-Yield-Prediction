import json
from pathlib import Path

import mlflow.sklearn
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "tea_leaf_v1"
EXAMPLE_PATH = MODEL_PATH / "serving_input_example.json"

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


@pytest.fixture(scope="module")
def model_and_sample():
    assert MODEL_PATH.exists(), "Saved model folder is missing"
    assert EXAMPLE_PATH.exists(), "Serving example is missing"

    model = mlflow.sklearn.load_model(str(MODEL_PATH))

    with open(EXAMPLE_PATH, encoding="utf-8") as file:
        payload = json.load(file)

    sample = pd.DataFrame(**payload["dataframe_split"])
    return model, sample


def test_saved_model_loads(model_and_sample):
    model, _ = model_and_sample
    assert model is not None
    assert hasattr(model, "predict")


def test_serving_example_has_expected_features(model_and_sample):
    _, sample = model_and_sample
    assert list(sample.columns) == FEATURES


def test_model_prediction_count(model_and_sample):
    model, sample = model_and_sample
    predictions = model.predict(sample)
    assert len(predictions) == len(sample)


def test_predictions_are_finite(model_and_sample):
    model, sample = model_and_sample
    predictions = model.predict(sample)
    assert np.isfinite(predictions).all()


def test_predictions_are_non_negative(model_and_sample):
    model, sample = model_and_sample
    predictions = model.predict(sample)
    assert (predictions >= 0).all()
