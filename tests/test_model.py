
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent

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
def model():
    mlflow.set_tracking_uri(
        f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
    )

    return mlflow.sklearn.load_model(
        "models:/TeaLeafYieldPredictor/1"
    )


@pytest.fixture(scope="module")
def sample_data():
    data = pd.read_csv(
        ROOT / "data/processed/val_clean.csv"
    )
    return data[FEATURES].head(5)


def test_model_loads(model):
    assert model is not None
    assert hasattr(model, "predict")


def test_expected_features(sample_data):
    assert list(sample_data.columns) == FEATURES


def test_prediction_count(model, sample_data):
    predictions = model.predict(sample_data)
    assert len(predictions) == len(sample_data)


def test_predictions_are_finite(model, sample_data):
    predictions = model.predict(sample_data)
    assert np.isfinite(predictions).all()


def test_predictions_are_non_negative(model, sample_data):
    predictions = model.predict(sample_data)
    assert (predictions >= 0).all()
