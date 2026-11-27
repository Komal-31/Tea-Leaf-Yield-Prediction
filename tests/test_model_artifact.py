import json
from pathlib import Path

import mlflow.sklearn
import numpy as np
import pandas as pd
import pytest
from mlflow import MlflowClient

from serving import (
    load_serving_config,
    load_serving_model,
    verify_champion_alignment,
)

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "tea_leaf_v4"
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
    assert MODEL_PATH.exists(), "Saved Version 4 champion model folder is missing"
    assert EXAMPLE_PATH.exists(), "Serving example is missing"

    model = load_serving_model()

    with open(EXAMPLE_PATH, encoding="utf-8") as file:
        payload = json.load(file)

    sample = pd.DataFrame(**payload["dataframe_split"])
    return model, sample


def test_historical_model_artifacts_preserved():
    for version in ("v1", "v2", "v3", "v4"):
        artifact_dir = ROOT / "models" / f"tea_leaf_{version}"
        assert artifact_dir.exists(), f"Missing historical artifact: {artifact_dir}"
        assert (artifact_dir / "MLmodel").exists()


def test_serving_config_points_to_champion_v4():
    cfg = load_serving_config()
    assert cfg["model_version"] == 4
    assert cfg["model_path"] == MODEL_PATH.resolve()
    assert cfg["features"] == FEATURES
    assert verify_champion_alignment(cfg) is True

    client = MlflowClient(tracking_uri=f"sqlite:///{cfg['mlflow_db'].as_posix()}")
    champ_mv = client.get_model_version_by_alias(
        cfg["registered_model_name"], cfg["champion_alias"]
    )
    assert int(champ_mv.version) == 4
    assert Path(champ_mv.source).as_posix().rstrip("/") == "models/tea_leaf_v4"


def test_serving_loads_without_mlflow_db_and_never_falls_back(tmp_path):
    cfg = load_serving_config()
    cfg["mlflow_db"] = tmp_path / "missing_mlflow.db"
    assert verify_champion_alignment(cfg) is False
    model = load_serving_model(cfg)
    assert hasattr(model, "predict")

    missing_cfg = {**cfg, "model_path": tmp_path / "missing_v4"}
    with pytest.raises(FileNotFoundError):
        load_serving_model(missing_cfg)


def test_champion_mismatch_raises_error():
    cfg = load_serving_config()
    bad_cfg = {**cfg, "model_version": 1, "model_dir": "models/tea_leaf_v1"}
    with pytest.raises(RuntimeError):
        verify_champion_alignment(bad_cfg)


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


def test_v4_matches_registered_champion_prediction(model_and_sample):
    model, sample = model_and_sample
    cfg = load_serving_config()
    mlflow.set_tracking_uri(f"sqlite:///{cfg['mlflow_db'].as_posix()}")
    champion = mlflow.sklearn.load_model(
        f"models:/{cfg['registered_model_name']}@{cfg['champion_alias']}"
    )
    assert np.allclose(model.predict(sample), champion.predict(sample))
