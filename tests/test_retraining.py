"""Unit tests for corrected baseline selection, chronological split, retraining, and promotion."""

import mlflow
from mlflow import MlflowClient
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor

from src.retrain import (
    build_pipeline,
    check_existing_retrained_provenance,
    chronological_split,
    decide_promotion,
    evaluate_regression,
    guard_historical_artifacts,
    select_baseline_model_version,
)


def test_chronological_split_ratio_ordering_tiebreak_and_no_leakage():
    dates = ["2026-07-01"] * 5 + ["2026-07-02"] * 5 + ["2026-07-03"] * 5 + ["2026-07-04"] * 5
    rng = np.random.default_rng(42)
    perm = rng.permutation(20)
    df = pd.DataFrame(
        {
            "record_id": [f"R{i:04d}" for i in perm],
            "date": [dates[i] for i in perm],
            "value": perm,
        }
    )

    train_part, eval_part = chronological_split(df, train_ratio=0.7)

    assert len(train_part) == 14
    assert len(eval_part) == 6
    assert set(train_part["record_id"]).isdisjoint(set(eval_part["record_id"]))
    assert pd.to_datetime(train_part["date"]).is_monotonic_increasing
    assert pd.to_datetime(eval_part["date"]).is_monotonic_increasing
    assert pd.to_datetime(train_part["date"]).max() <= pd.to_datetime(eval_part["date"]).min()


def test_promotion_below_five_percent():
    result = decide_promotion(
        baseline_mae=10.0,
        candidate_mae=9.51,
        improvement_threshold=0.05,
        baseline_version=3,
        candidate_version=4,
    )
    assert result["promoted"] is False
    assert result["decision"] == "keep"
    assert result["champion_version"] == 3
    assert result["improvement_percent"] == pytest.approx(4.9, abs=1e-6)


def test_promotion_exactly_five_percent():
    result = decide_promotion(
        baseline_mae=10.0,
        candidate_mae=9.5,
        improvement_threshold=0.05,
        baseline_version=3,
        candidate_version=4,
    )
    assert result["promoted"] is True
    assert result["decision"] == "promote"
    assert result["champion_version"] == 4
    assert result["improvement_percent"] == pytest.approx(5.0, abs=1e-6)

    result_fp = decide_promotion(
        baseline_mae=3.0,
        candidate_mae=2.85,
        improvement_threshold=0.05,
        baseline_version=3,
        candidate_version=4,
    )
    assert result_fp["promoted"] is True
    assert result_fp["champion_version"] == 4


def test_promotion_above_five_percent():
    result = decide_promotion(
        baseline_mae=8.0,
        candidate_mae=6.0,
        improvement_threshold=0.05,
        baseline_version=3,
        candidate_version=4,
    )
    assert result["promoted"] is True
    assert result["decision"] == "promote"
    assert result["champion_version"] == 4
    assert result["improvement_percent"] == pytest.approx(25.0, abs=1e-6)


def test_promotion_zero_baseline_mae():
    result = decide_promotion(
        baseline_mae=0.0,
        candidate_mae=0.0,
        improvement_threshold=0.05,
        baseline_version=3,
        candidate_version=4,
    )
    assert result["promoted"] is False
    assert result["decision"] == "keep"
    assert result["champion_version"] == 3


def test_guard_historical_artifacts_prevents_overwrite(tmp_path):
    v1 = tmp_path / "models" / "tea_leaf_v1"
    v2 = tmp_path / "models" / "tea_leaf_v2"
    v3 = tmp_path / "models" / "tea_leaf_v3"
    v4 = tmp_path / "models" / "tea_leaf_v4"

    with pytest.raises(ValueError):
        guard_historical_artifacts(v2, [v1, v2, v3])

    guard_historical_artifacts(v4, [v1, v2, v3])


def test_explicit_v3_baseline_selection_and_champion_alias(tmp_path):
    db_path = tmp_path / "mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{db_path.as_posix()}")
    client = MlflowClient()
    model_name = "SyntheticTeaPredictor"
    client.create_registered_model(model_name)

    for v in (1, 2, 3):
        art_dir = tmp_path / f"models/tea_leaf_v{v}"
        art_dir.mkdir(parents=True)
        client.create_model_version(
            name=model_name,
            source=f"models/tea_leaf_v{v}",
            run_id=f"run_{v}",
        )

    selected = select_baseline_model_version(
        client=client,
        model_name=model_name,
        baseline_version=3,
        expected_source="models/tea_leaf_v3",
        expected_run_id="run_3",
    )
    assert selected["version"] == 3
    assert selected["run_id"] == "run_3"

    # Non-existent v4 passes provenance check; existing v4 fails
    v4_dir = tmp_path / "models/tea_leaf_v4"
    check_existing_retrained_provenance(client, model_name, v4_dir, "models/tea_leaf_v4")

    client.create_model_version(name=model_name, source="models/tea_leaf_v4", run_id="run_4")
    client.set_registered_model_alias(model_name, "champion", "4")
    champ = client.get_model_version_by_alias(model_name, "champion")
    assert int(champ.version) == 4

    with pytest.raises(RuntimeError):
        check_existing_retrained_provenance(client, model_name, v4_dir, "models/tea_leaf_v4")


def test_build_pipeline_and_evaluate_regression_no_leakage():
    rng = np.random.default_rng(99)
    n = 60
    X = pd.DataFrame(
        {
            "worker_exp_yr": rng.integers(1, 30, size=n).astype(float),
            "rain_mm_week": rng.integers(50, 300, size=n),
            "slope_deg": rng.integers(0, 35, size=n).astype(float),
            "temp_c": rng.uniform(15, 32, size=n),
            "bush_age_yr": rng.integers(5, 60, size=n),
            "hours_worked": rng.uniform(4, 12, size=n),
            "flush": rng.choice(["first", "second", "monsoon"], size=n),
            "section": rng.choice(["Upper Garden", "Lower Garden", "River Side"], size=n),
        }
    )
    y = 3.0 * X["hours_worked"] + 0.1 * X["worker_exp_yr"] + rng.normal(0, 0.5, size=n)

    pipe = build_pipeline(HistGradientBoostingRegressor(max_iter=50, random_state=42), X)
    pipe.fit(X.iloc[:45], y.iloc[:45])
    preds = pipe.predict(X.iloc[45:])
    metrics = evaluate_regression(y.iloc[45:], preds)

    assert metrics["mae"] >= 0.0
    assert metrics["rmse"] >= metrics["mae"]
    assert np.isfinite(preds).all()
