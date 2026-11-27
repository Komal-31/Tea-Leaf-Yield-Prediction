"""Unit tests for chronological splitting, retraining pipeline, and promotion rule."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor

from src.retrain import build_pipeline, chronological_split, decide_promotion, evaluate_regression


def test_chronological_split_ratio_ordering_and_disjointness():
    # Intentionally shuffled dates
    dates = pd.date_range("2026-07-01", periods=100, freq="D").astype(str).tolist()
    rng = np.random.default_rng(42)
    perm = rng.permutation(100)
    df = pd.DataFrame(
        {
            "record_id": [f"R{i:04d}" for i in perm],
            "date": [dates[i] for i in perm],
            "value": perm,
        }
    )

    train_part, eval_part = chronological_split(df, train_ratio=0.7)

    assert len(train_part) == 70
    assert len(eval_part) == 30
    assert set(train_part["record_id"]).isdisjoint(set(eval_part["record_id"]))
    assert pd.to_datetime(train_part["date"]).is_monotonic_increasing
    assert pd.to_datetime(eval_part["date"]).is_monotonic_increasing
    assert pd.to_datetime(train_part["date"]).max() <= pd.to_datetime(eval_part["date"]).min()


def test_promotion_below_five_percent():
    result = decide_promotion(v1_mae=10.0, v2_mae=9.51, improvement_threshold=0.05)
    assert result["promoted"] is False
    assert result["decision"] == "keep"
    assert result["champion_version"] == 1
    assert result["improvement_percent"] == pytest.approx(4.9, abs=1e-6)


def test_promotion_exactly_five_percent():
    result = decide_promotion(v1_mae=10.0, v2_mae=9.5, improvement_threshold=0.05)
    assert result["promoted"] is True
    assert result["decision"] == "promote"
    assert result["champion_version"] == 2
    assert result["improvement_percent"] == pytest.approx(5.0, abs=1e-6)

    # Also test floating-point boundary like 3.0 -> 2.85
    result_fp = decide_promotion(v1_mae=3.0, v2_mae=2.85, improvement_threshold=0.05)
    assert result_fp["promoted"] is True
    assert result_fp["champion_version"] == 2


def test_promotion_above_five_percent():
    result = decide_promotion(v1_mae=8.0, v2_mae=6.0, improvement_threshold=0.05)
    assert result["promoted"] is True
    assert result["decision"] == "promote"
    assert result["champion_version"] == 2
    assert result["improvement_percent"] == pytest.approx(25.0, abs=1e-6)


def test_promotion_zero_baseline_mae():
    result = decide_promotion(v1_mae=0.0, v2_mae=0.0, improvement_threshold=0.05)
    assert result["promoted"] is False
    assert result["decision"] == "keep"
    assert result["champion_version"] == 1


def test_build_pipeline_and_evaluate_regression():
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
