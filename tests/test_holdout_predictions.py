"""Unit tests for holdout prediction generation, validation, and committed CSV schema."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.predict_holdout import generate_holdout_predictions, validate_predictions_df


ROOT = Path(__file__).resolve().parent.parent


class _LinearMockModel:
    """Deterministic model for testing holdout prediction formatting."""

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (X["hours_worked"] * 4.0 + 5.0).to_numpy(dtype=float)


def test_generate_holdout_predictions_schema_count_and_order(tmp_path):
    holdout_df = pd.DataFrame(
        {
            "record_id": ["R50003", "R50001", "R50002"],
            "date": ["2026-09-02", "2026-09-01", "2026-09-01"],
            "worker_exp_yr": [10.0, 5.0, 20.0],
            "rain_mm_week": [120, 200, 150],
            "slope_deg": [15.0, 20.0, 10.0],
            "temp_c": [22.0, 24.0, 21.0],
            "bush_age_yr": [25, 30, 15],
            "hours_worked": [9.5, 10.0, 8.5],
            "flush": ["monsoon", "second", "first"],
            "section": ["River Side", "Upper Garden", "Lower Garden"],
        }
    )
    features = [
        "worker_exp_yr",
        "rain_mm_week",
        "slope_deg",
        "temp_c",
        "bush_age_yr",
        "hours_worked",
        "flush",
        "section",
    ]
    out_csv = tmp_path / "holdout_predictions.csv"

    result = generate_holdout_predictions(
        model=_LinearMockModel(),
        holdout_df=holdout_df,
        feature_columns=features,
        output_path=out_csv,
    )

    assert out_csv.exists()
    loaded = pd.read_csv(out_csv)
    assert list(loaded.columns) == ["record_id", "prediction"]
    assert len(loaded) == len(holdout_df)
    assert list(loaded["record_id"]) == ["R50003", "R50001", "R50002"]
    assert np.isfinite(loaded["prediction"]).all()
    assert loaded["prediction"].notna().all()
    assert len(result) == 3


def test_validate_predictions_rejects_bad_inputs():
    holdout_df = pd.DataFrame({"record_id": ["R1", "R2"]})

    with pytest.raises(ValueError):
        validate_predictions_df(pd.DataFrame({"id": ["R1", "R2"], "prediction": [1.0, 2.0]}), holdout_df)

    with pytest.raises(ValueError):
        validate_predictions_df(pd.DataFrame({"record_id": ["R1"], "prediction": [1.0]}), holdout_df)

    with pytest.raises(ValueError):
        validate_predictions_df(
            pd.DataFrame({"record_id": ["R2", "R1"], "prediction": [1.0, 2.0]}), holdout_df
        )

    with pytest.raises(ValueError):
        validate_predictions_df(
            pd.DataFrame({"record_id": ["R1", "R2"], "prediction": [1.0, np.inf]}), holdout_df
        )

    with pytest.raises(ValueError):
        validate_predictions_df(
            pd.DataFrame({"record_id": ["R1", "R2"], "prediction": [1.0, np.nan]}), holdout_df
        )


def test_committed_holdout_predictions_structure():
    pred_path = ROOT / "predictions" / "holdout_predictions.csv"
    assert pred_path.exists()
    df = pd.read_csv(pred_path)
    assert list(df.columns) == ["record_id", "prediction"]
    assert len(df) == 250
    assert not df["record_id"].duplicated().any()
    assert df["prediction"].notna().all()
    assert np.isfinite(df["prediction"]).all()
    assert (df["prediction"] > 0).all()
