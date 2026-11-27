"""Unit tests for drift monitoring (PSI) and group-wise model evaluation."""

import numpy as np
import pandas as pd
import pytest

from src.monitor import compute_psi, compute_psi_report, evaluate_group_performance, interpret_psi


def test_psi_identical_distributions():
    rng = np.random.default_rng(42)
    ref = rng.normal(loc=20.0, scale=5.0, size=500)
    psi, bins_used = compute_psi(ref, ref.copy(), bins=10)
    assert bins_used == 10
    assert psi == pytest.approx(0.0, abs=1e-12)


def test_psi_shifted_distributions():
    rng = np.random.default_rng(7)
    ref = rng.normal(loc=7.0, scale=1.5, size=500)
    cur = rng.normal(loc=11.0, scale=1.5, size=300)
    psi, _ = compute_psi(ref, cur, bins=10)
    assert np.isfinite(psi)
    assert psi > 0.25
    assert interpret_psi(psi) == "Significant drift"


def test_psi_zero_frequency_bins_and_out_of_range():
    ref = np.linspace(0.0, 10.0, 200)
    # Current falls entirely outside reference upper bins, leaving multiple 0-count bins
    cur = np.full(100, 50.0)
    psi, bins_used = compute_psi(ref, cur, bins=10)
    assert bins_used == 10
    assert np.isfinite(psi)
    assert psi > 1.0


def test_psi_constant_columns():
    ref = pd.Series([15.0] * 100)
    cur_same = pd.Series([15.0] * 50)
    cur_shifted = pd.Series([18.0] * 50)

    psi_same, _ = compute_psi(ref, cur_same, bins=10)
    psi_shifted, _ = compute_psi(ref, cur_shifted, bins=10)

    assert psi_same == pytest.approx(0.0, abs=1e-12)
    assert np.isfinite(psi_shifted)
    assert psi_shifted > 0.25


def test_psi_missing_values_and_duplicate_quantiles():
    ref = pd.Series([1.0] * 80 + [2.0] * 10 + [3.0] * 10 + [np.nan] * 5)
    cur = pd.Series([1.0] * 40 + [2.0] * 5 + [3.0] * 5 + [np.nan] * 3)
    psi, bins_used = compute_psi(ref, cur, bins=10)
    assert bins_used >= 2
    assert np.isfinite(psi)
    assert psi == pytest.approx(0.0, abs=1e-6)


def test_compute_psi_report_identifies_most_drifted():
    rng = np.random.default_rng(123)
    ref_df = pd.DataFrame(
        {
            "stable_feat": rng.normal(10, 2, 200),
            "drifted_feat": rng.normal(5, 1, 200),
            "kg_per_worker": rng.normal(30, 4, 200),
        }
    )
    cur_df = pd.DataFrame(
        {
            "stable_feat": rng.normal(10, 2, 150),
            "drifted_feat": rng.normal(12, 1, 150),
            "kg_per_worker": rng.normal(31, 4, 150),
        }
    )
    report = compute_psi_report(
        ref_df,
        cur_df,
        numeric_columns=["stable_feat", "drifted_feat", "kg_per_worker"],
        target_column="kg_per_worker",
    )
    assert len(report) == 3
    most_drifted = report.loc[report["is_most_drifted_feature"], "column"].iloc[0]
    assert most_drifted == "drifted_feat"


class _DummyPredictor:
    """Simple deterministic predictor for group-wise evaluation tests."""

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), 30.0)


def test_evaluate_group_performance_identifies_worst_group():
    df = pd.DataFrame(
        {
            "worker_exp_yr": [5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "section": ["Upper Garden", "Upper Garden", "Lower Garden", "Lower Garden", "River Side", "River Side"],
            "kg_per_worker": [31.0, 29.0, 32.0, 28.0, 45.0, 47.0],
        }
    )
    perf = evaluate_group_performance(
        model=_DummyPredictor(),
        df=df,
        feature_columns=["worker_exp_yr", "section"],
    )
    worst = perf.loc[perf["is_worst_served"], "section"].iloc[0]
    assert worst == "River Side"
    river_mae = perf.loc[perf["section"] == "River Side", "mae"].iloc[0]
    assert river_mae == pytest.approx(16.0, abs=1e-6)
    overall_mae = perf.loc[perf["scope"] == "overall", "mae"].iloc[0]
    assert overall_mae == pytest.approx((1 + 1 + 2 + 2 + 15 + 17) / 6.0, abs=1e-6)
