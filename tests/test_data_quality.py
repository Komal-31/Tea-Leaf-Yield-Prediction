
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"


def load_datasets():
    train = pd.read_csv(DATA_DIR / "train_clean.csv")
    val = pd.read_csv(DATA_DIR / "val_clean.csv")
    return train, val


def test_processed_files_exist():
    assert (DATA_DIR / "train_clean.csv").exists()
    assert (DATA_DIR / "val_clean.csv").exists()


def test_required_columns():
    train, val = load_datasets()

    required = [
        "worker_exp_yr",
        "rain_mm_week",
        "slope_deg",
        "temp_c",
        "bush_age_yr",
        "hours_worked",
        "flush",
        "section",
        "kg_per_worker",
    ]

    for column in required:
        assert column in train.columns
        assert column in val.columns


def test_no_duplicate_record_ids():
    train, val = load_datasets()

    assert not train["record_id"].duplicated().any()
    assert not val["record_id"].duplicated().any()


def test_no_invalid_slope_sentinel():
    train, val = load_datasets()

    assert not (train["slope_deg"] == 999).any()
    assert not (val["slope_deg"] == 999).any()


def test_no_missing_target():
    train, val = load_datasets()

    assert train["kg_per_worker"].notna().all()
    assert val["kg_per_worker"].notna().all()


def test_no_train_validation_overlap():
    train, val = load_datasets()

    train_ids = set(train["record_id"])
    val_ids = set(val["record_id"])

    assert train_ids.isdisjoint(val_ids)

def test_experience_valid_range():
    train, val = load_datasets()

    for df in [train, val]:
        assert df["worker_exp_yr"].between(0, 30).all()


def test_river_side_experience_conversion():
    train, val = load_datasets()

    cleaned = pd.concat([train, val])

    raw = pd.read_csv(ROOT / "data" / "raw" / "train.csv")
    raw = raw.drop_duplicates(subset="record_id")

    river = raw[raw["section"] == "River Side"]

    expected = river.set_index("record_id")["worker_exp_yr"] / 12

    actual = cleaned.set_index("record_id").loc[
        expected.index, "worker_exp_yr"
    ]

    pd.testing.assert_series_equal(
        actual,
        expected,
        check_dtype=False
    )


def test_slope_valid_range():
    train, val = load_datasets()

    for df in [train, val]:
        assert df["slope_deg"].notna().all()
        assert df["slope_deg"].between(0, 90).all()


def test_chronological_split():
    train, val = load_datasets()

    train_dates = pd.to_datetime(train["date"])
    val_dates = pd.to_datetime(val["date"])

    assert train_dates.max() < val_dates.min()


def test_suspicious_target_flags():
    train, val = load_datasets()

    for df in [train, val]:
        expected = (df["kg_per_worker"] > 100).astype(int)

        pd.testing.assert_series_equal(
            df["target_outlier_flag"],
            expected,
            check_names=False,
            check_dtype=False
        )
