
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
