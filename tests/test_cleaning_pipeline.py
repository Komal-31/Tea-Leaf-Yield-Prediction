
import json

import pandas as pd
import pytest

from src import clean


@pytest.fixture
def synthetic_data(tmp_path, monkeypatch):

    raw_path = tmp_path / "train.csv"
    output_dir = tmp_path / "processed"

    train_path = output_dir / "train_clean.csv"
    val_path = output_dir / "val_clean.csv"

    rows = [
        ["R1", "2026-05-01", "River Side", 24, 10, 25],
        ["R2", "2026-05-02", "Upper Garden", 24, 20, 30],
        ["R3", "2026-05-03", "River Side", 120, 999, 35],
        ["R4", "2026-06-01", "Lower Garden", 12, 80, 40],
        ["R5", "2026-06-02", "River Side", 36, 999, 250],
        ["R1", "2026-05-01", "River Side", 24, 10, 25],
    ]

    columns = [
        "record_id",
        "date",
        "section",
        "worker_exp_yr",
        "slope_deg",
        "kg_per_worker",
    ]

    pd.DataFrame(rows, columns=columns).to_csv(
        raw_path, index=False
    )

    monkeypatch.setattr(clean, "RAW_PATH", raw_path)
    monkeypatch.setattr(clean, "TRAIN_PATH", train_path)
    monkeypatch.setattr(clean, "VAL_PATH", val_path)
    monkeypatch.setattr(clean, "OUTPUT_DIR", output_dir)

    clean.clean_data()

    train = pd.read_csv(train_path)
    val = pd.read_csv(val_path)

    with open(
        output_dir / "cleaning_metadata.json",
        encoding="utf-8"
    ) as f:
        metadata = json.load(f)

    return train, val, metadata


def test_duplicate_removal(synthetic_data):
    train, val, metadata = synthetic_data

    combined = pd.concat([train, val])

    assert len(combined) == 5
    assert not combined["record_id"].duplicated().any()
    assert metadata["duplicates_removed"] == 1


def test_experience_unit_conversion(synthetic_data):
    train, val, _ = synthetic_data

    combined = pd.concat([train, val])
    experience = combined.set_index("record_id")["worker_exp_yr"]

    assert experience["R1"] == 2
    assert experience["R3"] == 10
    assert experience["R5"] == 3


def test_other_sections_unchanged(synthetic_data):
    train, val, _ = synthetic_data

    combined = pd.concat([train, val])
    experience = combined.set_index("record_id")["worker_exp_yr"]

    assert experience["R2"] == 24
    assert experience["R4"] == 12


def test_slope_sentinel_imputation(synthetic_data):
    train, val, metadata = synthetic_data

    combined = pd.concat([train, val])
    slopes = combined.set_index("record_id")["slope_deg"]

    assert slopes["R3"] == 15
    assert slopes["R5"] == 15
    assert metadata["invalid_slopes"] == 2
    assert combined["slope_deg"].between(0, 90).all()


def test_training_only_median(synthetic_data):
    _, _, metadata = synthetic_data

    assert metadata["slope_median_from_train"] == 15


def test_chronological_split(synthetic_data):
    train, val, _ = synthetic_data

    assert len(train) == 3
    assert len(val) == 2

    assert pd.to_datetime(train["date"]).max() < (
        pd.to_datetime(val["date"]).min()
    )

    assert set(train["record_id"]).isdisjoint(
        set(val["record_id"])
    )


def test_suspicious_target_flagging(synthetic_data):
    train, val, metadata = synthetic_data

    combined = pd.concat([train, val])
    flagged = combined[combined["target_outlier_flag"] == 1]

    assert len(flagged) == 1
    assert flagged.iloc[0]["record_id"] == "R5"
    assert flagged.iloc[0]["kg_per_worker"] == 250
    assert metadata["suspicious_targets"] == 1


def test_cleaning_metadata(synthetic_data):
    _, _, metadata = synthetic_data

    assert metadata["original_rows"] == 6
    assert metadata["unique_rows"] == 5
    assert metadata["experience_rows_affected"] == 3
    assert metadata["experience_values_corrected"] == 3
    assert metadata["experience_conversion_factor"] == 12
    assert metadata["train_rows"] == 3
    assert metadata["validation_rows"] == 2
