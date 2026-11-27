from pathlib import Path
import pandas as pd
import numpy as np
import json
import yaml

# Project root
ROOT = Path(__file__).resolve().parent.parent

# Load centralized configuration
with open(ROOT / "params.yaml", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

PATHS = CONFIG["paths"]
DATA = CONFIG["data"]
CLEANING = CONFIG["cleaning"]

RAW_PATH = ROOT / PATHS["raw_train"]
TRAIN_PATH = ROOT / PATHS["processed_train"]
VAL_PATH = ROOT / PATHS["processed_val"]
OUTPUT_DIR = TRAIN_PATH.parent

CUTOFF_DATE = pd.Timestamp(CLEANING["validation_cutoff"])


def clean_data():

    # 1. Load original data
    df = pd.read_csv(RAW_PATH)
    original_rows = len(df)

    id_col = DATA["id_column"]
    date_col = DATA["date_column"]
    group_col = DATA["group_column"]
    target_col = DATA["target_column"]

    exp_col = CLEANING["experience_column"]
    slope_col = CLEANING["slope_column"]

    # 2. Remove duplicate records
    df = df.drop_duplicates(subset=id_col).copy()

    # 3. Parse dates
    df[date_col] = pd.to_datetime(
        df[date_col], errors="coerce"
    )

    if df[date_col].isna().any():
        raise ValueError("Invalid dates found in dataset")

    # 4. Correct experience recorded in months
    exp_mask = (
        df[group_col] == CLEANING["experience_unit_group"]
    )

    changed_mask = exp_mask & (df[exp_col] != 0)

    df[exp_col] = df[exp_col].astype(float)

    df.loc[exp_mask, exp_col] = (
        df.loc[exp_mask, exp_col]
        / CLEANING["experience_conversion_factor"]
    )

    # 5. Mark invalid slope readings as missing
    invalid_slope = (
        (df[slope_col] < CLEANING["slope_min"])
        | (df[slope_col] > CLEANING["slope_max"])
    )

    df.loc[invalid_slope, slope_col] = np.nan

    # 6. Flag suspicious target values
    df["target_outlier_flag"] = (
        df[target_col] > DATA["suspicious_target_threshold"]
    ).astype(int)

    # 7. Chronological train-validation split
    train = df[df[date_col] < CUTOFF_DATE].copy()
    val = df[df[date_col] >= CUTOFF_DATE].copy()

    if train.empty or val.empty:
        raise ValueError("Train or validation dataset is empty")

    # 8. Calculate median using training data only
    slope_median = train[slope_col].median()

    if pd.isna(slope_median):
        raise ValueError("Cannot calculate training slope median")

    # 9. Apply training median to both datasets
    train[slope_col] = train[slope_col].fillna(slope_median)
    val[slope_col] = val[slope_col].fillna(slope_median)

    # 10. Save cleaned datasets
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    VAL_PATH.parent.mkdir(parents=True, exist_ok=True)

    train.to_csv(TRAIN_PATH, index=False)
    val.to_csv(VAL_PATH, index=False)

    # 11. Save cleaning metadata
    metadata = {
        "original_rows": original_rows,
        "unique_rows": len(df),
        "duplicates_removed": original_rows - len(df),
        "experience_group": CLEANING["experience_unit_group"],
        "experience_rows_affected": int(exp_mask.sum()),
        "experience_values_corrected": int(changed_mask.sum()),
        "experience_conversion_factor": CLEANING[
            "experience_conversion_factor"
        ],
        "invalid_slopes": int(invalid_slope.sum()),
        "suspicious_targets": int(
            df["target_outlier_flag"].sum()
        ),
        "train_rows": len(train),
        "validation_rows": len(val),
        "slope_median_from_train": float(slope_median),
        "cutoff_date": str(CUTOFF_DATE.date())
    }

    metadata_path = OUTPUT_DIR / "cleaning_metadata.json"

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print("\n=== CLEANING COMPLETED ===")

    for key, value in metadata.items():
        print(f"{key}: {value}")

    print("\nSaved files:")
    print(TRAIN_PATH)
    print(VAL_PATH)
    print(metadata_path)


if __name__ == "__main__":
    clean_data()