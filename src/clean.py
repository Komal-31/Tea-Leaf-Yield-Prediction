
from pathlib import Path
import pandas as pd
import numpy as np
import json

# Project paths
ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT / "data" / "raw" / "train.csv"
OUTPUT_DIR = ROOT / "data" / "processed"

CUTOFF_DATE = pd.Timestamp("2026-06-01")


def clean_data():
    # 1. Load original data
    df = pd.read_csv(RAW_PATH)
    original_rows = len(df)

    # 2. Remove duplicate records
    df = df.drop_duplicates(subset="record_id").copy()

    # 3. Parse dates
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    if df["date"].isna().any():
        raise ValueError("Invalid dates found in dataset")

    # 4. Correct suspected months-versus-years errors
    exp_mask = (
        (df["worker_exp_yr"] > 50)
        & (df["worker_exp_yr"] % 12 == 0)
    )

    df.loc[exp_mask, "worker_exp_yr"] = (
        df.loc[exp_mask, "worker_exp_yr"] / 12
    )

    # 5. Mark invalid slope readings as missing
    invalid_slope = (
        (df["slope_deg"] < 0)
        | (df["slope_deg"] > 90)
    )

    df.loc[invalid_slope, "slope_deg"] = np.nan

    # 6. Flag suspicious target values
    df["target_outlier_flag"] = (
        df["kg_per_worker"] > 100
    ).astype(int)

    # 7. Chronological train-validation split
    train = df[df["date"] < CUTOFF_DATE].copy()
    val = df[df["date"] >= CUTOFF_DATE].copy()

    if train.empty or val.empty:
        raise ValueError("Train or validation dataset is empty")

    # 8. Calculate median using training data only
    slope_median = train["slope_deg"].median()

    # 9. Apply training median to both datasets
    train["slope_deg"] = train["slope_deg"].fillna(
        slope_median
    )

    val["slope_deg"] = val["slope_deg"].fillna(
        slope_median
    )

    # 10. Save cleaned datasets
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train.to_csv(
        OUTPUT_DIR / "train_clean.csv",
        index=False
    )

    val.to_csv(
        OUTPUT_DIR / "val_clean.csv",
        index=False
    )

    # 11. Save cleaning metadata
    metadata = {
        "original_rows": original_rows,
        "unique_rows": len(df),
        "duplicates_removed": original_rows - len(df),
        "experience_values_corrected": int(exp_mask.sum()),
        "invalid_slopes": int(invalid_slope.sum()),
        "suspicious_targets": int(
            df["target_outlier_flag"].sum()
        ),
        "train_rows": len(train),
        "validation_rows": len(val),
        "slope_median_from_train": float(slope_median),
        "cutoff_date": str(CUTOFF_DATE.date())
    }

    with open(
        OUTPUT_DIR / "cleaning_metadata.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(metadata, f, indent=4)

    print("\n=== CLEANING COMPLETED ===")
    for key, value in metadata.items():
        print(f"{key}: {value}")

    print("\nSaved files:")
    print(OUTPUT_DIR / "train_clean.csv")
    print(OUTPUT_DIR / "val_clean.csv")
    print(OUTPUT_DIR / "cleaning_metadata.json")


if __name__ == "__main__":
    clean_data()
