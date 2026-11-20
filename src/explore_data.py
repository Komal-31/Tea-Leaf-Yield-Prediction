
from pathlib import Path
import pandas as pd

# Locate the project root relative to this script
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Original dataset: read-only
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "train.csv"


def explore_data():
    print("\n========== MP148 DATA EXPLORATION ==========")

    df = pd.read_csv(DATA_PATH)

    print("\n1. Dataset shape:")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    print("\n2. Column names:")
    print(df.columns.tolist())

    print("\n3. First five records:")
    print(df.head().to_string())

    print("\n4. Column data types:")
    print(df.dtypes.to_string())

    print("\n5. Missing values:")
    print(df.isnull().sum().to_string())

    print("\n6. Duplicate rows:")
    print(df.duplicated().sum())

    print("\n7. Numerical statistics:")
    print(df.describe(include="number").to_string())

    print("\n8. Categorical values:")
    for column in df.select_dtypes(
        include=["object", "string", "category"]
    ).columns:
        print(f"\n{column}:")
        print(df[column].value_counts(dropna=False).head(20))

    print("\n========== EXPLORATION COMPLETE ==========")


if __name__ == "__main__":
    explore_data()
