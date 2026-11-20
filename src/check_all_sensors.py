
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
df = pd.read_csv(ROOT / "data" / "raw" / "train.csv")

df = df.drop_duplicates().copy()

columns = [
    "worker_exp_yr",
    "rain_mm_week",
    "slope_deg",
    "temp_c",
    "bush_age_yr",
    "hours_worked"
]

print("\n=== ALL NUMERICAL FEATURES ===")

for col in columns:
    counts = df[col].value_counts(dropna=False)

    most_common = counts.index[0]
    frequency = counts.iloc[0]
    percentage = frequency / len(df) * 100

    print(f"\nColumn: {col}")
    print(f"Most common value: {most_common}")
    print(f"Frequency: {frequency}")
    print(f"Percentage: {percentage:.2f}%")
    print(f"Unique values: {df[col].nunique()}")

    if percentage > 20:
        print("WARNING: Unusually frequent value — investigate")
    else:
        print("No extreme concentration detected")

print("\n=== CHECK COMPLETE ===")
