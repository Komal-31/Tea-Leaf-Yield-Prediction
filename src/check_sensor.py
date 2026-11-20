
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

df = pd.read_csv(ROOT / "data" / "raw" / "train.csv")

# Remove exact duplicates for this investigation only
df = df.drop_duplicates().copy()

# Sort by date and record ID
df = df.sort_values(["date", "record_id"])

print("\n=== STUCK SENSOR INVESTIGATION ===")

for col in ["rain_mm_week", "temp_c"]:
    print(f"\nSensor: {col}")

    print("Most frequent values:")
    print(df[col].value_counts().head(10))

    # Count consecutive identical readings
    groups = df[col].ne(df[col].shift()).cumsum()
    runs = df.groupby(groups).agg(
        value=(col, "first"),
        count=(col, "size"),
        start_date=("date", "first"),
        end_date=("date", "last")
    )

    print("\nLongest repeated sequences:")
    print(
        runs.sort_values("count", ascending=False)
        .head(10)
        .to_string(index=False)
    )

print("\n=== INVESTIGATION COMPLETE ===")
