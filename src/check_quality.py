
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
df = pd.read_csv(ROOT / "data" / "raw" / "train.csv")

print("\n=== DATA QUALITY INVESTIGATION ===")

print("\n1. Worker experience distribution:")
print(df["worker_exp_yr"].value_counts().sort_index().to_string())

print("\n2. Suspicious experience values (> 50):")
print(df.loc[df["worker_exp_yr"] > 50,
             "worker_exp_yr"].value_counts().sort_index())

print("\n3. Slope values above 90 degrees:")
print(df.loc[df["slope_deg"] > 90,
             "slope_deg"].value_counts().sort_index())

print("\n4. Target values above 100 kg:")
print(df.loc[df["kg_per_worker"] > 100,
             ["record_id", "kg_per_worker"]].to_string(index=False))

print("\n5. Duplicate rows:")
print(df.duplicated().sum())

print("\n6. Unique categorical values:")
for col in ["flush", "section"]:
    print(f"{col}: {df[col].unique().tolist()}")

print("\n=== INVESTIGATION COMPLETE ===")
