
from pathlib import Path
import mlflow
import mlflow.sklearn
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

# Connect to our existing MLflow database
mlflow.set_tracking_uri(
    f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
)

# Load registered model Version 1
model = mlflow.sklearn.load_model(
    "models:/TeaLeafYieldPredictor/1"
)

print("Registered model loaded successfully!")

# Read validation data
data = pd.read_csv(
    ROOT / "data/processed/val_clean.csv"
)

# Select the eight training features
features = [
    "worker_exp_yr",
    "rain_mm_week",
    "slope_deg",
    "temp_c",
    "bush_age_yr",
    "hours_worked",
    "flush",
    "section",
]

X = data[features].head(5)

# Generate predictions
predictions = model.predict(X)

print("\n=== SAMPLE PREDICTIONS ===")

for i, prediction in enumerate(predictions):
    actual = data.iloc[i]["kg_per_worker"]

    print(
        f"Record {i + 1}: "
        f"Predicted = {prediction:.2f}, "
        f"Actual = {actual:.2f}"
    )
