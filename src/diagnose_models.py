
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parent.parent

mlflow.set_tracking_uri(
    f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
)

experiment = mlflow.get_experiment_by_name(
    "Tea_Leaf_Yield_Prediction"
)

runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="attributes.status = 'FINISHED'",
    order_by=["start_time DESC"],
)

val = pd.read_csv(ROOT / "data/processed/val_clean.csv")

X = val.drop(
    columns=[
        "record_id",
        "date",
        "target_outlier_flag",
        "kg_per_worker",
    ],
    errors="ignore",
)

y = val["kg_per_worker"]
normal_mask = y <= 100

print("Total validation rows:", len(y))
print("Normal-range rows:", normal_mask.sum())

seen = set()

for _, run in runs.iterrows():
    name = run.get("params.model_name")

    if name not in {
        "DummyRegressor",
        "RandomForest",
        "HistGradientBoosting",
    } or name in seen:
        continue

    seen.add(name)

    
    model = mlflow.sklearn.load_model(
        f"runs:/{run['run_id']}/model"
    )


    predictions = model.predict(X)

    print(f"\n=== {name} ===")

    print(
        "Full MAE:",
        round(mean_absolute_error(y, predictions), 4),
    )

    print(
        "Normal-range MAE:",
        round(
            mean_absolute_error(
                y[normal_mask],
                predictions[normal_mask],
            ),
            4,
        ),
    )

    print(
        "Full RMSE:",
        round(
            mean_squared_error(y, predictions) ** 0.5,
            4,
        ),
    )

    print(
        "Normal-range RMSE:",
        round(
            mean_squared_error(
                y[normal_mask],
                predictions[normal_mask],
            ) ** 0.5,
            4,
        ),
    )
