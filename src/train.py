from pathlib import Path
import hashlib
import yaml

import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# Load centralized configuration
ROOT = Path(__file__).resolve().parent.parent

with open(ROOT / "params.yaml", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

PATHS = CONFIG["paths"]
DATA = CONFIG["data"]
TRAINING = CONFIG["training"]
MLFLOW = CONFIG["mlflow"]

TARGET = DATA["target_column"]
FEATURES = DATA["features"]


def get_data_sha256():
    raw_path = ROOT / PATHS["raw_train"]
    return hashlib.sha256(raw_path.read_bytes()).hexdigest()


def load_data():
    train = pd.read_csv(ROOT / PATHS["processed_train"])
    val = pd.read_csv(ROOT / PATHS["processed_val"])

    X_train = train[FEATURES].copy()
    X_val = val[FEATURES].copy()

    y_train = train[TARGET]
    y_val = val[TARGET]

    return X_train, X_val, y_train, y_val


def build_pipeline(model, X_train):

    numeric_features = X_train.select_dtypes(
        include="number"
    ).columns.tolist()

    categorical_features = [
        col for col in X_train.columns
        if col not in numeric_features
    ]

    preprocessing = TRAINING["preprocessing"]

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(
            strategy=preprocessing["numeric_imputation"]
        ))
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(
            strategy=preprocessing["categorical_imputation"]
        )),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        ))
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_transformer, numeric_features),
        ("categorical", categorical_transformer,
         categorical_features),
    ])

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])


def train_models():

    X_train, X_val, y_train, y_val = load_data()

    data_sha256 = get_data_sha256()

    print("Training records:", len(X_train))
    print("Validation records:", len(X_val))
    print("Features:", X_train.columns.tolist())
    print("Dataset SHA256:", data_sha256)

    db_path = ROOT / PATHS["mlflow_db"]

    mlflow.set_tracking_uri(
        f"sqlite:///{db_path.as_posix()}"
    )

    mlflow.set_experiment(MLFLOW["experiment_name"])

    models = {
        "DummyRegressor": DummyRegressor(
            **TRAINING["dummy"]
        ),

        "RandomForest": RandomForestRegressor(
            **TRAINING["random_forest"],
            random_state=TRAINING["random_state"]
        ),

        "HistGradientBoosting":
            HistGradientBoostingRegressor(
                **TRAINING["hist_gradient_boosting"],
                random_state=TRAINING["random_state"]
            ),
    }

    results = []

    for name, model in models.items():

        print(f"\nTraining {name}...")

        pipeline = build_pipeline(model, X_train)

        with mlflow.start_run(
            run_name=f"{name}_corrected"
        ) as run:

            pipeline.fit(X_train, y_train)

            predictions = pipeline.predict(X_val)

            mae = mean_absolute_error(y_val, predictions)
            rmse = mean_squared_error(
                y_val, predictions
            ) ** 0.5
            r2 = r2_score(y_val, predictions)

            mlflow.log_params({
                "model_name": name,
                "training_rows": len(X_train),
                "validation_rows": len(X_val),
                "data_sha256": data_sha256,
                "cleaning_version": "river_side_corrected",
                "training_strategy": "baseline",
            })

            mlflow.log_params(model.get_params())

            mlflow.log_artifact(
                str(ROOT / "params.yaml"),
                artifact_path="configuration"
            )

            mlflow.log_metric("mae", mae)
            mlflow.log_metric("val_mae", mae)
            mlflow.log_metric("val_rmse", rmse)
            mlflow.log_metric("val_r2", r2)

            mlflow.sklearn.log_model(
                sk_model=pipeline,
                name="model",
                input_example=X_train.head(3),
                skops_trusted_types=[
                    "numpy.dtype",
                    "sklearn.tree._tree.Tree",
                    "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
                ],
            )

            results.append({
                "model": name,
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
                "mlflow_run_id": run.info.run_id,
            })

            print(f"MAE: {mae:.4f}")
            print(f"RMSE: {rmse:.4f}")
            print(f"R2: {r2:.4f}")

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("mae")

    reports_dir = ROOT / PATHS["reports_dir"]
    reports_dir.mkdir(parents=True, exist_ok=True)

    report_path = (
        reports_dir
        / TRAINING["reports"]["baseline_comparison"]
    )

    results_df.to_csv(report_path, index=False)

    print("\n=== BASELINE MODEL COMPARISON ===")
    print(results_df.to_string(index=False))

    print("\nBest model:", results_df.iloc[0]["model"])
    print("Best MAE:", round(results_df.iloc[0]["mae"], 4))
    print("Report saved:", report_path)


if __name__ == "__main__":
    train_models()