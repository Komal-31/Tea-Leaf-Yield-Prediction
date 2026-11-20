
from pathlib import Path
import json

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


# Project paths
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"
MLRUNS_DIR = ROOT / "mlruns"

TARGET = "kg_per_worker"

# Never use identifiers, dates or target-derived flags
# as model features in this initial baseline.
DROP_COLUMNS = [
    "record_id",
    "date",
    "target_outlier_flag",
    TARGET,
]


def load_data():
    train = pd.read_csv(DATA_DIR / "train_clean.csv")
    val = pd.read_csv(DATA_DIR / "val_clean.csv")

    feature_columns = [
        col for col in train.columns
        if col not in DROP_COLUMNS
    ]

    X_train = train[feature_columns].copy()
    X_val = val[feature_columns].copy()

    y_train = train[TARGET]
    y_val = val[TARGET]

    return X_train, X_val, y_train, y_val


def build_pipeline(model, X_train):
    numeric_features = X_train.select_dtypes(
        include="number"
    ).columns.tolist()

    categorical_features = X_train.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_transformer = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            ),
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            ),
        ),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_transformer, numeric_features),
        (
            "categorical",
            categorical_transformer,
            categorical_features,
        ),
    ])

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])


def train_models():
    X_train, X_val, y_train, y_val = load_data()

    print("Training records:", len(X_train))
    print("Validation records:", len(X_val))
    print("Features:", X_train.columns.tolist())

    # Local MLflow tracking
    
# SQLite-based MLflow tracking
    MLFLOW_DB = ROOT / "mlflow.db"

    mlflow.set_tracking_uri(
    f"sqlite:///{MLFLOW_DB.as_posix()}"
)

    mlflow.set_experiment("Tea_Leaf_Yield_Prediction")


    models = {
        "DummyRegressor": DummyRegressor(
            strategy="median"
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
        ),
        "HistGradientBoosting":
            HistGradientBoostingRegressor(
                max_iter=200,
                random_state=42,
            ),
    }

    results = []

    for name, model in models.items():
        print(f"\nTraining {name}...")

        pipeline = build_pipeline(model, X_train)

        with mlflow.start_run(run_name=name):
            pipeline.fit(X_train, y_train)

            predictions = pipeline.predict(X_val)

            mae = mean_absolute_error(
                y_val, predictions
            )

            rmse = mean_squared_error(
                y_val, predictions
            ) ** 0.5

            r2 = r2_score(y_val, predictions)

            mlflow.log_param("model_name", name)
            mlflow.log_param(
                "training_rows", len(X_train)
            )
            mlflow.log_param(
                "validation_rows", len(X_val)
            )

            mlflow.log_params(model.get_params())

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
            })

            print(f"MAE: {mae:.4f}")
            print(f"RMSE: {rmse:.4f}")
            print(f"R2: {r2:.4f}")

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("mae")

    print("\n=== MODEL COMPARISON ===")
    print(results_df.to_string(index=False))

    # Save model comparison
    reports_dir = ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)

    results_df.to_csv(
        reports_dir / "model_comparison.csv",
        index=False,
    )

    print("\nBest model:",
          results_df.iloc[0]["model"])

    print("Best validation MAE:",
          round(results_df.iloc[0]["mae"], 4))


if __name__ == "__main__":
    train_models()
