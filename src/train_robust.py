
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.compose import ColumnTransformer
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

TARGET = "kg_per_worker"

EXCLUDED_COLUMNS = [
    "record_id",
    "date",
    "target_outlier_flag",
    TARGET,
]

# This is an experimental threshold, not a verified
# correction to the original target values.
SUSPICIOUS_THRESHOLD = 100


def load_data():
    train = pd.read_csv(DATA_DIR / "train_clean.csv")
    val = pd.read_csv(DATA_DIR / "val_clean.csv")

    original_rows = len(train)

    # Exclude suspicious targets from training only.
    # Validation remains completely unchanged.
    train = train.loc[
        train[TARGET] <= SUSPICIOUS_THRESHOLD
    ].copy()

    features = [
        column for column in train.columns
        if column not in EXCLUDED_COLUMNS
    ]

    X_train = train[features]
    y_train = train[TARGET]

    X_val = val[features]
    y_val = val[TARGET]

    print("Original training rows:", original_rows)
    print("Robust training rows:", len(train))
    print("Excluded training rows:", original_rows - len(train))
    print("Validation rows:", len(val))

    return X_train, y_train, X_val, y_val


def build_pipeline(model, X_train):
    numeric_features = X_train.select_dtypes(
        include="number"
    ).columns.tolist()

    categorical_features = X_train.select_dtypes(
        include=["object", "str", "category"]
    ).columns.tolist()

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_transformer = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="most_frequent"),
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


def main():
    X_train, y_train, X_val, y_val = load_data()

    mlflow.set_tracking_uri(
        f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
    )

    mlflow.set_experiment("Tea_Leaf_Yield_Prediction")

    models = {
        "RandomForest_Robust": RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
        ),
        "HistGradientBoosting_Robust":
            HistGradientBoostingRegressor(
                max_iter=200,
                random_state=42,
            ),
    }

    results = []

    for name, estimator in models.items():
        print(f"\nTraining {name}...")

        pipeline = build_pipeline(estimator, X_train)

        with mlflow.start_run(run_name=name):
            pipeline.fit(X_train, y_train)

            predictions = pipeline.predict(X_val)

            mae = mean_absolute_error(y_val, predictions)
            rmse = mean_squared_error(
                y_val, predictions
            ) ** 0.5
            r2 = r2_score(y_val, predictions)

            mlflow.log_params({
                "model_name": name,
                "training_strategy": "exclude_flagged_targets",
                "target_threshold": SUSPICIOUS_THRESHOLD,
                "training_rows": len(X_train),
                "validation_rows": len(X_val),
                "random_state": 42,
            })

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

    print("\n=== ROBUST MODEL COMPARISON ===")
    print(results_df.to_string(index=False))

    reports_dir = ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)

    results_df.to_csv(
        reports_dir / "robust_model_comparison.csv",
        index=False,
    )

    print("\nOriginal baseline MAE: 7.0483")
    print(
        "Best robust model MAE:",
        round(results_df.iloc[0]["mae"], 4),
    )


if __name__ == "__main__":
    main()
