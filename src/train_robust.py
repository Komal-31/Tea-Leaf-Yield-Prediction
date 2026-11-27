import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.ensemble import (
    RandomForestRegressor,
    HistGradientBoostingRegressor,
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

# Reuse shared configuration and preprocessing
try:
    from .train import (
        ROOT,
        PATHS,
        DATA,
        TRAINING,
        MLFLOW,
        TARGET,
        FEATURES,
        build_pipeline,
        get_data_sha256,
    )
except ImportError:
    from train import (
        ROOT,
        PATHS,
        DATA,
        TRAINING,
        MLFLOW,
        TARGET,
        FEATURES,
        build_pipeline,
        get_data_sha256,
    )


SUSPICIOUS_THRESHOLD = DATA[
    "suspicious_target_threshold"
]


def load_data():

    train = pd.read_csv(ROOT / PATHS["processed_train"])
    val = pd.read_csv(ROOT / PATHS["processed_val"])

    original_rows = len(train)

    # Exclude suspicious targets from training only
    if TRAINING["robust"]["exclude_suspicious_targets"]:
        train = train.loc[
            train[TARGET] <= SUSPICIOUS_THRESHOLD
        ].copy()

    X_train = train[FEATURES].copy()
    y_train = train[TARGET]

    X_val = val[FEATURES].copy()
    y_val = val[TARGET]

    print("Original training rows:", original_rows)
    print("Robust training rows:", len(train))
    print("Excluded training rows:",
          original_rows - len(train))
    print("Validation rows:", len(val))

    return X_train, y_train, X_val, y_val


def main():

    X_train, y_train, X_val, y_val = load_data()

    data_sha256 = get_data_sha256()

    db_path = ROOT / PATHS["mlflow_db"]

    mlflow.set_tracking_uri(
        f"sqlite:///{db_path.as_posix()}"
    )

    mlflow.set_experiment(MLFLOW["experiment_name"])

    models = {
        "RandomForest_Robust": RandomForestRegressor(
            **TRAINING["random_forest"],
            random_state=TRAINING["random_state"]
        ),

        "HistGradientBoosting_Robust":
            HistGradientBoostingRegressor(
                **TRAINING["hist_gradient_boosting"],
                random_state=TRAINING["random_state"]
            ),
    }

    results = []

    for name, estimator in models.items():

        print(f"\nTraining {name}...")

        pipeline = build_pipeline(estimator, X_train)

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
                "training_strategy": (
                    "exclude_flagged_targets"
                    if TRAINING["robust"][
                        "exclude_suspicious_targets"
                    ]
                    else "include_all_targets"
                ),
                "target_threshold": SUSPICIOUS_THRESHOLD,
                "training_rows": len(X_train),
                "validation_rows": len(X_val),
                "data_sha256": data_sha256,
                "cleaning_version": "river_side_corrected",
            })

            mlflow.log_params(estimator.get_params())

            mlflow.log_artifact(
                str(ROOT / "params.yaml"),
                artifact_path="configuration"
            )

            mlflow.log_metric("mae", mae)
            mlflow.log_metric("val_mae", mae)
            mlflow.log_metric("val_rmse", rmse)
            mlflow.log_metric("val_r2", r2)

            mlflow.sklearn.log_model( # type: ignore
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
        / TRAINING["reports"]["robust_comparison"]
    )

    results_df.to_csv(report_path, index=False)

    print("\n=== ROBUST MODEL COMPARISON ===")
    print(results_df.to_string(index=False))

    baseline_path = (
        reports_dir
        / TRAINING["reports"]["baseline_comparison"]
    )

    if baseline_path.exists():

        baseline = pd.read_csv(baseline_path)

        dummy = baseline.loc[
            baseline["model"] == "DummyRegressor", "mae"
        ]

        if not dummy.empty:
            print("Dummy baseline MAE:",
                  round(dummy.iloc[0], 4))

    print("Best robust model:",
          results_df.iloc[0]["model"])

    print("Best robust MAE:",
          round(results_df.iloc[0]["mae"], 4))

    print("Report saved:", report_path)


if __name__ == "__main__":
    main()