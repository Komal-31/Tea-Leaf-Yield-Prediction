"""Chronological retraining, model comparison, promotion rule, and MLflow registry for MP148 Stage 4."""

from pathlib import Path
import hashlib
import json
import shutil
import yaml

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(__file__).resolve().parent.parent
SKOPS_TRUSTED_TYPES = [
    "numpy.dtype",
    "sklearn.tree._tree.Tree",
    "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
]
ORIGINAL_V1_HISTORICAL_RUN_ID = "77cd0bd9428041ef8b2d6d23b161296e"


def load_params(params_path: Path | None = None) -> dict:
    """Load configuration from params.yaml."""
    path = params_path or (ROOT / "params.yaml")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def compute_file_sha256(path: Path) -> str:
    """Compute hex SHA-256 digest of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_pipeline(model, X_train: pd.DataFrame) -> Pipeline:
    """Build preprocessing + regressor pipeline matching src/train_robust.py."""
    numeric_features = X_train.select_dtypes(include="number").columns.tolist()
    categorical_features = X_train.select_dtypes(
        include=["object", "str", "category"]
    ).columns.tolist()

    numeric_transformer = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        [
            ("numeric", numeric_transformer, numeric_features),
            ("categorical", categorical_transformer, categorical_features),
        ]
    )

    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def chronological_split(
    df: pd.DataFrame,
    date_col: str = "date",
    id_col: str = "record_id",
    train_ratio: float = 0.7,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deterministically sort by date ascending and split into train and evaluation parts."""
    if not (0.0 < train_ratio < 1.0):
        raise ValueError("train_ratio must be strictly between 0 and 1")
    if df.empty:
        raise ValueError("Cannot split an empty DataFrame")

    sorted_df = df.copy()
    sorted_df["_parsed_date"] = pd.to_datetime(sorted_df[date_col], errors="raise")
    sort_cols = ["_parsed_date", id_col] if id_col in sorted_df.columns else ["_parsed_date"]
    sorted_df = sorted_df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    sorted_df = sorted_df.drop(columns=["_parsed_date"])

    split_idx = int(len(sorted_df) * train_ratio)
    if split_idx <= 0 or split_idx >= len(sorted_df):
        raise ValueError("Split produced an empty partition")

    train_part = sorted_df.iloc[:split_idx].copy().reset_index(drop=True)
    eval_part = sorted_df.iloc[split_idx:].copy().reset_index(drop=True)

    if id_col in sorted_df.columns:
        overlap = set(train_part[id_col]).intersection(set(eval_part[id_col]))
        if overlap:
            raise ValueError(f"Record overlap detected across split: {overlap}")

    return train_part, eval_part


def evaluate_regression(y_true, y_pred) -> dict[str, float]:
    """Compute MAE and RMSE for regression predictions."""
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(mean_squared_error(y_true, y_pred) ** 0.5)
    return {"mae": mae, "rmse": rmse}


def decide_promotion(
    v1_mae: float,
    v2_mae: float,
    improvement_threshold: float = 0.05,
) -> dict:
    """Apply the exact faculty promotion rule: promote v2 iff v2_mae <= (1 - threshold) * v1_mae."""
    if v1_mae < 0 or v2_mae < 0:
        raise ValueError("MAE values must be non-negative")

    if v1_mae == 0.0:
        improvement_percent = 0.0 if v2_mae == 0.0 else float("-inf")
        promoted = False
        required_max_mae = 0.0
        reason = "Baseline v1 MAE is 0.0; keeping v1 as champion."
    else:
        improvement_percent = ((v1_mae - v2_mae) / v1_mae) * 100.0
        required_max_mae = (1.0 - improvement_threshold) * v1_mae
        promoted = bool(v2_mae <= required_max_mae + 1e-12)
        if promoted:
            reason = (
                f"v2 MAE ({v2_mae:.4f}) is <= 95% threshold ({required_max_mae:.4f}) "
                f"with {improvement_percent:.2f}% improvement over v1 ({v1_mae:.4f})."
            )
        else:
            reason = (
                f"v2 MAE ({v2_mae:.4f}) exceeds 95% threshold ({required_max_mae:.4f}) "
                f"with {improvement_percent:.2f}% improvement over v1 ({v1_mae:.4f}); keeping v1."
            )

    return {
        "v1_mae": float(v1_mae),
        "v2_mae": float(v2_mae),
        "improvement_percent": float(improvement_percent),
        "required_threshold_percent": float(improvement_threshold * 100.0),
        "required_max_mae": float(required_max_mae),
        "promoted": promoted,
        "decision": "promote" if promoted else "keep",
        "champion_version": 2 if promoted else 1,
        "reason": reason,
    }


def _reproduce_baseline_runs_if_needed(
    client: MlflowClient,
    experiment_id: str,
    data_sha256: str,
    train_clean: pd.DataFrame,
    val_clean: pd.DataFrame,
    features: list[str],
    target: str,
    suspicious_threshold: float,
) -> str:
    """Reproduce Stage 2 baseline & robust runs if mlflow.db has no prior runs.

    Returns the genuine MLflow run_id corresponding to HistGradientBoosting_Robust.
    """
    existing = mlflow.search_runs(
        experiment_ids=[experiment_id],
        filter_string="attributes.status = 'FINISHED'",
    )
    if not existing.empty:
        hgb_runs = existing[existing.get("params.model_name") == "HistGradientBoosting_Robust"]
        if not hgb_runs.empty:
            return str(hgb_runs.iloc[0]["run_id"])
        return str(existing.iloc[0]["run_id"])

    X_train_full = train_clean[features].copy()
    y_train_full = train_clean[target]
    X_val = val_clean[features].copy()
    y_val = val_clean[target]

    baseline_models = {
        "DummyRegressor": DummyRegressor(strategy="median"),
        "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        "HistGradientBoosting": HistGradientBoostingRegressor(max_iter=200, random_state=42),
    }

    for name, estimator in baseline_models.items():
        pipe = build_pipeline(estimator, X_train_full)
        with mlflow.start_run(run_name=f"{name}_reproduced", experiment_id=experiment_id):
            pipe.fit(X_train_full, y_train_full)
            preds = pipe.predict(X_val)
            mae = float(mean_absolute_error(y_val, preds))
            rmse = float(mean_squared_error(y_val, preds) ** 0.5)
            r2 = float(r2_score(y_val, preds))
            mlflow.log_params(
                {
                    "model_name": name,
                    "data_sha256": data_sha256,
                    "training_strategy": "all_train_clean",
                    "training_rows": len(X_train_full),
                    "validation_rows": len(X_val),
                }
            )
            mlflow.log_metrics({"mae": mae, "val_mae": mae, "val_rmse": rmse, "val_r2": r2})
            mlflow.set_tag("reproduced_from", "src/train.py")

    train_robust = train_clean.loc[train_clean[target] <= suspicious_threshold].copy()
    X_train_rob = train_robust[features].copy()
    y_train_rob = train_robust[target]

    robust_models = {
        "RandomForest_Robust": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        "HistGradientBoosting_Robust": HistGradientBoostingRegressor(max_iter=200, random_state=42),
    }

    v1_run_id = ""
    for name, estimator in robust_models.items():
        pipe = build_pipeline(estimator, X_train_rob)
        with mlflow.start_run(run_name=f"{name}_reproduced", experiment_id=experiment_id) as run:
            pipe.fit(X_train_rob, y_train_rob)
            preds = pipe.predict(X_val)
            mae = float(mean_absolute_error(y_val, preds))
            rmse = float(mean_squared_error(y_val, preds) ** 0.5)
            r2 = float(r2_score(y_val, preds))
            mlflow.log_params(
                {
                    "model_name": name,
                    "data_sha256": data_sha256,
                    "training_strategy": "exclude_flagged_targets",
                    "target_threshold": suspicious_threshold,
                    "training_rows": len(X_train_rob),
                    "validation_rows": len(X_val),
                    "random_state": 42,
                }
            )
            mlflow.log_metrics({"mae": mae, "val_mae": mae, "val_rmse": rmse, "val_r2": r2})
            mlflow.set_tag("reproduced_from", "src/train_robust.py")
            if name == "HistGradientBoosting_Robust":
                v1_run_id = run.info.run_id
                mlflow.set_tag("preserved_v1_artifact", "models/tea_leaf_v1")
                mlflow.set_tag("original_v1_historical_run_id", ORIGINAL_V1_HISTORICAL_RUN_ID)

    return v1_run_id


def _ensure_registered_model(client: MlflowClient, model_name: str) -> None:
    """Create the registered model if it does not already exist."""
    try:
        client.get_registered_model(model_name)
    except Exception:
        client.create_registered_model(model_name)


def main() -> dict:
    """Execute chronological retraining, comparison, MLflow tracking, and champion assignment."""
    params = load_params()
    paths = params["paths"]
    data_cfg = params["data"]
    retrain_cfg = params["retraining"]
    mlflow_cfg = params["mlflow"]

    features = data_cfg["features"]
    target = data_cfg["target_column"]
    date_col = data_cfg["date_column"]
    id_col = data_cfg["id_column"]
    suspicious_threshold = float(data_cfg["suspicious_target_threshold"])

    raw_train_path = ROOT / paths["raw_train"]
    data_sha256 = compute_file_sha256(raw_train_path)

    train_clean = pd.read_csv(ROOT / paths["processed_train"])
    val_clean = pd.read_csv(ROOT / paths["processed_val"])
    later_df = pd.read_csv(ROOT / paths["raw_later"])

    # MLflow tracking setup
    mlflow_db_path = ROOT / paths["mlflow_db"]
    mlflow.set_tracking_uri(f"sqlite:///{mlflow_db_path.as_posix()}")
    experiment_name = mlflow_cfg["experiment_name"]
    mlflow.set_experiment(experiment_name)
    client = MlflowClient()
    experiment = mlflow.get_experiment_by_name(experiment_name)

    v1_run_id = _reproduce_baseline_runs_if_needed(
        client=client,
        experiment_id=experiment.experiment_id,
        data_sha256=data_sha256,
        train_clean=train_clean,
        val_clean=val_clean,
        features=features,
        target=target,
        suspicious_threshold=suspicious_threshold,
    )

    model_name = mlflow_cfg["registered_model_name"]
    _ensure_registered_model(client, model_name)

    existing_versions = client.search_model_versions(f"name='{model_name}'")
    if not existing_versions:
        v1_mv = client.create_model_version(
            name=model_name,
            source=paths["v1_model_dir"],
            run_id=v1_run_id,
            description=(
                "Preserved original v1 HistGradientBoosting_Robust artifact "
                f"(historical run_id={ORIGINAL_V1_HISTORICAL_RUN_ID})."
            ),
        )
        v1_version = int(v1_mv.version)
    else:
        v1_version = min(int(mv.version) for mv in existing_versions)

    # Load preserved v1 model
    v1_model = mlflow.sklearn.load_model(str(ROOT / paths["v1_model_dir"]))

    # Chronological 70:30 split of later.csv
    later_train, later_eval = chronological_split(
        later_df,
        date_col=date_col,
        id_col=id_col,
        train_ratio=float(retrain_cfg["later_train_ratio"]),
    )

    # Combine cleaned original train.csv (train_clean + val_clean) excluding suspicious >100 targets
    cleaned_full = pd.concat([train_clean, val_clean], ignore_index=True)
    cleaned_robust = cleaned_full.loc[cleaned_full[target] <= suspicious_threshold].copy()
    excluded_typo_rows = int(len(cleaned_full) - len(cleaned_robust))

    combined_train = pd.concat([cleaned_robust, later_train], ignore_index=True)
    X_train_v2 = combined_train[features].copy()
    y_train_v2 = combined_train[target]

    X_eval = later_eval[features].copy()
    y_eval = later_eval[target]

    # Train v2 (only on combined_train; never touch later_eval)
    v2_estimator = HistGradientBoostingRegressor(
        max_iter=int(retrain_cfg["max_iter"]),
        random_state=int(retrain_cfg["random_state"]),
    )
    v2_pipeline = build_pipeline(v2_estimator, X_train_v2)

    with mlflow.start_run(run_name="HistGradientBoosting_v2", experiment_id=experiment.experiment_id) as run:
        v2_pipeline.fit(X_train_v2, y_train_v2)

        v1_eval_preds = v1_model.predict(X_eval)
        v2_eval_preds = v2_pipeline.predict(X_eval)

        v1_metrics = evaluate_regression(y_eval, v1_eval_preds)
        v2_metrics = evaluate_regression(y_eval, v2_eval_preds)

        promo = decide_promotion(
            v1_mae=v1_metrics["mae"],
            v2_mae=v2_metrics["mae"],
            improvement_threshold=float(retrain_cfg["promotion_improvement_threshold"]),
        )

        split_boundary_train_end = str(later_train[date_col].iloc[-1])
        split_boundary_eval_start = str(later_eval[date_col].iloc[0])

        mlflow.log_params(
            {
                "model_name": "HistGradientBoosting_v2",
                "model_type": retrain_cfg["model_type"],
                "max_iter": int(retrain_cfg["max_iter"]),
                "random_state": int(retrain_cfg["random_state"]),
                "data_sha256": data_sha256,
                "training_strategy": "cleaned_train_robust_plus_later_70pct",
                "cleaned_train_rows": int(len(cleaned_robust)),
                "excluded_suspicious_rows": excluded_typo_rows,
                "later_train_rows": int(len(later_train)),
                "training_rows": int(len(combined_train)),
                "evaluation_rows": int(len(later_eval)),
                "later_train_date_range": f"{later_train[date_col].iloc[0]}..{split_boundary_train_end}",
                "later_eval_date_range": f"{split_boundary_eval_start}..{later_eval[date_col].iloc[-1]}",
                "promotion_decision": promo["decision"],
            }
        )
        mlflow.log_metrics(
            {
                "mae": v2_metrics["mae"],
                "eval_mae": v2_metrics["mae"],
                "eval_rmse": v2_metrics["rmse"],
                "v1_eval_mae": v1_metrics["mae"],
                "v1_eval_rmse": v1_metrics["rmse"],
                "improvement_percent": promo["improvement_percent"],
            }
        )
        mlflow.sklearn.log_model(
            sk_model=v2_pipeline,
            name="model",
            input_example=X_train_v2.head(3),
            skops_trusted_types=SKOPS_TRUSTED_TYPES,
        )
        v2_run_id = run.info.run_id

    # Save portable v2 artifact
    v2_dir = ROOT / paths["v2_model_dir"]
    if v2_dir.exists():
        shutil.rmtree(v2_dir)
    mlflow.sklearn.save_model(
        sk_model=v2_pipeline,
        path=str(v2_dir),
        input_example=X_train_v2.head(3),
        skops_trusted_types=SKOPS_TRUSTED_TYPES,
    )

    # Register v2 (referencing portable relative path in repo)
    v2_mv = client.create_model_version(
        name=model_name,
        source=paths["v2_model_dir"],
        run_id=v2_run_id,
        description=(
            f"Retrained v2 on cleaned train ({len(cleaned_robust)} rows) + "
            f"first 70% of later.csv ({len(later_train)} rows)."
        ),
    )
    v2_version = int(v2_mv.version)

    champion_version = v2_version if promo["promoted"] else v1_version
    champion_alias = mlflow_cfg["champion_alias"]
    client.set_registered_model_alias(model_name, champion_alias, str(champion_version))

    # Verify champion loads cleanly
    loaded_champion = mlflow.sklearn.load_model(f"models:/{model_name}@{champion_alias}")
    _ = loaded_champion.predict(X_eval.head(2))

    # Save comparison CSV
    reports_dir = ROOT / paths["reports_dir"]
    reports_dir.mkdir(parents=True, exist_ok=True)

    comparison_df = pd.DataFrame(
        [
            {
                "model_version": f"v{v1_version}",
                "model_name": "HistGradientBoosting_Robust_v1",
                "train_rows": 837,
                "later_train_rows": 0,
                "eval_rows": int(len(later_eval)),
                "eval_date_start": str(later_eval[date_col].iloc[0]),
                "eval_date_end": str(later_eval[date_col].iloc[-1]),
                "split_boundary": f"{split_boundary_train_end} -> {split_boundary_eval_start}",
                "mae": round(v1_metrics["mae"], 6),
                "rmse": round(v1_metrics["rmse"], 6),
                "improvement_percent": 0.0,
                "promotion_decision": promo["decision"],
                "is_champion": not promo["promoted"],
                "mlflow_run_id": v1_run_id,
            },
            {
                "model_version": f"v{v2_version}",
                "model_name": "HistGradientBoosting_v2",
                "train_rows": int(len(combined_train)),
                "later_train_rows": int(len(later_train)),
                "eval_rows": int(len(later_eval)),
                "eval_date_start": str(later_eval[date_col].iloc[0]),
                "eval_date_end": str(later_eval[date_col].iloc[-1]),
                "split_boundary": f"{split_boundary_train_end} -> {split_boundary_eval_start}",
                "mae": round(v2_metrics["mae"], 6),
                "rmse": round(v2_metrics["rmse"], 6),
                "improvement_percent": round(promo["improvement_percent"], 4),
                "promotion_decision": promo["decision"],
                "is_champion": promo["promoted"],
                "mlflow_run_id": v2_run_id,
            },
        ]
    )
    comparison_path = reports_dir / "model_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)

    decision_report = {
        "experiment_name": experiment_name,
        "registered_model_name": model_name,
        "data_sha256": data_sha256,
        "v1_run_id": v1_run_id,
        "v1_historical_run_id": ORIGINAL_V1_HISTORICAL_RUN_ID,
        "v2_run_id": v2_run_id,
        "v1_version": v1_version,
        "v2_version": v2_version,
        "champion_version": champion_version,
        "champion_alias": champion_alias,
        "cleaned_train_rows": int(len(cleaned_robust)),
        "excluded_suspicious_rows": excluded_typo_rows,
        "later_total_rows": int(len(later_df)),
        "later_train_rows": int(len(later_train)),
        "later_eval_rows": int(len(later_eval)),
        "later_train_date_range": [str(later_train[date_col].iloc[0]), split_boundary_train_end],
        "later_eval_date_range": [split_boundary_eval_start, str(later_eval[date_col].iloc[-1])],
        "v1_mae": round(v1_metrics["mae"], 6),
        "v1_rmse": round(v1_metrics["rmse"], 6),
        "v2_mae": round(v2_metrics["mae"], 6),
        "v2_rmse": round(v2_metrics["rmse"], 6),
        "improvement_percent": round(promo["improvement_percent"], 4),
        "required_threshold_percent": promo["required_threshold_percent"],
        "required_max_mae": round(promo["required_max_mae"], 6),
        "promotion_decision": promo["decision"],
        "promoted": promo["promoted"],
        "decision_reason": promo["reason"],
    }
    decision_path = reports_dir / "promotion_decision.json"
    with open(decision_path, "w", encoding="utf-8") as f:
        json.dump(decision_report, f, indent=2)

    print("\n=== MODEL COMPARISON (LAST 30% OF LATER.CSV) ===")
    print(comparison_df.to_string(index=False))
    print("\n=== PROMOTION DECISION ===")
    print(json.dumps(decision_report, indent=2))
    return decision_report


if __name__ == "__main__":
    main()
