"""Corrected chronological retraining, V3 vs V4 comparison, promotion rule, and MLflow registry."""

from pathlib import Path
import hashlib
import json
import yaml

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    from src.train import build_pipeline as train_build_pipeline
except ImportError:
    from train import build_pipeline as train_build_pipeline


ROOT = Path(__file__).resolve().parent.parent
SKOPS_TRUSTED_TYPES = [
    "numpy.dtype",
    "sklearn.tree._tree.Tree",
    "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
]


def load_params(params_path: Path | None = None) -> dict:
    """Load centralized configuration from params.yaml."""
    path = params_path or (ROOT / "params.yaml")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def compute_file_sha256(path: Path) -> str:
    """Compute hex SHA-256 digest of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_pipeline(model, X_train: pd.DataFrame):
    """Reuse the project's params.yaml-configured preprocessing pipeline."""
    return train_build_pipeline(model, X_train)


def chronological_split(
    df: pd.DataFrame,
    date_col: str = "date",
    id_col: str = "record_id",
    train_ratio: float = 0.7,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deterministically sort by date ascending (tie-broken by record_id) and split."""
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
    baseline_mae: float,
    candidate_mae: float,
    improvement_threshold: float = 0.05,
    baseline_version: int = 3,
    candidate_version: int = 4,
) -> dict:
    """Apply the 5% promotion rule comparing candidate against the explicit baseline version."""
    if baseline_mae < 0 or candidate_mae < 0:
        raise ValueError("MAE values must be non-negative")

    if baseline_mae == 0.0:
        improvement_percent = 0.0 if candidate_mae == 0.0 else float("-inf")
        promoted = False
        required_max_mae = 0.0
        reason = (
            f"Baseline V{baseline_version} MAE is 0.0; "
            f"keeping V{baseline_version} as champion."
        )
    else:
        improvement_percent = ((baseline_mae - candidate_mae) / baseline_mae) * 100.0
        required_max_mae = (1.0 - improvement_threshold) * baseline_mae
        promoted = bool(candidate_mae <= required_max_mae + 1e-12)
        if promoted:
            reason = (
                f"V{candidate_version} MAE ({candidate_mae:.4f}) is <= 95% threshold "
                f"({required_max_mae:.4f}) with {improvement_percent:.2f}% improvement over "
                f"V{baseline_version} ({baseline_mae:.4f})."
            )
        else:
            reason = (
                f"V{candidate_version} MAE ({candidate_mae:.4f}) exceeds 95% threshold "
                f"({required_max_mae:.4f}) with {improvement_percent:.2f}% improvement over "
                f"V{baseline_version} ({baseline_mae:.4f}); keeping V{baseline_version}."
            )

    return {
        "baseline_version": int(baseline_version),
        "candidate_version": int(candidate_version),
        "baseline_mae": float(baseline_mae),
        "candidate_mae": float(candidate_mae),
        "improvement_percent": float(improvement_percent),
        "required_threshold_percent": float(improvement_threshold * 100.0),
        "required_max_mae": float(required_max_mae),
        "promoted": promoted,
        "decision": "promote" if promoted else "keep",
        "champion_version": int(candidate_version) if promoted else int(baseline_version),
        "reason": reason,
    }


def guard_historical_artifacts(target_dir: Path, historical_dirs: list[Path]) -> None:
    """Prevent overwriting or deleting preserved historical model directories."""
    resolved_target = target_dir.resolve()
    for hist_dir in historical_dirs:
        if resolved_target == hist_dir.resolve():
            raise ValueError(
                f"Refusing to overwrite historical model directory: {hist_dir}"
            )


def select_baseline_model_version(
    client: MlflowClient,
    model_name: str,
    baseline_version: int,
    expected_source: str | None = None,
    expected_run_id: str | None = None,
) -> dict:
    """Explicitly select and verify the requested baseline version from MLflow registry."""
    mv = client.get_model_version(model_name, str(baseline_version))
    if expected_source and Path(mv.source).as_posix() != Path(expected_source).as_posix():
        raise ValueError(
            f"Baseline version {baseline_version} source mismatch: "
            f"expected '{expected_source}', found '{mv.source}'"
        )
    if expected_run_id and mv.run_id != expected_run_id:
        raise ValueError(
            f"Baseline version {baseline_version} run_id mismatch: "
            f"expected '{expected_run_id}', found '{mv.run_id}'"
        )
    return {
        "version": int(mv.version),
        "source": mv.source,
        "run_id": mv.run_id,
        "status": mv.status,
    }


def check_existing_retrained_provenance(
    client: MlflowClient,
    model_name: str,
    target_dir: Path,
    expected_source: str,
) -> list[dict]:
    """Inspect existing registry versions >= 4 or pointing to target_dir before saving."""
    versions = client.search_model_versions(f"name='{model_name}'")
    matches = [
        {
            "version": int(mv.version),
            "source": mv.source,
            "run_id": mv.run_id,
            "status": mv.status,
        }
        for mv in versions
        if int(mv.version) >= 4 or Path(mv.source).as_posix() == Path(expected_source).as_posix()
    ]
    if matches:
        raise RuntimeError(
            f"Found pre-existing retrained version(s) in registry: {matches}. "
            "Inspect provenance before creating another version."
        )
    if target_dir.exists() and any(target_dir.iterdir()):
        raise RuntimeError(
            f"Target model directory '{target_dir}' already exists and is non-empty. "
            "Inspect provenance rather than overwriting."
        )
    return matches


def verify_all_runs_have_sha256(experiment_id: str, expected_sha256: str) -> int:
    """Verify every finished run in the experiment logs the expected data_sha256."""
    runs = mlflow.search_runs(
        experiment_ids=[experiment_id],
        filter_string="attributes.status = 'FINISHED'",
    )
    if runs.empty:
        raise ValueError("No finished MLflow runs found")
    if "params.data_sha256" not in runs.columns:
        raise ValueError("Missing params.data_sha256 in MLflow runs")
    missing = runs[runs["params.data_sha256"] != expected_sha256]
    if not missing.empty:
        bad_ids = missing["run_id"].tolist()
        raise ValueError(f"Runs missing expected data_sha256: {bad_ids}")
    return int(len(runs))


def main() -> dict:
    """Execute corrected V3-baseline chronological retraining, comparison, and registration."""
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

    baseline_version = int(retrain_cfg["baseline_model_version"])
    baseline_dir = ROOT / retrain_cfg["baseline_model_dir"]
    expected_baseline_run_id = str(retrain_cfg["baseline_run_id"])

    retrained_rel_dir = str(retrain_cfg["retrained_model_dir"])
    retrained_dir = ROOT / retrained_rel_dir
    historical_dirs = [ROOT / p for p in retrain_cfg["historical_model_dirs"]]
    guard_historical_artifacts(retrained_dir, historical_dirs)

    raw_train_path = ROOT / paths["raw_train"]
    data_sha256 = compute_file_sha256(raw_train_path)

    train_clean = pd.read_csv(ROOT / paths["processed_train"])
    val_clean = pd.read_csv(ROOT / paths["processed_val"])
    later_df = pd.read_csv(ROOT / paths["raw_later"])

    mlflow_db_path = ROOT / paths["mlflow_db"]
    mlflow.set_tracking_uri(f"sqlite:///{mlflow_db_path.as_posix()}")
    experiment_name = mlflow_cfg["experiment_name"]
    mlflow.set_experiment(experiment_name)
    client = MlflowClient()
    experiment = mlflow.get_experiment_by_name(experiment_name)

    model_name = mlflow_cfg["registered_model_name"]
    baseline_info = select_baseline_model_version(
        client=client,
        model_name=model_name,
        baseline_version=baseline_version,
        expected_source=retrain_cfg["baseline_model_dir"],
        expected_run_id=expected_baseline_run_id,
    )
    check_existing_retrained_provenance(
        client=client,
        model_name=model_name,
        target_dir=retrained_dir,
        expected_source=retrained_rel_dir,
    )

    # Load corrected baseline model (V3)
    baseline_model = mlflow.sklearn.load_model(str(baseline_dir))

    # Chronological 70:30 split of later.csv (deterministic tie-break by record_id)
    later_train, later_eval = chronological_split(
        later_df,
        date_col=date_col,
        id_col=id_col,
        train_ratio=float(retrain_cfg["later_train_ratio"]),
    )

    # Combine corrected cleaned original train.csv (train_clean + val_clean) excluding >100 typo targets
    cleaned_full = pd.concat([train_clean, val_clean], ignore_index=True)
    cleaned_robust = cleaned_full.loc[cleaned_full[target] <= suspicious_threshold].copy()
    excluded_typo_rows = int(len(cleaned_full) - len(cleaned_robust))

    combined_train = pd.concat([cleaned_robust, later_train], ignore_index=True)
    X_train_v4 = combined_train[features].copy()
    y_train_v4 = combined_train[target]

    X_eval = later_eval[features].copy()
    y_eval = later_eval[target]

    # Fit preprocessing and regressor strictly on combined_train
    v4_estimator = HistGradientBoostingRegressor(
        max_iter=int(retrain_cfg["max_iter"]),
        random_state=int(retrain_cfg["random_state"]),
    )
    v4_pipeline = build_pipeline(v4_estimator, X_train_v4)

    split_boundary_train_end = str(later_train[date_col].iloc[-1])
    split_boundary_eval_start = str(later_eval[date_col].iloc[0])

    with mlflow.start_run(
        run_name=str(retrain_cfg["retrained_model_name"]),
        experiment_id=experiment.experiment_id,
    ) as run:
        v4_pipeline.fit(X_train_v4, y_train_v4)

        v3_eval_preds = baseline_model.predict(X_eval)
        v4_eval_preds = v4_pipeline.predict(X_eval)

        v3_metrics = evaluate_regression(y_eval, v3_eval_preds)
        v4_metrics = evaluate_regression(y_eval, v4_eval_preds)

        # Register path assigns next registry version cleanly after logging
        # Preliminary decision assuming candidate_version = baseline_version + 1 (verified below)
        promo = decide_promotion(
            baseline_mae=v3_metrics["mae"],
            candidate_mae=v4_metrics["mae"],
            improvement_threshold=float(retrain_cfg["promotion_improvement_threshold"]),
            baseline_version=baseline_version,
            candidate_version=baseline_version + 1,
        )

        mlflow.log_params(
            {
                "model_name": str(retrain_cfg["retrained_model_name"]),
                "model_type": retrain_cfg["model_type"],
                "baseline_model_version": baseline_version,
                "baseline_run_id": baseline_info["run_id"],
                "data_sha256": data_sha256,
                "cleaning_version": "river_side_corrected",
                "training_strategy": "corrected_cleaned_train_robust_plus_later_70pct",
                "cleaned_train_rows": int(len(cleaned_robust)),
                "excluded_suspicious_rows": excluded_typo_rows,
                "later_train_rows": int(len(later_train)),
                "training_rows": int(len(combined_train)),
                "evaluation_rows": int(len(later_eval)),
                "later_train_date_range": f"{later_train[date_col].iloc[0]}..{split_boundary_train_end}",
                "later_eval_date_range": f"{split_boundary_eval_start}..{later_eval[date_col].iloc[-1]}",
                "split_boundary": f"{split_boundary_train_end} -> {split_boundary_eval_start}",
                "promotion_decision": promo["decision"],
            }
        )
        mlflow.log_params(v4_estimator.get_params())
        mlflow.log_metrics(
            {
                "mae": v4_metrics["mae"],
                "eval_mae": v4_metrics["mae"],
                "eval_rmse": v4_metrics["rmse"],
                "baseline_eval_mae": v3_metrics["mae"],
                "baseline_eval_rmse": v3_metrics["rmse"],
                "improvement_percent": promo["improvement_percent"],
            }
        )
        mlflow.log_artifact(str(ROOT / "params.yaml"), artifact_path="configuration")
        mlflow.sklearn.log_model(
            sk_model=v4_pipeline,
            name="model",
            input_example=X_train_v4.head(3),
            skops_trusted_types=SKOPS_TRUSTED_TYPES,
        )
        v4_run_id = run.info.run_id

    # Save portable V4 artifact without touching V1/V2/V3
    mlflow.sklearn.save_model(
        sk_model=v4_pipeline,
        path=str(retrained_dir),
        input_example=X_train_v4.head(3),
        skops_trusted_types=SKOPS_TRUSTED_TYPES,
    )

    v4_mv = client.create_model_version(
        name=model_name,
        source=retrained_rel_dir,
        run_id=v4_run_id,
        description=(
            f"Retrained model on corrected cleaned train ({len(cleaned_robust)} rows) + "
            f"first 70% of later.csv ({len(later_train)} rows), evaluated against V{baseline_version}."
        ),
    )
    retrained_version = int(v4_mv.version)

    # Recompute decision with exact registry version returned by MLflow
    promo = decide_promotion(
        baseline_mae=v3_metrics["mae"],
        candidate_mae=v4_metrics["mae"],
        improvement_threshold=float(retrain_cfg["promotion_improvement_threshold"]),
        baseline_version=baseline_version,
        candidate_version=retrained_version,
    )
    champion_version = promo["champion_version"]
    champion_alias = mlflow_cfg["champion_alias"]
    client.set_registered_model_alias(model_name, champion_alias, str(champion_version))

    # Verify @champion loads and resolves properly
    loaded_champion = mlflow.sklearn.load_model(f"models:/{model_name}@{champion_alias}")
    _ = loaded_champion.predict(X_eval.head(2))

    # Verify all existing and new runs contain data_sha256
    total_verified_runs = verify_all_runs_have_sha256(experiment.experiment_id, data_sha256)

    reports_dir = ROOT / paths["reports_dir"]
    reports_dir.mkdir(parents=True, exist_ok=True)

    comparison_df = pd.DataFrame(
        [
            {
                "assignment_role": "baseline",
                "registry_version": baseline_version,
                "model_name": "HistGradientBoosting_Robust_V3_corrected",
                "train_rows": 837,
                "later_train_rows": 0,
                "eval_rows": int(len(later_eval)),
                "eval_date_start": str(later_eval[date_col].iloc[0]),
                "eval_date_end": str(later_eval[date_col].iloc[-1]),
                "split_boundary": f"{split_boundary_train_end} -> {split_boundary_eval_start}",
                "mae": round(v3_metrics["mae"], 6),
                "rmse": round(v3_metrics["rmse"], 6),
                "improvement_percent": 0.0,
                "promotion_decision": promo["decision"],
                "is_champion": champion_version == baseline_version,
                "mlflow_run_id": baseline_info["run_id"],
            },
            {
                "assignment_role": "retrained",
                "registry_version": retrained_version,
                "model_name": str(retrain_cfg["retrained_model_name"]),
                "train_rows": int(len(combined_train)),
                "later_train_rows": int(len(later_train)),
                "eval_rows": int(len(later_eval)),
                "eval_date_start": str(later_eval[date_col].iloc[0]),
                "eval_date_end": str(later_eval[date_col].iloc[-1]),
                "split_boundary": f"{split_boundary_train_end} -> {split_boundary_eval_start}",
                "mae": round(v4_metrics["mae"], 6),
                "rmse": round(v4_metrics["rmse"], 6),
                "improvement_percent": round(promo["improvement_percent"], 4),
                "promotion_decision": promo["decision"],
                "is_champion": champion_version == retrained_version,
                "mlflow_run_id": v4_run_id,
            },
        ]
    )
    comparison_path = reports_dir / retrain_cfg["comparison_filename"]
    comparison_df.to_csv(comparison_path, index=False)

    champion_artifact_dir = (
        retrained_rel_dir if champion_version == retrained_version else retrain_cfg["baseline_model_dir"]
    )
    decision_report = {
        "experiment_name": experiment_name,
        "registered_model_name": model_name,
        "data_sha256": data_sha256,
        "baseline_model_version": baseline_version,
        "baseline_run_id": baseline_info["run_id"],
        "retrained_model_version": retrained_version,
        "retrained_run_id": v4_run_id,
        "champion_version": champion_version,
        "champion_alias": champion_alias,
        "champion_artifact_dir": champion_artifact_dir,
        "total_verified_mlflow_runs": total_verified_runs,
        "cleaned_train_rows": int(len(cleaned_robust)),
        "excluded_suspicious_rows": excluded_typo_rows,
        "later_total_rows": int(len(later_df)),
        "later_train_rows": int(len(later_train)),
        "later_eval_rows": int(len(later_eval)),
        "later_train_date_range": [str(later_train[date_col].iloc[0]), split_boundary_train_end],
        "later_eval_date_range": [split_boundary_eval_start, str(later_eval[date_col].iloc[-1])],
        "v3_eval_mae": round(v3_metrics["mae"], 6),
        "v3_eval_rmse": round(v3_metrics["rmse"], 6),
        "v4_eval_mae": round(v4_metrics["mae"], 6),
        "v4_eval_rmse": round(v4_metrics["rmse"], 6),
        "improvement_percent": round(promo["improvement_percent"], 4),
        "required_threshold_percent": promo["required_threshold_percent"],
        "required_max_mae": round(promo["required_max_mae"], 6),
        "promotion_decision": promo["decision"],
        "promoted": promo["promoted"],
        "decision_reason": promo["reason"],
    }
    decision_path = reports_dir / retrain_cfg["decision_filename"]
    with open(decision_path, "w", encoding="utf-8") as f:
        json.dump(decision_report, f, indent=2)

    # Log generated reports into the V4 MLflow run for complete traceability
    with mlflow.start_run(run_id=v4_run_id):
        mlflow.log_artifact(str(comparison_path), artifact_path="reports")
        mlflow.log_artifact(str(decision_path), artifact_path="reports")

    print("\n=== MODEL COMPARISON (FINAL 30% OF LATER.CSV: V3 vs V4) ===")
    print(comparison_df.to_string(index=False))
    print("\n=== PROMOTION DECISION ===")
    print(json.dumps(decision_report, indent=2))
    return decision_report


if __name__ == "__main__":
    main()
