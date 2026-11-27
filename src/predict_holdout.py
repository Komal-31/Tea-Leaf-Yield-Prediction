"""Generate and validate holdout predictions using the MLflow champion model."""

from pathlib import Path
import json
import yaml

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent


def load_params(params_path: Path | None = None) -> dict:
    """Load configuration from params.yaml."""
    path = params_path or (ROOT / "params.yaml")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def validate_predictions_df(
    predictions_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    id_col: str = "record_id",
) -> None:
    """Validate holdout predictions DataFrame against faculty submission constraints."""
    expected_cols = ["record_id", "prediction"]
    if list(predictions_df.columns) != expected_cols:
        raise ValueError(
            f"Expected columns {expected_cols}, got {list(predictions_df.columns)}"
        )
    if len(predictions_df) != len(holdout_df):
        raise ValueError(
            f"Prediction row count ({len(predictions_df)}) != holdout row count ({len(holdout_df)})"
        )
    if predictions_df["record_id"].isna().any():
        raise ValueError("Prediction record_id contains missing values")
    if predictions_df["record_id"].duplicated().any():
        raise ValueError("Prediction record_id contains duplicates")
    if not (predictions_df["record_id"].values == holdout_df[id_col].values).all():
        raise ValueError("Prediction record_ids do not preserve holdout order and identity")
    if predictions_df["prediction"].isna().any():
        raise ValueError("Predictions contain missing (NaN) values")
    preds_num = pd.to_numeric(predictions_df["prediction"], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(preds_num).all():
        raise ValueError("Predictions contain non-finite (+/-inf) values")


def generate_holdout_predictions(
    model,
    holdout_df: pd.DataFrame,
    feature_columns: list[str],
    output_path: Path,
    id_col: str = "record_id",
) -> pd.DataFrame:
    """Predict on holdout DataFrame, validate, and write CSV."""
    preds = np.asarray(model.predict(holdout_df[feature_columns]), dtype=float)
    result = pd.DataFrame(
        {
            "record_id": holdout_df[id_col].values,
            "prediction": np.round(preds, 4),
        }
    )
    validate_predictions_df(result, holdout_df, id_col=id_col)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result


def main() -> dict:
    """Load @champion from MLflow registry, predict holdout_features.csv, and validate."""
    params = load_params()
    paths = params["paths"]
    data_cfg = params["data"]
    mlflow_cfg = params["mlflow"]

    mlflow_db = ROOT / paths["mlflow_db"]
    mlflow.set_tracking_uri(f"sqlite:///{mlflow_db.as_posix()}")

    model_uri = f"models:/{mlflow_cfg['registered_model_name']}@{mlflow_cfg['champion_alias']}"
    champion_model = mlflow.sklearn.load_model(model_uri)

    holdout_df = pd.read_csv(ROOT / paths["raw_holdout"])
    output_path = ROOT / paths["predictions_dir"] / "holdout_predictions.csv"

    preds_df = generate_holdout_predictions(
        model=champion_model,
        holdout_df=holdout_df,
        feature_columns=data_cfg["features"],
        output_path=output_path,
        id_col=data_cfg["id_column"],
    )

    # Re-read from disk and validate saved CSV
    disk_df = pd.read_csv(output_path)
    validate_predictions_df(disk_df, holdout_df, id_col=data_cfg["id_column"])

    # Independently reload champion from registry and confirm saved predictions match
    client = MlflowClient()
    champ_mv = client.get_model_version_by_alias(
        mlflow_cfg["registered_model_name"], mlflow_cfg["champion_alias"]
    )
    reloaded_champion = mlflow.sklearn.load_model(model_uri)
    reloaded_preds = np.asarray(
        reloaded_champion.predict(holdout_df[data_cfg["features"]]), dtype=float
    )
    if not np.allclose(disk_df["prediction"].to_numpy(dtype=float), reloaded_preds, rtol=0.0, atol=1e-4):
        raise ValueError("Saved predictions do not match independently reloaded champion outputs")

    summary = {
        "model_uri": model_uri,
        "champion_version": int(champ_mv.version),
        "champion_source": champ_mv.source,
        "champion_run_id": champ_mv.run_id,
        "holdout_rows": int(len(holdout_df)),
        "prediction_rows": int(len(disk_df)),
        "columns": list(disk_df.columns),
        "prediction_min": round(float(disk_df["prediction"].min()), 4),
        "prediction_mean": round(float(disk_df["prediction"].mean()), 4),
        "prediction_max": round(float(disk_df["prediction"].max()), 4),
        "output_path": str(output_path),
        "independent_reload_verified": True,
        "validated": True,
    }
    print("\n=== HOLDOUT PREDICTIONS SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print("\nFirst 5 predictions:")
    print(disk_df.head().to_string(index=False))
    return summary


if __name__ == "__main__":
    main()
