"""Shared model-serving configuration and champion artifact loader."""

from pathlib import Path
from typing import Any

import mlflow.sklearn
import pandas as pd
import yaml
from mlflow import MlflowClient

ROOT = Path(__file__).resolve().parent


def load_serving_config(params_path: Path | None = None) -> dict[str, Any]:
    """Load centralized serving configuration from params.yaml."""
    path = params_path or (ROOT / "params.yaml")
    with open(path, encoding="utf-8") as f:
        params = yaml.safe_load(f)

    serving_cfg = params["serving"]
    data_cfg = params["data"]
    paths_cfg = params["paths"]

    model_dir = str(serving_cfg["model_dir"])
    return {
        "model_dir": model_dir,
        "model_path": (ROOT / model_dir).resolve(),
        "model_version": int(serving_cfg["model_version"]),
        "registered_model_name": str(serving_cfg["registered_model_name"]),
        "champion_alias": str(serving_cfg["champion_alias"]),
        "max_hours_worked": float(serving_cfg["max_hours_worked"]),
        "min_rain_mm_week": float(serving_cfg["min_rain_mm_week"]),
        "features": list(data_cfg["features"]),
        "mlflow_db": (ROOT / paths_cfg["mlflow_db"]).resolve(),
    }


def verify_champion_alignment(config: dict[str, Any]) -> bool:
    """Verify the configured model matches the MLflow @champion when mlflow.db is present.

    In containerized deployments where mlflow.db is excluded by .dockerignore,
    this returns False (registry check skipped) so the portable MLflow artifact
    can still be served without requiring an MLflow server or database.
    """
    mlflow_db: Path = config["mlflow_db"]
    if not mlflow_db.exists():
        return False

    client = MlflowClient(tracking_uri=f"sqlite:///{mlflow_db.as_posix()}")
    champ_mv = client.get_model_version_by_alias(
        config["registered_model_name"],
        config["champion_alias"],
    )

    expected_version = int(config["model_version"])
    actual_version = int(champ_mv.version)
    if actual_version != expected_version:
        raise RuntimeError(
            f"Serving model_version ({expected_version}) does not match "
            f"MLflow @{config['champion_alias']} version ({actual_version})."
        )

    expected_dir = Path(config["model_dir"]).as_posix().rstrip("/")
    actual_source = Path(champ_mv.source).as_posix().rstrip("/")
    if actual_source != expected_dir:
        raise RuntimeError(
            f"Serving model_dir ({expected_dir}) does not match "
            f"MLflow @{config['champion_alias']} source ({actual_source})."
        )
    return True


def load_serving_model(config: dict[str, Any] | None = None):
    """Load the configured portable champion model artifact without silent fallback."""
    cfg = config or load_serving_config()
    model_path: Path = cfg["model_path"]

    if not model_path.exists() or not (model_path / "MLmodel").exists():
        raise FileNotFoundError(
            f"Configured champion model artifact not found at {model_path}"
        )

    verify_champion_alignment(cfg)
    return mlflow.sklearn.load_model(str(model_path))


def build_feature_dataframe(payload: dict[str, Any], features: list[str]) -> pd.DataFrame:
    """Build a single-row DataFrame in exact model feature order."""
    return pd.DataFrame(
        [[payload[feature] for feature in features]],
        columns=features,
    )
