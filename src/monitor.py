"""Drift monitoring (PSI) and group-wise model evaluation for MP148 Stage 4."""

from pathlib import Path
import json
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


ROOT = Path(__file__).resolve().parent.parent
PSI_METHODOLOGY = (
    "Reference-based 10-bin quantile PSI: quantile edges fitted on reference data "
    "(np.linspace(0, 1, bins+1)), deduplicated via np.unique, outer edges extended to "
    "(-inf, +inf) to capture out-of-range values; constant columns use "
    "[-inf, c-1e-9, c+1e-9, +inf]; NaNs dropped; zero-frequency bins smoothed by "
    "clipping proportions to eps=1e-6 before sum((cur_pct - ref_pct) * ln(cur_pct / ref_pct)). "
    "Equivalence to lab P13 could not be verified because P13 was unavailable in the workspace."
)


def load_params(params_path: Path | None = None) -> dict:
    """Load configuration from params.yaml."""
    path = params_path or (ROOT / "params.yaml")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def interpret_psi(psi_value: float, moderate: float = 0.1, significant: float = 0.25) -> str:
    """Classify PSI magnitude using standard monitoring thresholds."""
    if psi_value < moderate:
        return "No significant drift"
    if psi_value < significant:
        return "Moderate drift"
    return "Significant drift"


def compute_psi(
    reference: pd.Series | np.ndarray | list[float],
    current: pd.Series | np.ndarray | list[float],
    bins: int = 10,
    eps: float = 1e-6,
) -> tuple[float, int]:
    """Compute Population Stability Index (PSI) using reference-derived quantile bins.

    Returns:
        Tuple of (psi_value, effective_bin_count).
    """
    if bins < 1:
        raise ValueError("bins must be at least 1")

    ref_arr = pd.to_numeric(pd.Series(reference), errors="coerce").dropna().to_numpy(dtype=float)
    cur_arr = pd.to_numeric(pd.Series(current), errors="coerce").dropna().to_numpy(dtype=float)

    ref_arr = ref_arr[np.isfinite(ref_arr)]
    cur_arr = cur_arr[np.isfinite(cur_arr)]

    if len(ref_arr) == 0 or len(cur_arr) == 0:
        raise ValueError("Reference and current series must contain at least one finite value")

    ref_min = float(np.min(ref_arr))
    ref_max = float(np.max(ref_arr))

    if np.isclose(ref_min, ref_max, rtol=0.0, atol=1e-12):
        center = ref_min
        edges = np.array([-np.inf, center - 1e-9, center + 1e-9, np.inf], dtype=float)
    else:
        quantiles = np.linspace(0.0, 1.0, bins + 1)
        raw_edges = np.quantile(ref_arr, quantiles)
        unique_edges = np.unique(raw_edges)
        if len(unique_edges) < 2:
            center = float(unique_edges[0])
            edges = np.array([-np.inf, center - 1e-9, center + 1e-9, np.inf], dtype=float)
        else:
            edges = unique_edges.astype(float)
            edges[0] = -np.inf
            edges[-1] = np.inf

    ref_counts, _ = np.histogram(ref_arr, bins=edges)
    cur_counts, _ = np.histogram(cur_arr, bins=edges)

    ref_pct = np.clip(ref_counts / len(ref_arr), eps, None)
    cur_pct = np.clip(cur_counts / len(cur_arr), eps, None)

    psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
    if abs(psi) < 1e-15:
        psi = 0.0
    return psi, len(edges) - 1


def compute_psi_report(
    ref_df: pd.DataFrame,
    cur_df: pd.DataFrame,
    numeric_columns: list[str],
    target_column: str = "kg_per_worker",
    id_column: str = "record_id",
    bins: int = 10,
    eps: float = 1e-6,
    moderate_threshold: float = 0.1,
    significant_threshold: float = 0.25,
) -> pd.DataFrame:
    """Compute PSI across all numeric columns (excluding record identifiers)."""
    rows = []
    for col in numeric_columns:
        if col == id_column or col not in ref_df.columns or col not in cur_df.columns:
            continue
        ref_series = ref_df[col]
        cur_series = cur_df[col]
        psi_val, bins_used = compute_psi(ref_series, cur_series, bins=bins, eps=eps)
        rows.append(
            {
                "column": col,
                "role": "target" if col == target_column else "feature",
                "psi": round(psi_val, 6),
                "drift_interpretation": interpret_psi(
                    psi_val, moderate=moderate_threshold, significant=significant_threshold
                ),
                "bins_used": bins_used,
                "reference_count": int(ref_series.notna().sum()),
                "current_count": int(cur_series.notna().sum()),
                "reference_missing": int(ref_series.isna().sum()),
                "current_missing": int(cur_series.isna().sum()),
            }
        )

    report_df = pd.DataFrame(rows)
    if report_df.empty:
        return report_df

    max_overall_col = report_df.loc[report_df["psi"].idxmax(), "column"]
    feature_df = report_df[report_df["role"] == "feature"]
    max_feature_col = (
        feature_df.loc[feature_df["psi"].idxmax(), "column"]
        if not feature_df.empty
        else max_overall_col
    )

    report_df["is_most_drifted_overall"] = report_df["column"] == max_overall_col
    report_df["is_most_drifted_feature"] = report_df["column"] == max_feature_col
    report_df["methodology"] = PSI_METHODOLOGY
    return report_df.sort_values("psi", ascending=False).reset_index(drop=True)


def evaluate_group_performance(
    model,
    df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str = "kg_per_worker",
    group_column: str = "section",
) -> pd.DataFrame:
    """Evaluate a trained model overall and per section group."""
    eval_df = df.copy()
    preds = np.asarray(model.predict(eval_df[feature_columns]), dtype=float)
    eval_df["_pred"] = preds

    overall_mae = float(mean_absolute_error(eval_df[target_column], eval_df["_pred"]))
    overall_rmse = float(mean_squared_error(eval_df[target_column], eval_df["_pred"]) ** 0.5)

    group_rows = []
    for group_val, gdf in eval_df.groupby(group_column, sort=True):
        mae = float(mean_absolute_error(gdf[target_column], gdf["_pred"]))
        rmse = float(mean_squared_error(gdf[target_column], gdf["_pred"]) ** 0.5)
        group_rows.append(
            {
                "scope": "section",
                "section": str(group_val),
                "record_count": int(len(gdf)),
                "mae": round(mae, 6),
                "rmse": round(rmse, 6),
                "mean_actual": round(float(gdf[target_column].mean()), 4),
                "mean_predicted": round(float(gdf["_pred"].mean()), 4),
            }
        )

    group_df = pd.DataFrame(group_rows)
    worst_section = group_df.loc[group_df["mae"].idxmax(), "section"] if not group_df.empty else None
    group_df["is_worst_served"] = group_df["section"] == worst_section
    group_df = group_df.sort_values("mae", ascending=False).reset_index(drop=True)

    overall_row = pd.DataFrame(
        [
            {
                "scope": "overall",
                "section": "ALL",
                "record_count": int(len(eval_df)),
                "mae": round(overall_mae, 6),
                "rmse": round(overall_rmse, 6),
                "mean_actual": round(float(eval_df[target_column].mean()), 4),
                "mean_predicted": round(float(eval_df["_pred"].mean()), 4),
                "is_worst_served": False,
            }
        ]
    )

    return pd.concat([overall_row, group_df], ignore_index=True)


def save_monitoring_chart(
    drift_df: pd.DataFrame,
    group_df: pd.DataFrame,
    output_path: Path,
    baseline_version: int = 3,
) -> None:
    """Save a two-panel chart showing PSI drift and section-wise baseline MAE."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    # Panel 1: PSI per numeric column
    d_plot = drift_df.sort_values("psi", ascending=True)
    colors = [
        "#d62728" if val >= 0.25 else ("#ff7f0e" if val >= 0.1 else "#2ca02c")
        for val in d_plot["psi"]
    ]
    axes[0].barh(d_plot["column"], d_plot["psi"], color=colors)
    axes[0].axvline(0.1, color="#ff7f0e", linestyle="--", linewidth=1, label="Moderate (0.10)")
    axes[0].axvline(0.25, color="#d62728", linestyle="--", linewidth=1, label="Significant (0.25)")
    axes[0].set_title("Feature & Target Drift (PSI: Corrected Train vs Later)")
    axes[0].set_xlabel("Population Stability Index (PSI)")
    axes[0].legend(loc="lower right", fontsize=8)

    # Panel 2: Baseline MAE by section
    g_plot = group_df[group_df["scope"] == "section"].sort_values("mae", ascending=True)
    g_colors = ["#d62728" if w else "#1f77b4" for w in g_plot["is_worst_served"]]
    axes[1].barh(g_plot["section"], g_plot["mae"], color=g_colors)
    overall_mae = float(group_df.loc[group_df["scope"] == "overall", "mae"].iloc[0])
    axes[1].axvline(overall_mae, color="#333333", linestyle="--", linewidth=1, label=f"Overall MAE ({overall_mae:.2f})")
    axes[1].set_title(f"Model v{baseline_version} Performance by Section (later.csv)")
    axes[1].set_xlabel("Mean Absolute Error (MAE, kg/worker)")
    axes[1].legend(loc="lower right", fontsize=8)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> dict:
    """Run drift monitoring and baseline group-wise evaluation on project datasets."""
    params = load_params()
    paths = params["paths"]
    data_cfg = params["data"]
    mon_cfg = params["monitoring"]

    baseline_version = int(mon_cfg["baseline_model_version"])
    baseline_model_dir = mon_cfg["baseline_model_dir"]

    train_clean = pd.read_csv(ROOT / paths["processed_train"])
    val_clean = pd.read_csv(ROOT / paths["processed_val"])
    cleaned_train = pd.concat([train_clean, val_clean], ignore_index=True)
    later_df = pd.read_csv(ROOT / paths["raw_later"])

    drift_df = compute_psi_report(
        ref_df=cleaned_train,
        cur_df=later_df,
        numeric_columns=data_cfg["numeric_columns"],
        target_column=data_cfg["target_column"],
        id_column=data_cfg["id_column"],
        bins=int(mon_cfg["psi_bins"]),
        eps=float(mon_cfg["psi_epsilon"]),
        moderate_threshold=float(mon_cfg["psi_thresholds"]["moderate"]),
        significant_threshold=float(mon_cfg["psi_thresholds"]["significant"]),
    )

    reports_dir = ROOT / paths["reports_dir"]
    reports_dir.mkdir(parents=True, exist_ok=True)
    drift_path = reports_dir / mon_cfg["drift_report_filename"]
    drift_df.to_csv(drift_path, index=False)

    baseline_model = mlflow.sklearn.load_model(str(ROOT / baseline_model_dir))
    group_df = evaluate_group_performance(
        model=baseline_model,
        df=later_df,
        feature_columns=data_cfg["features"],
        target_column=data_cfg["target_column"],
        group_column=data_cfg["group_column"],
    )
    group_path = reports_dir / mon_cfg["group_performance_filename"]
    group_df.to_csv(group_path, index=False)

    chart_path = reports_dir / mon_cfg["chart_filename"]
    save_monitoring_chart(
        drift_df,
        group_df,
        chart_path,
        baseline_version=baseline_version,
    )

    most_drifted_feature = str(
        drift_df.loc[drift_df["is_most_drifted_feature"], "column"].iloc[0]
    )
    most_drifted_overall = str(
        drift_df.loc[drift_df["is_most_drifted_overall"], "column"].iloc[0]
    )
    worst_section = str(
        group_df.loc[group_df["is_worst_served"], "section"].iloc[0]
    )
    overall_mae = float(group_df.loc[group_df["scope"] == "overall", "mae"].iloc[0])

    summary = {
        "baseline_model_version": baseline_version,
        "baseline_model_dir": baseline_model_dir,
        "reference_rows": int(len(cleaned_train)),
        "current_rows": int(len(later_df)),
        "most_drifted_feature": most_drifted_feature,
        "most_drifted_overall": most_drifted_overall,
        "overall_baseline_mae_on_later": overall_mae,
        "worst_served_section": worst_section,
        "drift_report_path": str(drift_path),
        "group_performance_path": str(group_path),
        "chart_path": str(chart_path),
    }

    print("\n=== DRIFT REPORT (PSI: CORRECTED TRAIN vs LATER) ===")
    print(drift_df.drop(columns=["methodology"]).to_string(index=False))
    print(f"\n=== GROUP PERFORMANCE (v{baseline_version} ON LATER.CSV) ===")
    print(group_df.to_string(index=False))
    print("\n=== MONITORING SUMMARY ===")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
