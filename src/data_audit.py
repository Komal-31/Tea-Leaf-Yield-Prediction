"""Audit all seven data quality problems on the raw MP148 datasets."""

from pathlib import Path
import json
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parent.parent


def load_params(params_path: Path | None = None) -> dict:
    """Load project configuration from params.yaml."""
    path = params_path or (ROOT / "params.yaml")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def audit_seven_issues(
    train_df: pd.DataFrame,
    later_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
) -> dict:
    """Investigate all seven faculty-specified data quality issues."""
    target = "kg_per_worker"
    features = [c for c in holdout_df.columns if c not in {"record_id", "date"}]

    # 1. Leaking column: present in train/later but missing in holdout, or near-perfect target correlation
    missing_from_holdout = sorted(set(train_df.columns) - set(holdout_df.columns) - {target})
    numeric_cols = [
        c for c in features if pd.api.types.is_numeric_dtype(train_df[c])
    ]
    high_corr_cols = [
        c
        for c in numeric_cols
        if abs(train_df[c].corr(train_df[target])) > 0.95
    ]
    leaking_cols = sorted(set(missing_from_holdout) | set(high_corr_cols))
    leaking_column = {
        "present": bool(leaking_cols),
        "column": leaking_cols[0] if leaking_cols else None,
        "missing_non_target_from_holdout": missing_from_holdout,
        "high_corr_columns": high_corr_cols,
    }

    # 2. Duplicate rows (exact copies in raw train.csv)
    dup_count = int(train_df.duplicated().sum())
    duplicate_rows = {
        "present": dup_count > 0,
        "count": dup_count,
    }

    unique_df = train_df.drop_duplicates().copy()

    # 3. Unit mixup across section groups (compare group medians/maxima)
    unit_mixup = {"present": False, "column": None, "group_value": None, "factor": None}
    for col in numeric_cols:
        grp_stats = unique_df.groupby("section")[col].agg(["min", "median", "mean", "max"])
        medians = grp_stats["median"]
        other_medians = {sec: medians.drop(sec).median() for sec in medians.index}
        for sec, med in medians.items():
            other_med = other_medians[sec]
            if other_med > 0:
                ratio = med / other_med
                for factor in (10, 12, 60, 100, 1000):
                    if 0.8 * factor <= ratio <= 1.25 * factor:
                        all_divisible = bool((unique_df.loc[unique_df["section"] == sec, col] % factor == 0).all())
                        if all_divisible:
                            unit_mixup = {
                                "present": True,
                                "column": col,
                                "group_value": str(sec),
                                "factor": int(factor),
                                "affected_unique_rows": int((unique_df["section"] == sec).sum()),
                                "affected_raw_rows": int((train_df["section"] == sec).sum()),
                            }
                            break
            if unit_mixup["present"]:
                break
        if unit_mixup["present"]:
            break

    # 4. Sentinel values (impossible readings like -999, -1, 999, 9999)
    sentinel_candidates = [-999, -99, -1, 999, 9999]
    sentinel_value = {"present": False, "column": None, "value": None, "count": None}
    for col in numeric_cols:
        for val in sentinel_candidates:
            cnt_raw = int((train_df[col] == val).sum())
            cnt_unique = int((unique_df[col] == val).sum())
            if cnt_raw > 0:
                sentinel_value = {
                    "present": True,
                    "column": col,
                    "value": int(val),
                    "count": cnt_raw,
                    "unique_count": cnt_unique,
                }
                break
        if sentinel_value["present"]:
            break

    # 5. Stuck sensor: consecutive identical readings in chronological order
    sorted_df = unique_df.sort_values(["date", "record_id"]).reset_index(drop=True)
    longest_run = {"column": None, "count": 0, "start_date": None, "end_date": None, "value": None}
    for col in numeric_cols:
        grp_ids = sorted_df[col].ne(sorted_df[col].shift()).cumsum()
        runs = sorted_df.groupby(grp_ids).agg(
            value=(col, "first"),
            count=(col, "size"),
            start_date=("date", "first"),
            end_date=("date", "last"),
        )
        top = runs.sort_values("count", ascending=False).iloc[0]
        if int(top["count"]) > longest_run["count"]:
            longest_run = {
                "column": col,
                "count": int(top["count"]),
                "start_date": str(top["start_date"]),
                "end_date": str(top["end_date"]),
                "value": float(top["value"]),
            }
    # A stuck sensor lasts for a long run of consecutive records/days (e.g., >= 10)
    stuck_present = longest_run["count"] >= 10
    stuck_sensor = {
        "present": stuck_present,
        "column": longest_run["column"] if stuck_present else None,
        "start_date": longest_run["start_date"] if stuck_present else None,
        "end_date": longest_run["end_date"] if stuck_present else None,
        "max_observed_run": longest_run,
    }

    # 6. Target typos: 10x errors far above normal range
    later_max = float(later_df[target].max()) if target in later_df.columns else 100.0
    typo_mask_raw = train_df[target] > max(100.0, later_max * 1.5)
    typo_mask_unique = unique_df[target] > max(100.0, later_max * 1.5)
    typo_count = int(typo_mask_raw.sum())
    target_typos = {
        "present": typo_count > 0,
        "count": typo_count,
        "unique_count": int(typo_mask_unique.sum()),
        "min_typo_value": float(train_df.loc[typo_mask_raw, target].min()) if typo_count > 0 else None,
        "max_normal_value": float(train_df.loc[~typo_mask_raw, target].max()),
    }

    # 7. Spelling variants: compare categorical levels in train vs clean later.csv
    spelling_variants = {"present": False, "column": None, "true_levels": None}
    for col in ("flush", "section"):
        train_levels = set(unique_df[col].dropna().astype(str).unique())
        later_levels = set(later_df[col].dropna().astype(str).unique())
        if train_levels != later_levels:
            spelling_variants = {
                "present": True,
                "column": col,
                "true_levels": len(later_levels),
                "train_levels": sorted(train_levels),
                "later_levels": sorted(later_levels),
            }
            break

    return {
        "leaking_column": leaking_column,
        "unit_mixup": unit_mixup,
        "sentinel_value": sentinel_value,
        "duplicate_rows": duplicate_rows,
        "stuck_sensor": stuck_sensor,
        "target_typos": target_typos,
        "spelling_variants": spelling_variants,
    }


def main():
    params = load_params()
    train_df = pd.read_csv(ROOT / params["paths"]["raw_train"])
    later_df = pd.read_csv(ROOT / params["paths"]["raw_later"])
    holdout_df = pd.read_csv(ROOT / params["paths"]["raw_holdout"])

    results = audit_seven_issues(train_df, later_df, holdout_df)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
