# Monitoring, Retraining, Registry & Holdout Predictions — Anushka Sharma

**Student:** Anushka Sharma
**Enrollment:** S24CSEU2761
**GitHub Username:** `nush-ka`
**Course:** SCSE3040 — MLOps
**Group:** G055 | **Problem ID:** MP148
**Project:** Tea Leaf Yield Prediction (Patna Tea Growers' Estate)

---

## Summary & Why Corrected Retraining Was Required

Following PR #2, Ravish Aggarwal corrected `src/clean.py` in PR #3 (`3138d2a`) so that **all** `River Side` `worker_exp_yr` values recorded in months (`345` unique rows, `343` non-zero values) are divided by `12` — fixing the `<= 50` filter bug that previously left `12, 24, 36, 48` unconverted. That produced the corrected Version 3 baseline (`models/tea_leaf_v3/`, validation MAE `3.017484`, MLflow run `50a5ecb488a3438aa1d76aece128c2b2`).

Because Version 2 had been trained prior to the `River Side` cleaning fix, Stage 4 was re-executed cleanly against the **corrected Version 3 baseline**:
1. Recalculated 10-bin PSI drift using the corrected `data/processed/` dataset (`Tea-Leaf-Datasets-Corrected.zip`).
2. Re-evaluated **Version 3** across `later.csv` overall and per `section`.
3. Fixed `src/retrain.py` to explicitly select **Version 3** as the baseline (rather than oldest registered version), preserve `models/tea_leaf_v1/`, `models/tea_leaf_v2/`, and `models/tea_leaf_v3/`, train **Version 4** (`models/tea_leaf_v4/`), compare V3 vs V4 on the final 30% (`90` rows) of `later.csv`, promote V4 to `@champion`, and regenerate `predictions/holdout_predictions.csv` and `reports/findings.yaml`.

---

## How Corrected Data Affected Monitoring

- Converting all `River Side` experience values (`<= 50` included) reduced `worker_exp_yr` PSI from `0.066845` to **`0.033341`**, confirming that the apparent mild experience shift was actually a data-cleaning artifact.
- `hours_worked` remains the dominant drift driver (`PSI = 5.565345`, mean shifted from `7.01` h in Jan–Jun to `10.02` h in Jul–Aug), explaining why V3 still suffered high error on `later.csv` (`7.609520` overall MAE) and especially `River Side` (`13.656958` MAE).

---

## PSI Methodology (`src/monitor.py`)

`src/monitor.py` implements a standard reference-based 10-bin quantile PSI:
1. **Reference vs Current:** Corrected cleaned original `train.csv` (`data/processed/train_clean.csv` + `data/processed/val_clean.csv`, 1,000 rows) vs `data/raw/later.csv` (300 rows). Identifier column `record_id` is explicitly excluded.
2. **Binning:** 10 quantile boundaries are computed from reference data only (`np.quantile(ref, np.linspace(0, 1, 11))`), deduplicated with `np.unique`, and outer bounds extended to `(-inf, +inf)` to capture out-of-range values in `later.csv`.
3. **Edge Cases:** Constant columns use `[-inf, c - 1e-9, c + 1e-9, +inf]`; `NaN`s are dropped; zero-frequency bin proportions are smoothed by clipping to `eps = 1e-6`.
4. **P13 Equivalence Note:** The lab P13 notebook was unavailable in the repository and local workspace, so exact equivalence to P13 could not be verified; the standard reference-based quantile binning above is documented in code and `reports/drift_report.csv`.

---

## Corrected PSI Results (`reports/drift_report.csv`)

| Column | Role | PSI | Interpretation | Reference Rows | Current Rows |
|---|---|---:|---|---:|---:|
| `hours_worked` | feature | **5.565345** | Significant drift | 1000 | 300 |
| `kg_per_worker` | target | 2.931244 | Significant drift | 1000 | 300 |
| `bush_age_yr` | feature | 0.045195 | No significant drift | 1000 | 300 |
| `temp_c` | feature | 0.038319 | No significant drift | 1000 | 300 |
| `slope_deg` | feature | 0.034319 | No significant drift | 1000 | 300 |
| `worker_exp_yr` | feature | 0.033341 | No significant drift | 1000 | 300 |
| `rain_mm_week` | feature | 0.023348 | No significant drift | 1000 | 300 |

- **Most drifted column (overall and feature):** `hours_worked`

---

## Corrected V3 Group-Wise Performance (`reports/group_performance.csv`)

Evaluated `models/tea_leaf_v3/` on all 300 rows of `data/raw/later.csv`:

| Scope | Section | Records | MAE (kg) | RMSE (kg) | Mean Actual | Mean Predicted | Worst Served? |
|---|---|---:|---:|---:|---:|---:|:---:|
| overall | ALL | 300 | **7.609520** | 10.170568 | 42.2490 | 35.0513 | — |
| section | **River Side** | 106 | **13.656958** | 15.169144 | 47.9481 | 34.2912 | **Yes** |
| section | Upper Garden | 112 | 4.636935 | 6.267052 | 39.2491 | 35.2695 | No |
| section | Lower Garden | 82 | 3.852216 | 5.229279 | 38.9793 | 35.7360 | No |

- **Worst-performing section:** `River Side` (`13.656958` MAE)

---

## Chronological Retraining & V3 vs V4 Comparison

1. `later.csv` (300 rows, `2026-07-01` to `2026-08-31`) sorted deterministically by `(date, record_id)` and split 70:30:
   - Retraining slice (first 70%): **210** rows (`2026-07-01` to `2026-08-14`)
   - Evaluation slice (last 30%): **90** rows (`2026-08-14` to `2026-08-31`), zero `record_id` overlap
2. Combined training set: `987` corrected cleaned original `train.csv` rows (`1,000` minus `13` suspicious `kg_per_worker > 100` typos, including `val_clean.csv` June records per `PROBLEM.md`) + `210` `later_70` rows = **1,197 training rows**. Preprocessing (`SimpleImputer` + `OneHotEncoder`) fitted strictly on training data.
3. Evaluated both **V3** and **V4** on the exact same final **90** rows of `later.csv` (note: V3's original validation MAE of `3.017484` on `val_clean.csv` is a different dataset and not directly compared against V4's `later` evaluation MAE):

| Role | Registry Version | Model Name | Eval Rows | Eval MAE | Eval RMSE | Improvement % | Decision |
|---|---|---|---:|---:|---:|---:|---|
| Baseline | **v3** | `HistGradientBoosting_Robust_V3_corrected` | 90 | 7.438888 | 10.616149 | — | — |
| Retrained | **v4** | `HistGradientBoosting_v4_corrected` | 90 | **2.338811** | **3.106338** | **68.5597%** | **promote** |

- **5% Promotion Rule:** `2.338811 <= 0.95 * 7.438888 (7.066944)` → **`promote`**; `@champion` alias moved to **Version 4** (`models/tea_leaf_v4/`).

---

## MLflow Tracking & Registry (`mlflow.db`)

- **Tracking URI:** `sqlite:///mlflow.db`
- **Experiment:** `Tea_Leaf_Yield_Prediction` (`12` finished runs, every run verified to log `data_sha256 = 1771136da68a117cf0cda6fef352ba5ab237ccac8f42dabbb9487ced62b95fd`)
- **Registered Model:** `TeaLeafYieldPredictor`
  - **Version 1:** `models/tea_leaf_v1` (run `b68942e1a3674af1951462c2d388abb6`) — preserved
  - **Version 2:** `models/tea_leaf_v2` (run `6075cd17dbe14c1fba353916d5754e7d`) — preserved
  - **Version 3 (corrected baseline):** `models/tea_leaf_v3` (run `50a5ecb488a3438aa1d76aece128c2b2`, `val_mae = 3.017484`) — preserved
  - **Version 4 (`@champion`):** `models/tea_leaf_v4` (run `14f07265039e441f9032415bb8cfd48b`, `eval_mae = 2.338811`)

---

## Holdout Predictions (`predictions/holdout_predictions.csv`)

- Loaded `models:/TeaLeafYieldPredictor@champion` (Version 4) and generated predictions for all **250** rows of `data/raw/holdout_features.csv` (`R50001`–`R50250`).
- **Format:** exact `record_id,prediction` header (251 lines), zero nulls, zero duplicates, all finite (`min = 20.8515`, `mean = 41.3157`, `max = 63.5445`).
- Independently reloaded `models:/TeaLeafYieldPredictor@champion` and verified saved CSV values match within `1e-4`.

---

## Reproduction Commands

```bash
.venv/bin/python src/clean.py
.venv/bin/python -m src.data_audit
.venv/bin/python -m src.monitor
.venv/bin/python -m src.retrain
.venv/bin/python -m src.predict_holdout
.venv/bin/pip check
.venv/bin/python -m pytest tests/ -v
```

---

## Remaining Limitations

1. **Serving app still points to `models/tea_leaf_v1`:** `app.py` and `streamlit_app.py` must be updated by teammates to load `models/tea_leaf_v4` (plus `test_api.py`).
2. **Teammate-owned compliance gaps:** FastAPI Pydantic `Field` rules, `/health` `"ok"`, JSON logging, Docker tag `mp148-api:1.0`, and Kubernetes manifest updates (`mp148-api`, `mp148-svc`, 4 replicas, `/health` readiness probe) remain pending as documented in `reports/assignment_compliance_audit.md`.
3. **P13 & Google Drive:** Lab P13 was unavailable locally (equivalence noted as unverified); `dvc` CLI is not installed in `.venv` (though raw MD5s and `Tea-Leaf-Datasets-Corrected.zip` size `71,237` were verified directly against `.dvc` metadata).
