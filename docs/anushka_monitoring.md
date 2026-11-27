# Monitoring, Retraining, Registry & Holdout Predictions — Anushka Sharma

**Student:** Anushka Sharma
**Enrollment:** S24CSEU2761
**GitHub Username:** `nush-ka`
**Course:** SCSE3040 — MLOps
**Group:** G055 | **Problem ID:** MP148
**Project:** Tea Leaf Yield Prediction (Patna Tea Growers' Estate)

---

## Contribution Summary

| # | Deliverable | Status |
|---|---|---|
| 1 | Seven data quality audit script (`src/data_audit.py`) | ✅ Completed |
| 2 | PSI drift monitoring & section-wise v1 evaluation (`src/monitor.py`) | ✅ Completed |
| 3 | Chronological retraining, 5% promotion rule & MLflow registry (`src/retrain.py`) | ✅ Completed |
| 4 | Champion model holdout predictions (`src/predict_holdout.py`) | ✅ Completed |
| 5 | Filled original faculty template (`reports/findings.yaml`) | ✅ Completed |
| 6 | Automated unit tests (`tests/test_monitoring.py`, `tests/test_retraining.py`, `tests/test_holdout_predictions.py`) | ✅ Completed |
| 7 | Assignment compliance audit (`reports/assignment_compliance_audit.md`) | ✅ Completed |
| 8 | Documentation & README update | ✅ Completed |

---

## Files Created & Modified

### Created
- `params.yaml` — Centralized configuration read via `yaml.safe_load`
- `src/__init__.py` — Marks `src` as a Python package
- `src/data_audit.py` — Programmatic audit of all 7 faculty data quality problems
- `src/monitor.py` — Reference-based 10-bin PSI drift monitoring & v1 section-wise evaluation
- `src/retrain.py` — Chronological 70:30 split, v2 retraining, 5% promotion rule, MLflow tracking & registry
- `src/predict_holdout.py` — `@champion` inference and CSV schema validation on `holdout_features.csv`
- `models/tea_leaf_v2/` — Portable MLflow artifact for the promoted v2 model
- `mlflow.db` — SQLite MLflow tracking & registry database at repository root
- `reports/drift_report.csv` — PSI report for all 7 numeric columns
- `reports/group_performance.csv` — Overall and per-section v1 MAE/RMSE on `later.csv`
- `reports/drift_and_group_mae.png` — Visualization of PSI drift and v1 MAE by section
- `reports/promotion_decision.json` — Machine-readable promotion rule report
- `reports/findings.yaml` — Completed original faculty `findings.yaml` template
- `reports/assignment_compliance_audit.md` — Full audit against `PROBLEM.md` & `STUDENT_HANDBOOK.md`
- `predictions/holdout_predictions.csv` — 250 holdout predictions (`record_id,prediction`)
- `tests/test_monitoring.py`, `tests/test_retraining.py`, `tests/test_holdout_predictions.py`
- `docs/anushka_monitoring.md` — This document

### Modified
- `reports/model_comparison.csv` — Updated with v1 vs v2 chronological evaluation results on the final 30% of `later.csv`
- `.gitignore` — Stopped ignoring `mlflow.db` so the SQLite tracking/registry DB can be committed
- `.github/workflows/ci.yml` — Added `anushka-monitoring` branch trigger and Stage 4 unit test step
- `README.md` — Added Stage 4 monitoring/retraining workflow and group enrolment numbers

---

## PSI Drift Methodology

Because the original lab P13 notebook was not present in the repository or local workspace, `src/monitor.py` implements a clearly documented standard reference-based 10-bin quantile PSI:

1. **Reference vs Current:** Cleaned original `train.csv` (`data/processed/train_clean.csv` + `data/processed/val_clean.csv`, 1,000 rows) serves as the reference distribution; `data/raw/later.csv` (300 rows) serves as the current distribution.
2. **Binning:** 10 quantile bin boundaries are computed from the reference distribution only (`np.quantile(ref, np.linspace(0, 1, 11))`), deduplicated with `np.unique`, and the outer edges are extended to `(-inf, +inf)` so out-of-range values in `later.csv` are captured in the boundary bins.
3. **Edge Cases:** Constant numeric columns use `[-inf, c - 1e-9, c + 1e-9, +inf]`; `NaN` values are dropped prior to binning; zero-frequency bin proportions are smoothed via `np.clip(pct, 1e-6, None)`.
4. **Formula & Thresholds:**
   $$\text{PSI} = \sum_{i=1}^{B} (p^{\text{cur}}_i - p^{\text{ref}}_i) \ln\left(\frac{p^{\text{cur}}_i}{p^{\text{ref}}_i}\right)$$
   - `PSI < 0.10`: No significant drift
   - `0.10 <= PSI < 0.25`: Moderate drift
   - `PSI >= 0.25`: Significant drift

---

## Actual PSI Results (`reports/drift_report.csv`)

| Column | Role | PSI | Interpretation | Reference Rows | Current Rows |
|---|---|---:|---|---:|---:|
| `hours_worked` | feature | **5.565345** | Significant drift | 1000 | 300 |
| `kg_per_worker` | target | 2.931244 | Significant drift | 1000 | 300 |
| `worker_exp_yr` | feature | 0.066845 | No significant drift | 1000 | 300 |
| `bush_age_yr` | feature | 0.045195 | No significant drift | 1000 | 300 |
| `temp_c` | feature | 0.038319 | No significant drift | 1000 | 300 |
| `slope_deg` | feature | 0.034319 | No significant drift | 1000 | 300 |
| `rain_mm_week` | feature | 0.023348 | No significant drift | 1000 | 300 |

- **Most drifted numeric column (overall and feature):** `hours_worked` (mean shifted from `7.01` hours in Jan–Jun to `10.02` hours in Jul–Aug, driving target `kg_per_worker` up from ~28.3 kg normal-range mean to `42.25` kg).

---

## Group-Wise v1 Performance (`reports/group_performance.csv`)

Evaluated preserved `models/tea_leaf_v1/` on all 300 rows of `data/raw/later.csv`:

| Scope | Section | Records | MAE (kg) | RMSE (kg) | Mean Actual | Mean Predicted | Worst Served? |
|---|---|---:|---:|---:|---:|---:|:---:|
| overall | ALL | 300 | **7.663842** | 10.232528 | 42.2490 | 35.0144 | — |
| section | **River Side** | 106 | **13.728336** | 15.221534 | 47.9481 | 34.2198 | **Yes** |
| section | Upper Garden | 112 | 4.656064 | 6.408278 | 39.2491 | 35.2076 | No |
| section | Lower Garden | 82 | 3.932558 | 5.240919 | 38.9793 | 35.7778 | No |

- **Worst-served section:** `River Side` (MAE = `13.728336` kg/worker).

---

## Chronological Retraining & Promotion Decision

1. `data/raw/later.csv` (300 rows, `2026-07-01` to `2026-08-31`) was sorted ascending by `(date, record_id)` and split 70:30:
   - **Retraining slice (first 70%):** 210 rows (`2026-07-01` to `2026-08-14`)
   - **Final evaluation slice (last 30%):** 90 rows (`2026-08-14` to `2026-08-31`), zero record overlap.
2. Cleaned original `train.csv` (`train_clean.csv` + `val_clean.csv`, 1,000 rows) had 13 typo targets (`kg_per_worker > 100`) excluded, yielding 987 clean rows + 210 `later_70` rows = **1,197 training rows**.
3. `HistGradientBoostingRegressor(max_iter=200, random_state=42)` was fitted on the 1,197 combined training rows using the established preprocessing pipeline (`SimpleImputer` + `OneHotEncoder`), never touching the 90 evaluation rows.

| Model | Eval Rows | Eval MAE | Eval RMSE | Improvement % | Decision |
|---|---:|---:|---:|---:|---|
| v1 (`HistGradientBoosting_Robust`) | 90 | 7.438206 | 10.588706 | — | — |
| **v2 (`HistGradientBoosting_v2`)** | 90 | **2.315265** | **3.096960** | **68.8733%** | **promote** |

- **Promotion rule check:** `2.315265 <= 0.95 * 7.438206 (7.066296)` → **PROMOTED** (`decision: "promote"`, `@champion` → **version 2**).

---

## MLflow Tracking & Model Registry

- **Tracking URI:** `sqlite:///mlflow.db`
- **Experiment:** `Tea_Leaf_Yield_Prediction` (ID `1`)
- **Dataset hash logged in every run (`data_sha256`):** `1771136da68a117cf0cda6fef352ba5ab237ccac8f42dabbb9487ced62b95fd`

> [!NOTE]
> The team leader's original `mlflow.db` was excluded from Git and the shared Google Drive folder required sign-in, so Stage 2 baseline experiments were faithfully reproduced in `mlflow.db` with genuine new run IDs (`reproduced_from: src/train.py` / `src/train_robust.py`), while Version 1 of `TeaLeafYieldPredictor` points directly to the preserved original `models/tea_leaf_v1/` artifact (whose historical `MLmodel` `run_id` is `77cd0bd9428041ef8b2d6d23b161296e`).

| Run Name | Run ID | MAE | Notes |
|---|---|---:|---|
| `DummyRegressor_reproduced` | `71cd08ab95a74952a1203d707548f019` | 7.048344 | Reproduced from `src/train.py` |
| `RandomForest_reproduced` | `de63023edc4c4d75b15808f10499d3bf` | 8.466450 | Reproduced from `src/train.py` |
| `HistGradientBoosting_reproduced` | `5f4daadc2132473eaf0e75e2a2f2f4c8` | 13.586968 | Reproduced from `src/train.py` |
| `RandomForest_Robust_reproduced` | `940247fde3e942ed81249cfddbb47e5d` | 3.374149 | Reproduced from `src/train_robust.py` |
| `HistGradientBoosting_Robust_reproduced` | `b68942e1a3674af1951462c2d388abb6` | 3.217001 | Reproduced v1 baseline (linked to v1) |
| **`HistGradientBoosting_v2`** | **`6075cd17dbe14c1fba353916d5754e7d`** | **2.315265** | **Stage 4 retrained v2 (`@champion`)** |

- **Registered Model:** `TeaLeafYieldPredictor`
  - **Version 1:** `source=models/tea_leaf_v1`
  - **Version 2 (`@champion`):** `source=models/tea_leaf_v2`

---

## Holdout Predictions (`predictions/holdout_predictions.csv`)

- Loaded `models:/TeaLeafYieldPredictor@champion` (Version 2) and predicted all 250 rows of `data/raw/holdout_features.csv` (`R50001`–`R50250`).
- **Columns:** `record_id,prediction` (250 data rows, 251 lines with header, 0 nulls, 0 duplicates, all finite).
- **Prediction range:** min `21.2117`, mean `41.3235`, max `62.0734` kg/worker.

---

## Reproduction Commands

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python src/clean.py
.venv/bin/python -m src.data_audit
.venv/bin/python -m src.monitor
.venv/bin/python -m src.retrain
.venv/bin/python -m src.predict_holdout
.venv/bin/pip check
.venv/bin/python -m pytest tests/ -v
```

---

## Automated Test & Dependency Check Results

- `.venv/bin/pip check`: `No broken requirements found.`
- `.venv/bin/python -m pytest tests/ -v`: **73 passed in 2.39s** (`test_api.py`, `test_data_quality.py`, `test_deployment.py`, `test_holdout_predictions.py`, `test_model.py`, `test_model_artifact.py`, `test_monitoring.py`, `test_retraining.py`).

---

## Known Limitations & Remaining Blockers

1. **Serving app still points to v1:** `app.py` and `streamlit_app.py` still load `models/tea_leaf_v1/`. Since `v2` (`models/tea_leaf_v2/`) was promoted to `@champion`, Ravish/Komal/Shubham must update the served model path (and update the hardcoded `25.28` expectation in `tests/test_api.py`) if serving the new champion.
2. **Pre-existing assignment compliance gaps:** See `reports/assignment_compliance_audit.md` for API validation (`hours_worked <= 15`, `rain_mm_week >= 0`, `/health` `"ok"`), JSON logging, Docker tag (`mp148-api:1.0`), Kubernetes resource names/4 replicas/`/health` readiness probe, and unpinned lines in `requirements.txt`.
3. **Google Drive shared folder:** `1ABgoa-ansVhs4FKV9nRqg6v5tZMEicON` requires Google sign-in; raw datasets and faculty templates were verified directly against `~/Downloads/G055_MP148` with 100% matching `.dvc` MD5 hashes.
