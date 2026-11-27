# Assignment Compliance Audit — MP148 (Group G055)

**Auditor:** Anushka Sharma (`S24CSEU2761`, GitHub: `nush-ka`)
**Reference Specifications:** `PROBLEM.md` and `STUDENT_HANDBOOK.md` (Group G055 / MP148 package)
**Date:** 2026-11-27

> [!IMPORTANT]
> Earlier team implementations (FastAPI, Docker, Kubernetes, logging, and data cleaning) were **not** silently rewritten during Stage 4. All discrepancies against the faculty specification are documented below with exact evidence and suggested fixes for team leader (Ravish Aggarwal) coordination.

---

## Summary Scorecard

| Category | PASS | FAIL | NOT VERIFIED |
|---|---:|---:|---:|
| General & Package Structure | 3 | 2 | 0 |
| DVC & Dataset Integrity | 4 | 0 | 0 |
| MLflow Tracking & Registry | 6 | 0 | 0 |
| FastAPI Service & Validation | 3 | 4 | 0 |
| Docker & `.dockerignore` | 2 | 1 | 0 |
| Kubernetes Manifests | 5 | 4 | 0 |
| GitHub Actions CI | 3 | 0 | 1 |
| Structured Logging | 0 | 1 | 0 |
| Monitoring, Retraining & Promotion | 4 | 0 | 0 |
| Final Submission Artifacts | 3 | 0 | 0 |
| **Total** | **33** | **12** | **1** |

---

## Detailed Requirement-by-Requirement Audit

### 1. General Requirements

| Requirement | Status | Evidence | Suggested Fix |
|---|---|---|---|
| `README.md` with group ID, problem ID, and all enrolment numbers | **FAIL** (Partial) | `README.md` has `G055` and `MP148`, but enrolment numbers were missing before Stage 4 update (added in Stage 4 README update). | Keep all four enrolment numbers (`S24CSEU1466`, `S24CSEU2760`, `S24CSEU2764`, `S24CSEU2761`) in `README.md`. |
| Pinned `requirements.txt` | **FAIL** | Lines 1–9 are pinned (`mlflow==3.18.0`, etc.), but lines 10–14 (`fastapi`, `httpx2`, `uvicorn[standard]`, `streamlit`, `pyyaml`) lack `==<version>`. | Pin exact versions (e.g. `fastapi==0.165.0`, `httpx2==2.16.2`, `uvicorn[standard]==0.54.0`, `streamlit==1.67.0`, `pyyaml==6.0.4`). |
| `params.yaml` read by code via `yaml.safe_load` | **FAIL** (Partial) | Stage 4 added `params.yaml` and reads it via `yaml.safe_load` in `src/data_audit.py`, `src/monitor.py`, `src/retrain.py`, `src/predict_holdout.py`. Stage 1–2 scripts (`src/clean.py`, `src/train.py`, `src/train_robust.py`) still hardcode constants. | Refactor `src/clean.py`, `src/train.py`, and `src/train_robust.py` to read from `params.yaml`. |
| `src` package | **PASS** | `src/__init__.py` added alongside modular scripts (`clean.py`, `train.py`, `train_robust.py`, `data_audit.py`, `monitor.py`, `retrain.py`, `predict_holdout.py`). | — |
| At least 3 automated tests + `TestClient` test | **PASS** | 73 tests pass across `tests/` (`tests/test_api.py` and `tests/test_deployment.py` use `fastapi.testclient.TestClient`). | — |

### 2. DVC & Data Integrity

| Requirement | Status | Evidence | Suggested Fix |
|---|---|---|---|
| DVC initialized | **PASS** | `.dvc/config` and `.dvcignore` present. | — |
| Original `train.csv` tracked | **PASS** | `data/raw/train.csv.dvc` tracks `train.csv` (plus `later.csv.dvc`, `holdout_features.csv.dvc`, `processed.dvc`). | — |
| Original file hash correct | **PASS** | Local MD5 matches `.dvc` metadata: `train.csv` = `e98a648a0caf409148007d04f4e7899d`, `later.csv` = `89d433fe1109f78caf631a43ac9ce648`, `holdout_features.csv` = `3cde1f8c84942e65961192aa0441cffd`. | — |
| Raw datasets not committed | **PASS** | `data/raw/.gitignore` excludes `train.csv`, `later.csv`, `holdout_features.csv`. | — |

> [!NOTE]
> **Data Cleaning Discrepancy (`src/clean.py`):** `src/clean.py` corrects `worker_exp_yr` using `(worker_exp_yr > 50) & (worker_exp_yr % 12 == 0)` rather than `section == "River Side"`. In `data/raw/train.csv`, all 345 unique `River Side` rows were recorded in months (0 to 360), so 53 unique rows with `worker_exp_yr` in `{12, 24, 36, 48}` (1–4 years) were left undivided. Left untouched in Stage 4 to preserve the original v1 baseline; coordinate with Ravish if cleaning is revised in a future release.

### 3. MLflow Tracking & Model Registry

| Requirement | Status | Evidence | Suggested Fix |
|---|---|---|---|
| Root `mlflow.db` committed | **PASS** | `.gitignore` updated to stop ignoring `mlflow.db`; `sqlite:///mlflow.db` committed at repo root. | — |
| At least 3 runs | **PASS** | 6 finished runs in experiment `Tea_Leaf_Yield_Prediction` (5 reproduced Stage 2 runs + 1 Stage 4 `HistGradientBoosting_v2` run). | — |
| `data_sha256` parameter in every run | **PASS** | Every run logs `data_sha256 = 1771136da68a117cf0cda6fef352ba5ab237ccac8f42dabbb9487ced62b95fd`. | — |
| MAE metric | **PASS** | Every run logs `mae` (plus `val_mae` / `eval_mae`). | — |
| At least 2 registered model versions | **PASS** | `TeaLeafYieldPredictor` v1 (`source=models/tea_leaf_v1`) and v2 (`source=models/tea_leaf_v2`). | — |
| `champion` alias | **PASS** | Alias `champion` assigned to v2 and verified via `mlflow.sklearn.load_model("models:/TeaLeafYieldPredictor@champion")`. | — |

### 4. FastAPI Service (`app.py`)

| Requirement | Status | Evidence | Suggested Fix |
|---|---|---|---|
| `POST /predict` endpoint | **PASS** | Implemented at `app.py:111`. | — |
| `GET /health` endpoint | **PASS** | Implemented at `app.py:94`. | — |
| Exactly 8 required prediction input fields | **PASS** | `PredictionInput` (`app.py:75-83`) has `worker_exp_yr`, `rain_mm_week`, `slope_deg`, `temp_c`, `bush_age_yr`, `hours_worked`, `flush`, `section`. | — |
| `hours_worked` > 15 rejected via Pydantic `Field` | **FAIL** | `app.py:81` defines `hours_worked: float` with no `Field(..., le=15)` constraint. | Change to `hours_worked: float = Field(..., le=15)`. |
| `rain_mm_week` < 0 rejected via Pydantic `Field` | **FAIL** | `app.py:77` defines `rain_mm_week: float` with no `Field(..., ge=0)` constraint. | Change to `rain_mm_week: float = Field(..., ge=0)`. |
| Invalid inputs return HTTP 422 | **FAIL** (Partial) | Missing/wrong-type inputs return 422, but `hours_worked > 15` and `rain_mm_week < 0` currently return 200. | Add Pydantic `Field` bounds and HTTP 422 tests in `tests/test_api.py`. |
| `GET /health` returns `{"status": "ok"}` | **FAIL** | `app.py:97` returns `{"status": "healthy", "model_loaded": True}`. `tests/test_api.py:13,83` explicitly assert `"healthy"`. | Update `app.py` and `tests/test_api.py` (with Komal/Shubham) to return `{"status": "ok", ...}`. |
| Served model updated to champion | **FAIL** | `app.py:29` and `streamlit_app.py:9` load `models/tea_leaf_v1`, whereas Stage 4 promoted `v2` (`models/tea_leaf_v2`) to `@champion`. | Coordinate with Ravish/Komal/Shubham to update `MODEL_PATH` to `models/tea_leaf_v2` (and update `test_api.py` expected value `25.28` if needed). |

### 5. Docker (`Dockerfile`, `.dockerignore`)

| Requirement | Status | Evidence | Suggested Fix |
|---|---|---|---|
| `Dockerfile` present & compliant (`python:3.12-slim`, cached `requirements.txt`, `EXPOSE 8000`, `uvicorn`) | **PASS** | Verified in `Dockerfile:4-32`. | — |
| `.dockerignore` present | **PASS** | Verified in `.dockerignore:1-71` (also excludes `mlflow.db`). | — |
| Required image name and tag `mp148-api:1.0` | **FAIL** | `.github/workflows/ci.yml`, `k8s/deployment.yaml`, `README.md`, and `docs/shubham_deployment.md` use `tea-leaf-api:1.0.0`. | Update tag to `mp148-api:1.0` across `k8s/deployment.yaml`, `.github/workflows/ci.yml`, and `README.md`. |

### 6. Kubernetes (`k8s/deployment.yaml`, `k8s/service.yaml`)

| Requirement | Status | Evidence | Suggested Fix |
|---|---|---|---|
| Deployment name `mp148-api` | **FAIL** | `k8s/deployment.yaml:4` has `name: tea-leaf-api` (and `tests/test_deployment.py:80` asserts `tea-leaf-api`). | Rename Deployment to `mp148-api` and update `tests/test_deployment.py` & `ci.yml`. |
| Service name `mp148-svc` | **FAIL** | `k8s/service.yaml:4` has `name: tea-leaf-api-service` (and `tests/test_deployment.py:131` asserts it). | Rename Service to `mp148-svc` and update `tests/test_deployment.py` & `ci.yml`. |
| Exactly 4 replicas | **FAIL** | `k8s/deployment.yaml:8` has `replicas: 3` (and `tests/test_deployment.py:83` asserts `3`). | Set `replicas: 4` and update `tests/test_deployment.py`. |
| Container port `8000` | **PASS** | `k8s/deployment.yaml:27` sets `containerPort: 8000`. | — |
| Service port `80` & target port `8000` | **PASS** | `k8s/service.yaml:13-14` sets `port: 80`, `targetPort: 8000`. | — |
| Matching selectors | **PASS** | Deployment `selector.matchLabels.app` and Service `selector.app` both match `tea-leaf-api`. | — |
| Readiness probe on `/health` | **FAIL** | `k8s/deployment.yaml:50` probes `/ready` instead of `/health`. | Point `readinessProbe.httpGet.path` to `/health` (and update `tests/test_deployment.py`). |
| Liveness probe on `/health` | **PASS** | `k8s/deployment.yaml:58` probes `/health`. | — |
| Liveness waits longer than readiness | **PASS** | `livenessProbe.initialDelaySeconds` (5s) > `readinessProbe.initialDelaySeconds` (2s). | — |

### 7. CI (`.github/workflows/ci.yml`)

| Requirement | Status | Evidence | Suggested Fix |
|---|---|---|---|
| Triggers on `push` | **PASS** | `.github/workflows/ci.yml:3-5` defines `on: push:`. | — |
| Installs requirements | **PASS** | Runs `pip install -r requirements.txt`. | — |
| Executes `pytest` | **PASS** | Runs `pytest` on CI-independent tests (updated to include `test_monitoring.py`, `test_retraining.py`, `test_holdout_predictions.py`). | Note: `src/test_registered_model.py` uses a `test_` prefix; rename to `src/verify_registered_model.py` so bare `pytest` does not try to collect it. |
| Successful workflow run on GitHub | **NOT VERIFIED** | Pending push of `anushka-monitoring` and CI run observation. | Monitor GitHub Actions after pushing `anushka-monitoring`. |

### 8. Logging (`app.py`)

| Requirement | Status | Evidence | Suggested Fix |
|---|---|---|---|
| JSON logging, not plain text | **FAIL** | `app.py:19` uses plain-text formatter `"%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"`. | Replace with a `logging.Formatter` emitting single-line JSON (`json.dumps(...)`) per prediction to `stdout`. |

### 9. Monitoring, Retraining, Promotion & Submission (Stage 4)

| Requirement | Status | Evidence | Suggested Fix |
|---|---|---|---|
| PSI for every numeric column | **PASS** | `reports/drift_report.csv` covers all 7 numeric columns (`hours_worked` PSI = `5.565345`). | — |
| Worst-served group | **PASS** | `reports/group_performance.csv` identifies `River Side` (MAE = `13.728336`). | — |
| Chronological retraining (70:30) | **PASS** | `src/retrain.py` sorts `later.csv` by `(date, record_id)` into 210 train / 90 eval rows with zero overlap. | — |
| Exact 5% promotion rule | **PASS** | `v2` MAE `2.315265 <= 0.95 * 7.438206 (7.066296)` -> `promote` (`68.87%` improvement). | — |
| `reports/findings.yaml` | **PASS** | Filled using exact faculty template and validated via `yaml.safe_load`. | — |
| `predictions/holdout_predictions.csv` | **PASS** | 250 predictions (`R50001`..`R50250`), exact `record_id,prediction` header, finite values. | — |
| Public GitHub repository | **PASS** | `https://github.com/ravishcodes/Tea-Leaf-Yield-Prediction` cloned without authentication. | Verify in incognito browser before final submission. |
