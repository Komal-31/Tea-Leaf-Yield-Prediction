# Individual Contribution Report — Komal Bisht

**Contributor:** Komal Bisht  
**Enrollment:** S24CSEU2760  
**GitHub Username:** `Komal-31`  
**Course:** SCSE3040 — MLOps  
**Group:** G055 | **Problem ID:** MP148  
**Project:** Tea Leaf Yield Prediction  
**Contribution Branch:** `komal-final-corrections` (branched from `origin/main` at `cf11283`)  

---

## 1. Original Contributions

Prior to Stage 4 final corrections, I implemented the serving layer, interactive frontend, API/artifact testing, CI workflow, and Render deployment configuration across five commits on `main`:

| Commit | Subject | Contribution Summary |
|---|---|---|
| `55809f7` | Add automated model tests and GitHub Actions CI | Initial model artifact validation (`tests/test_model_artifact.py`) and GitHub Actions CI pipeline (`.github/workflows/ci.yml`). |
| `c22e6c5` | Add FastAPI serving and Render deployment configuration | Initial FastAPI application (`app.py`) with `/`, `/health`, `/predict`, Pydantic input schema, and `render.yaml`. |
| `fdd5f0f` | Set Render region and free deployment plan | Updated `render.yaml` with `region: singapore` and `plan: free`. |
| `77faad2` | Add Streamlit interface for tea leaf yield prediction | Built `streamlit_app.py` with cached model loading and an 8-feature form predicting yield in kg per worker. |
| `83cb1c1` | Add HTTP API tests and httpx2 dependency | Added `tests/test_api.py` (unit + `fastapi.testclient.TestClient` HTTP tests) and `httpx2` in `requirements.txt`. |

---

## 2. GitHub Contributor Attribution Investigation & Fix

### Why Earlier Commits Were Missing from GitHub Contributors

The five historical commits above (`55809f7`, `c22e6c5`, `fdd5f0f`, `77faad2`, `83cb1c1`) are present in `ravishcodes/Tea-Leaf-Yield-Prediction` (`main`), but were authored with a typo in the local Git email address:

- **Incorrect historical author email:** `komalbisht3101@gmai.com` (missing `l` in `gmail.com`)
- **Correct GitHub email:** `komalbisht3101@gmail.com`

Because GitHub links commits to a user profile via a verified email address on that account, `komalbisht3101@gmai.com` did not resolve to `Komal-31`.

### Corrective Action Taken

- Configured local repository Git identity to:
  - `user.name`: `Komal-31`
  - `user.email`: `komalbisht3101@gmail.com`
- Verified `git var GIT_AUTHOR_IDENT` reports `Komal-31 <komalbisht3101@gmail.com>`.
- Verified GitHub CLI/credential authentication is active as `Komal-31` with collaborator `push` access to `ravishcodes/Tea-Leaf-Yield-Prediction`.
- Preserved shared history without force-pushing or rewriting `main`. Note: updating local Git configuration attributes all **new** commits on `komal-final-corrections` to `Komal-31`; it does not retroactively alter the five historical commit hashes without a team-approved history rewrite.

---

## 3. Final API, Serving & Testing Corrections (`komal-final-corrections`)

### 3.1 Final Version 4 Champion Integration
- Added a centralized `serving:` configuration block to `params.yaml` specifying:
  - `model_dir: models/tea_leaf_v4`
  - `model_version: 4`
  - `registered_model_name: TeaLeafYieldPredictor`
  - `champion_alias: champion`
  - `max_hours_worked: 15`
  - `min_rain_mm_week: 0`
- Created `serving.py` to load `params.yaml`, resolve paths relative to the project root, verify alignment with `TeaLeafYieldPredictor@champion` whenever `mlflow.db` is present, and load the portable `models/tea_leaf_v4/` artifact without requiring a running MLflow server or `mlflow.db` inside the production container.
- Removed hardcoded `models/tea_leaf_v1` references in `app.py` and `streamlit_app.py`, with no silent fallback to Version 1. All four historical artifacts (`models/tea_leaf_v1/` through `models/tea_leaf_v4/`) are preserved.

### 3.2 FastAPI Input Validation & Health Corrections (`app.py`)
- Updated `PredictionInput` to enforce assignment-specific Pydantic `Field` constraints while retaining all 8 required features (`worker_exp_yr`, `rain_mm_week`, `slope_deg`, `temp_c`, `bush_age_yr`, `hours_worked`, `flush`, `section`):
  - `hours_worked: float = Field(..., ge=0, le=15)`
  - `rain_mm_week: float = Field(..., ge=0)`
- Invalid inputs (including `hours_worked > 15`, `hours_worked < 0`, `rain_mm_week < 0`, wrong types, or missing fields) return HTTP `422`.
- Updated `GET /health` to return `{"status": "ok", "model_loaded": true, "model_version": 4}` (HTTP `200`) when loaded, and HTTP `503` (`"status": "error"`, `"model_loaded": false`) if model loading fails. Preserved `GET /ready`.
- Updated `POST /predict` and `GET /` responses to include `"model_version": 4`.

### 3.3 Streamlit Frontend Update (`streamlit_app.py`)
- Updated `streamlit_app.py` to load `models/tea_leaf_v4/` via `serving.py` with `@st.cache_resource`.
- Displays the served champion version (`Model Version: v4 (@champion) — models/tea_leaf_v4`) and prediction label in `kg per worker`.
- Restricts `hours_worked` to `[0.0, 15.0]` and `rain_mm_week` to `>= 0.0`, with explicit error handling for model-loading and input validation errors.

### 3.4 Automated Tests & Shared-File Compatibility
- Updated `tests/test_api.py` and `tests/test_model_artifact.py` to verify Version 4 artifact availability, `@champion` registry alignment, container loading without `mlflow.db`, no silent fallback to v1, `GET /health` `"ok"` (and 503 when unloaded), boundary acceptance (`hours_worked=15`, `rain_mm_week=0`), HTTP 422 rejection (`hours_worked > 15`, `rain_mm_week < 0`, `hours_worked < 0`), and numerical agreement between API responses and direct `models/tea_leaf_v4/` predictions.
- **Shared-file changes documented for Shubham Mittal (`shubham-deployment`):**
  - `Dockerfile`: added `COPY params.yaml .` and `COPY serving.py .` before `app.py` so the containerized API has its centralized serving config and helper.
  - `.github/workflows/ci.yml`: updated Docker and Kubernetes smoke test assertions to check `status == 'ok'` and `model_version == 4` and added HTTP 422 smoke checks for `hours_worked=16` and `rain_mm_week=-5`.
  - Intentionally left untouched for Shubham's remaining deployment work: Docker image tag `mp148-api:1.0`, Kubernetes Deployment `mp148-api`, Kubernetes Service `mp148-svc`, `replicas: 4`, pointing `readinessProbe` to `/health`, and JSON logging in `app.py`.

---

## 4. Verification Summary

- **Dependency check (`python -m pip check`):** `No broken requirements found.`
- **CI-independent automated tests:** **88 passed** across `test_cleaning_pipeline.py`, `test_model_artifact.py`, `test_api.py`, `test_deployment.py`, `test_monitoring.py`, `test_retraining.py`, and `test_holdout_predictions.py`.
- **Full `pytest tests/ -v` run:** **89 passed** (including `tests/test_model.py::test_model_loads`), while `11` tests in `tests/test_data_quality.py` and `4` tests in `tests/test_model.py` could not execute without local `data/processed/train_clean.csv` and `data/processed/val_clean.csv` (private DVC/Google Drive datasets excluded from Git).
- **Live FastAPI verification (`uvicorn app:app --host 127.0.0.1 --port 8000`):**
  - `GET /health` -> HTTP `200`, `{"status": "ok", "model_loaded": true, "model_version": 4}`
  - `GET /ready` -> HTTP `200`, `{"status": "ready", "model_loaded": true, "model_version": 4}`
  - `GET /docs` -> HTTP `200` (Swagger UI + OpenAPI schema showing `hours_worked` `minimum: 0.0, maximum: 15.0` and `rain_mm_week` `minimum: 0.0`)
  - `POST /predict` with example input (`worker_exp_yr=12, rain_mm_week=150, slope_deg=17, temp_c=25, bush_age_yr=10, hours_worked=6, flush="monsoon", section="River Side"`) -> HTTP `200`, `{"predicted_yield_kg_per_worker": 24.39, "model_version": 4}` (matches direct `models/tea_leaf_v4` and `TeaLeafYieldPredictor@champion` prediction `24.386374395934784` rounded to 2 decimals).
  - `POST /predict` boundary (`hours_worked=15, rain_mm_week=0`) -> HTTP `200`, `{"predicted_yield_kg_per_worker": 55.7, "model_version": 4}`.
  - `POST /predict` with `hours_worked=16` -> HTTP `422`.
  - `POST /predict` with `rain_mm_week=-5` -> HTTP `422`.
- **Streamlit verification (`streamlit run streamlit_app.py` + `AppTest`):**
  - Headless server returned HTTP `200` on `/_stcore/health` and `/`.
  - Form submission with default example input produced `24.39 kg per worker` (`Model Version: v4 (@champion) — models/tea_leaf_v4`), matching FastAPI and direct Version 4 predictions with zero errors.
