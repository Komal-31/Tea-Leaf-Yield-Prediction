# =============================================================
# Tea Leaf Yield Prediction API — Production Docker Image
# =============================================================
FROM python:3.12-slim AS base

# Prevent Python from writing .pyc files and enable unbuffered stdout
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# --------------- Dependencies (cached layer) ------------------
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# --------------- Application code & model ---------------------
COPY params.yaml .
COPY serving.py .
COPY app.py .
COPY streamlit_app.py .
COPY models/ models/

# --------------- Non-root user --------------------------------
RUN useradd -u 1000 --create-home appuser
USER 1000:1000

# --------------- Container health check -----------------------
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
