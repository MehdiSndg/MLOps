FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash git curl \
    build-essential \
    libgomp1 \
    libopenblas-dev \
    libssl-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

ENV MLFLOW_PORT=5000 \
    MLFLOW_BACKEND_URI=file:/app/mlruns \
    MLFLOW_ARTIFACT_ROOT=/app/mlruns \
    AUTO_START_MLFLOW_SERVER=true \
    PIPELINE=all

EXPOSE 5000

ENTRYPOINT ["bash", "/app/docker/entrypoint.sh"]
