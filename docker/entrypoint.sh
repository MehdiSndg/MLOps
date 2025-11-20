#!/usr/bin/env bash
set -euo pipefail

PIPELINE="${PIPELINE:-all}"
START_SERVER="${AUTO_START_MLFLOW_SERVER:-false}"

if [[ -z "${MLFLOW_TRACKING_URI:-}" ]]; then
  echo "[entrypoint] WARNING: MLFLOW_TRACKING_URI not set; defaulting to host.docker.internal:5001"
  export MLFLOW_TRACKING_URI="http://host.docker.internal:5001"
fi

if [[ "${START_SERVER}" =~ ^(true|1|yes)$ ]]; then
  echo "[entrypoint] WARNING: AUTO_START_MLFLOW_SERVER is enabled; container will log locally."
fi

python - <<'PY'
from src.utils.dvc_utils import setup_dvc_remote, test_s3_connection

if setup_dvc_remote():
    print("[entrypoint] DVC remote configured.")
else:
    print("[entrypoint] DVC remote not configured (check env vars).")

if not test_s3_connection():
    print("[entrypoint] WARNING: S3 connection failed; DVC/MLflow artifacts may not sync.")
PY

run_pipeline() {
  case "${PIPELINE}" in
    data|train|deploy|all)
      python run_pipelines.py --pipeline "${PIPELINE}"
      ;;
    zenml_data)
      python run_zenml_pipeline.py data
      ;;
    zenml_train)
      python run_zenml_pipeline.py train
      ;;
    zenml_deploy)
      python run_zenml_pipeline.py deploy
      ;;
    *)
      echo "[entrypoint] Unknown PIPELINE value: ${PIPELINE}"
      exit 1
      ;;
  esac
}

run_pipeline

if [[ -n "${SERVER_PID}" ]]; then
  kill "${SERVER_PID}" >/dev/null 2>&1 || true
  wait "${SERVER_PID}" || true
fi
