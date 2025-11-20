import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import mlflow
from mlflow.tracking import MlflowClient

from src.utils.logger import get_logger

logger = get_logger(__name__)


def _as_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _can_reach_uri(uri: str) -> bool:
    """Quick reachability check: set URI and list experiments."""
    try:
        mlflow.set_tracking_uri(uri)
        MlflowClient().search_experiments(max_results=1)
        return True
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("MLflow URI %s not reachable: %s", uri, exc)
        return False


def _mlflow_cli_cmd() -> list[str]:
    path = shutil.which("mlflow")
    if path:
        return [path]
    logger.warning("MLflow CLI not found; falling back to python -m mlflow.")
    return [sys.executable, "-m", "mlflow"]


def _start_local_server(port: int, backend_uri: str, artifact_root: str) -> str:
    """Start an in-process MLflow server if none is reachable."""
    log_path = os.getenv("MLFLOW_SERVER_LOG", str(Path("mlflow_server.log")))
    log_file_path = Path(log_path)
    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = _mlflow_cli_cmd() + [
        "server",
        "--backend-store-uri",
        backend_uri,
        "--default-artifact-root",
        artifact_root,
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
    ]
    wait_seconds = float(os.getenv("MLFLOW_SERVER_WAIT_SECONDS", "60"))
    sleep_step = 0.2
    max_checks = int(wait_seconds / sleep_step)
    try:
        log_file = open(log_file_path, "ab", buffering=0)
        proc = subprocess.Popen(cmd, stdout=log_file, stderr=log_file)
        local_uri = f"http://127.0.0.1:{port}"
        # Wait briefly for the server to come up
        for _ in range(max_checks):
            if _can_reach_uri(local_uri):
                logger.info("Started local MLflow server at %s", local_uri)
                return local_uri
            time.sleep(sleep_step)
        if proc.poll() is not None:
            try:
                with open(log_path, "rb") as fh:
                    tail = fh.readlines()[-20:]
                    decoded_tail = b"".join(tail).decode(errors="ignore")
                    logger.warning(
                        "MLflow server exited early with code %s. Last log lines:\n%s",
                        proc.returncode,
                        decoded_tail,
                    )
            except Exception:
                logger.warning("MLflow server exited early with code %s", proc.returncode)
        else:
            logger.warning(
                "Local MLflow server did not become ready in %ss; falling back.",
                wait_seconds,
            )
    except FileNotFoundError:
        logger.warning(
            "MLflow CLI not found when trying to start local server; falling back."
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Failed to start local MLflow server: %s", exc)
    return ""


def _choose_tracking_uri() -> str:
    port = int(os.getenv("MLFLOW_PORT", "5000"))
    backend_uri = os.getenv("MLFLOW_BACKEND_URI", "file:./mlruns")
    artifact_root = os.getenv("MLFLOW_ARTIFACT_ROOT", "./mlruns")

    env_uri = os.getenv("MLFLOW_TRACKING_URI")
    candidates: List[str] = []
    if env_uri:
        candidates.append(env_uri)
    # Common defaults when running inside Docker on Windows/Mac
    candidates.extend(
        [
            f"http://host.docker.internal:{port}",
            f"http://localhost:{port}",
        ]
    )

    for uri in candidates:
        if _can_reach_uri(uri):
            return uri

    # Optionally auto-start a local MLflow server inside the container
    if _as_bool(os.getenv("AUTO_START_MLFLOW_SERVER", "true"), True):
        local_uri = _start_local_server(
            port=port, backend_uri=backend_uri, artifact_root=artifact_root
        )
        if local_uri:
            return local_uri

    # Final fallback to file store
    fallback = backend_uri if backend_uri.startswith("file:") else "file:./mlruns"
    logger.warning("Using MLflow file backend at %s (no server reachable).", fallback)
    return fallback


def configure_mlflow(
    experiment_name: str, tracking_uri: str | None = None, artifact_location: str | None = None
) -> str:
    """Set tracking URI/experiment; ensure artifact root exists for file-based setups."""
    resolved_uri = tracking_uri
    if resolved_uri:
        if not _can_reach_uri(resolved_uri):
            logger.warning(
                "Configured MLflow URI %s not reachable; falling back to auto-detected backend.",
                resolved_uri,
            )
            resolved_uri = _choose_tracking_uri()
    else:
        resolved_uri = _choose_tracking_uri()

    mlflow.set_tracking_uri(resolved_uri)
    if artifact_location:
        if artifact_location.startswith("file:"):
            Path(artifact_location.replace("file:", "")).mkdir(parents=True, exist_ok=True)
        elif resolved_uri.startswith("file:"):
            # Artifact root is remote (e.g., S3) but backend is local file store; nothing to create locally.
            logger.debug("Artifact location %s is remote; skipping local directory creation.", artifact_location)
    mlflow.set_experiment(experiment_name)
    logger.info("MLflow configured (experiment=%s, tracking_uri=%s)", experiment_name, resolved_uri)
    return resolved_uri


def start_run(experiment_name: str, run_name: str | None = None):
    tracking_uri = configure_mlflow(
        experiment_name=experiment_name,
        tracking_uri=os.getenv("MLFLOW_TRACKING_URI"),
        artifact_location=os.getenv("MLFLOW_ARTIFACT_ROOT"),
    )
    run = mlflow.start_run(run_name=run_name)
    logger.info("MLflow run started: %s (tracking_uri=%s)", run.info.run_id, tracking_uri)
    return run


def log_params(params: Dict[str, Any]) -> None:
    mlflow.log_params(params)
    logger.info("Logged parameters to MLflow.")


def log_metrics(metrics: Dict[str, float]) -> None:
    mlflow.log_metrics(metrics)
    logger.info("Logged metrics to MLflow.")


def log_metrics_to_run(run_id: str, metrics: Dict[str, float]) -> None:
    client = MlflowClient()
    for key, value in metrics.items():
        client.log_metric(run_id=run_id, key=key, value=value)
    logger.info("Logged metrics to MLflow run %s.", run_id)


def log_artifact(path: str) -> None:
    mlflow.log_artifact(path)
    logger.info("Logged artifact: %s", path)


def log_artifacts_to_run(run_id: str, artifact_path: str, path: str) -> None:
    client = MlflowClient()
    client.log_artifacts(run_id=run_id, local_dir=path, artifact_path=artifact_path)
    logger.info("Logged artifacts from %s to MLflow run %s (dest=%s).", path, run_id, artifact_path)


def log_autogluon_model(run_id: str, predictor: Any, artifact_path: str) -> str:
    """Log AutoGluon predictor if supported; otherwise fallback to artifacts."""
    try:
        import mlflow.autogluon  # type: ignore

        mlflow.autogluon.log_model(predictor, artifact_path=artifact_path)
        model_uri = f"runs:/{run_id}/{artifact_path}"
        logger.info("Logged AutoGluon model to MLflow: %s", model_uri)
        return model_uri
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("mlflow.autogluon logging failed (%s); falling back to raw artifacts.", exc)
        predictor_path = predictor.path
        log_artifacts_to_run(run_id, artifact_path=artifact_path, path=predictor_path)
        return f"runs:/{run_id}/{artifact_path}"


def register_model(model_uri: str, name: str, run_id: str | None = None) -> str:
    client = MlflowClient()
    try:
        client.get_registered_model(name)
    except Exception:
        client.create_registered_model(name)
    version = client.create_model_version(name=name, source=model_uri, run_id=run_id)
    logger.info("Created model version: %s v%s", name, version.version)
    return version.version


def end_run(run_id: str, status: str = "FINISHED") -> None:
    """Terminate a run by id to avoid dangling active runs."""
    client = MlflowClient()
    try:
        client.set_terminated(run_id=run_id, status=status)
        logger.info("MLflow run %s marked as %s.", run_id, status)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Failed to terminate MLflow run %s: %s", run_id, exc)
