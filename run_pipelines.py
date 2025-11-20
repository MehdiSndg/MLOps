import argparse
import os
import shutil
from pathlib import Path
import tempfile
from typing import Any, Dict

import mlflow
from dotenv import load_dotenv
import yaml
from autogluon.tabular import TabularPredictor

from src.data.load_data import load_raw_data
from src.data.preprocess import preprocess_data
from src.training.evaluate import evaluate_model
from src.training.train_autogluon import train_autogluon
from src.utils import mlflow_utils
from src.utils.config_loader import load_config
from src.utils.dvc_utils import setup_dvc_remote, test_s3_connection
from src.utils.logger import get_logger
from src.steps.security.owasp_checks import run_data_security_checks, run_adversarial_noise_test
from src.steps.security.model_integrity import record_model_integrity
from src.steps.security.dependency_scan import scan_dependencies
from src.steps.security.atlas_mapping import map_to_atlas
from src.steps.security.generate_security_report import generate_security_report

logger = get_logger(__name__)


def _load_mlflow_config(path: str = "src/config/mlflow_config.yaml") -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as fp:
        return yaml.safe_load(fp) or {}


def _apply_mlflow_env(config: Dict[str, Any]) -> None:
    tracking_uri = config.get("tracking_uri")
    backend = config.get("backend_uri")
    artifact_root = config.get("artifact_root")
    port = config.get("port")
    if tracking_uri:
        os.environ.setdefault("MLFLOW_TRACKING_URI", tracking_uri)
    if backend:
        os.environ.setdefault("MLFLOW_BACKEND_URI", backend)
    if artifact_root:
        os.environ.setdefault("MLFLOW_ARTIFACT_ROOT", artifact_root)
    if port:
        os.environ.setdefault("MLFLOW_PORT", str(port))


def run_data_local(config_path: str) -> str:
    cfg = load_config(config_path)
    paths = cfg["paths"]
    preprocess_cfg = cfg["preprocess"]
    training_cfg = cfg["training"]

    df = load_raw_data(paths["raw_data"])
    processed = preprocess_data(
        df,
        label_column=training_cfg["label_column"],
        test_size=preprocess_cfg["test_size"],
        random_state=preprocess_cfg["random_state"],
    )
    Path(paths["processed_data"]).parent.mkdir(parents=True, exist_ok=True)
    processed.to_csv(paths["processed_data"], index=False)
    logger.info("Saved processed data to %s", paths["processed_data"])
    return paths["processed_data"]


def run_train_local(config_path: str) -> dict:
    cfg = load_config(config_path)
    paths = cfg["paths"]
    training_cfg = cfg["training"]
    experiment_name = cfg["mlflow"]["experiment_name"]
    models_dir = paths["models_dir"]
    model_name = training_cfg.get("model_name", "autogluon_best")
    Path(models_dir).mkdir(parents=True, exist_ok=True)

    if not os.path.exists(paths["processed_data"]):
        run_data_local(config_path)

    predictor, leaderboard_path, fi_path, run_id, model_uri = train_autogluon(
        processed_path=paths["processed_data"],
        label_column=training_cfg["label_column"],
        presets=training_cfg["presets"],
        time_limit=training_cfg["time_limit"],
        eval_metric=training_cfg["eval_metric"],
        models_dir=models_dir,
        experiment_name=experiment_name,
        hyperparameters=training_cfg.get("hyperparameters"),
    )

    metrics, metrics_path = evaluate_model(
        predictor=predictor,
        processed_path=paths["processed_data"],
        label_column=training_cfg["label_column"],
        experiment_name=experiment_name,
        output_dir=models_dir,
        run_id=run_id,
    )

    mlflow_utils.log_artifacts_to_run(
        run_id=run_id, artifact_path="autogluon_model_artifacts", path=models_dir
    )
    version = mlflow_utils.register_model(model_uri=model_uri, name=model_name, run_id=run_id)
    mlflow_utils.end_run(run_id)

    artifacts = {
        "processed_path": paths["processed_data"],
        "leaderboard": leaderboard_path,
        "feature_importance": fi_path,
        "metrics": metrics_path,
        "model_dir": models_dir,
        "model_uri": model_uri,
        "model_name": model_name,
        "model_version": version,
    }
    logger.info("Training completed with artifacts: %s", artifacts)
    return artifacts


def run_deploy_local(config_path: str) -> str:
    cfg = load_config(config_path)
    paths = cfg["paths"]
    training_cfg = cfg["training"]
    registry_dir = paths["registry_dir"]
    Path(registry_dir).mkdir(parents=True, exist_ok=True)

    models_dir = Path(paths["models_dir"])
    tmp_dir = Path(tempfile.mkdtemp(prefix="deploy_models_"))
    tmp_model_dir = tmp_dir / "models"
    shutil.copytree(models_dir, tmp_model_dir, dirs_exist_ok=True)
    try:
        predictor = TabularPredictor.load(tmp_model_dir)
        best_model = predictor.get_model_best() if hasattr(predictor, "get_model_best") else predictor.model_best
        predictor.save(registry_dir)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    best_path = os.path.join(registry_dir, best_model)
    logger.info("Deployed best model %s to %s", best_model, best_path)
    logger.info("Registered model name: %s", training_cfg.get("model_name", "autogluon_best"))
    return best_path


def run_security_checks(config_path: str) -> str:
    cfg = load_config(config_path)
    paths = cfg["paths"]
    training_cfg = cfg["training"]
    experiment_name = cfg["mlflow"]["experiment_name"]
    processed_data = paths["processed_data"]
    if not os.path.exists(processed_data):
        run_data_local(config_path)

    security_dir = Path("artifacts/security")
    security_dir.mkdir(parents=True, exist_ok=True)

    data_results = run_data_security_checks(processed_data)
    adversarial_results = run_adversarial_noise_test(
        models_dir=paths["models_dir"],
        processed_path=processed_data,
        label_column=training_cfg["label_column"],
    )
    model_integrity = record_model_integrity(paths["models_dir"], security_dir)
    dependency_results = scan_dependencies("requirements.txt")
    atlas_summary = map_to_atlas(
        data_results.to_dict(),
        adversarial_results,
        dependency_results,
    )
    mlflow.end_run()
    report_path = generate_security_report(
        security_dir / "security_report.json",
        data_results.to_dict(),
        adversarial_results,
        model_integrity,
        dependency_results,
        atlas_summary,
    )

    run = mlflow_utils.start_run(experiment_name, run_name="security-checks")
    metrics = {
        "security_missing_values": data_results.missing_values,
        "security_anomaly_rows": data_results.anomaly_rows,
        "security_pii_matches": sum(data_results.pii_matches.values()),
        "security_adversarial_change_ratio": adversarial_results["change_ratio"],
        "security_dependency_vulns": len(dependency_results["vulnerabilities"]),
    }
    mlflow_utils.log_metrics(metrics)
    mlflow_utils.log_artifact(str(report_path))
    mlflow_utils.log_artifacts_to_run(run.info.run_id, "security", str(security_dir))
    mlflow_utils.end_run(run.info.run_id)
    logger.info("Security checks completed; report saved to %s", report_path)
    return str(report_path)


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run MLOps pipelines.")
    parser.add_argument(
        "--pipeline",
        choices=["data", "train", "deploy", "security", "all"],
        default="all",
        help="Which pipeline to run.",
    )
    args = parser.parse_args()

    mlflow_cfg = _load_mlflow_config()
    _apply_mlflow_env(mlflow_cfg)
    os.environ.setdefault("MLFLOW_TRACKING_URI", "file:./mlruns")

    if setup_dvc_remote():
        logger.info("DVC remote configured.")
    else:
        logger.warning("DVC remote not configured; skipping push/pull.")

    try:
        if not test_s3_connection():
            logger.warning("S3 connection failed; DVC operations may not work.")
    except Exception as exc:
        logger.warning("S3 connectivity check raised an exception: %s", exc)

    config_path = "src/config/config.yaml"
    processed_path = None
    if args.pipeline in ("data", "all"):
        processed_path = run_data_local(config_path)
    if args.pipeline in ("train", "all"):
        processed_path = processed_path or run_data_local(config_path)
        artifacts = run_train_local(config_path)
        logger.info("Training artifacts: %s", artifacts)
    if args.pipeline in ("deploy", "all"):
        best_path = run_deploy_local(config_path)
        logger.info("Deployment completed: %s", best_path)
    if args.pipeline in ("security", "all"):
        report = run_security_checks(config_path)
        logger.info("Security report generated: %s", report)


if __name__ == "__main__":
    main()
