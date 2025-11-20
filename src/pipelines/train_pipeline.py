import os
from src import zenml_patches  # noqa: F401
from zenml import pipeline

from src.steps.train_step import train_step
from src.steps.evaluate_step import evaluate_step
from src.steps.register_step import register_step
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


@pipeline
def train_pipeline(
    processed_path: str,
    label_column: str,
    presets: str,
    time_limit: int,
    eval_metric: str,
    models_dir: str,
    experiment_name: str,
    model_name: str,
    hyperparameters: dict | None = None,
):
    train_outputs = train_step(
        processed_path=processed_path,
        label_column=label_column,
        presets=presets,
        time_limit=time_limit,
        eval_metric=eval_metric,
        models_dir=models_dir,
        experiment_name=experiment_name,
        hyperparameters=hyperparameters,
    )
    evaluate_step(
        train_outputs=train_outputs,
        models_dir=models_dir,
        processed_path=processed_path,
        label_column=label_column,
        experiment_name=experiment_name,
        output_dir=models_dir,
    )
    register_step(
        train_outputs=train_outputs,
        models_dir=models_dir,
        model_name=model_name,
    )


def run_train_pipeline(config_path: str = "src/config/config.yaml") -> dict:
    cfg = load_config(config_path)
    paths = cfg["paths"]
    training_cfg = cfg["training"]
    experiment_name = cfg["mlflow"]["experiment_name"]
    models_dir = paths["models_dir"]
    os.makedirs(models_dir, exist_ok=True)

    train_pipeline(
        processed_path=paths["processed_data"],
        label_column=training_cfg["label_column"],
        presets=training_cfg["presets"],
        time_limit=training_cfg["time_limit"],
        eval_metric=training_cfg["eval_metric"],
        models_dir=models_dir,
        experiment_name=experiment_name,
        model_name=training_cfg.get("model_name", "autogluon_best"),
        hyperparameters=training_cfg.get("hyperparameters"),
    )
    artifacts = {
        "processed_path": paths["processed_data"],
        "leaderboard": os.path.join(models_dir, "leaderboard.csv"),
        "feature_importance": os.path.join(models_dir, "feature_importance.csv"),
        "metrics": os.path.join(models_dir, "evaluation_metrics.json"),
        "model_name": training_cfg.get("model_name", "autogluon_best"),
    }
    logger.info("Train pipeline completed with artifacts: %s", artifacts)
    return artifacts
