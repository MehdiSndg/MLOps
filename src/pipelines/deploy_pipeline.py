import os
from autogluon.tabular import TabularPredictor
from src import zenml_patches  # noqa: F401
from zenml import pipeline, step

from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


@step
def load_predictor_step(models_dir: str) -> TabularPredictor:
    predictor = TabularPredictor.load(models_dir)
    logger.info("Loaded predictor from %s", models_dir)
    return predictor


@step
def save_best_model_step(predictor: TabularPredictor, registry_dir: str) -> str:
    if hasattr(predictor, "get_model_best"):
        best_model = predictor.get_model_best()
    else:
        best_model = getattr(predictor, "model_best", "best")
    os.makedirs(registry_dir, exist_ok=True)
    predictor.save(registry_dir)
    logger.info("Best model %s saved to registry: %s", best_model, registry_dir)
    return os.path.join(registry_dir, best_model)


@pipeline
def deploy_pipeline(models_dir: str, registry_dir: str):
    predictor = load_predictor_step(models_dir=models_dir)
    best_path = save_best_model_step(predictor=predictor, registry_dir=registry_dir)
    return best_path


def run_deploy_pipeline(config_path: str = "src/config/config.yaml") -> str:
    cfg = load_config(config_path)
    paths = cfg["paths"]
    flow = deploy_pipeline(
        models_dir=paths["models_dir"],
        registry_dir=paths["registry_dir"],
    )
    best_path = flow
    logger.info("Deploy pipeline exported best model to %s", best_path)
    return best_path
