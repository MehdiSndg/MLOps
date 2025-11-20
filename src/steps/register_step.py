from typing import NamedTuple

from src import zenml_patches  # noqa: F401
from zenml import step

from src.utils import mlflow_utils
from src.utils.logger import get_logger
from src.steps.train_step import TrainOutputs

logger = get_logger(__name__)


class RegisterOutputs(NamedTuple):
    model_name: str
    model_version: str


@step
def register_step(
    train_outputs: TrainOutputs, models_dir: str, model_name: str
) -> RegisterOutputs:
    """Log trained artifacts to MLflow and register a model version."""
    mlflow_utils.log_artifacts_to_run(
        run_id=train_outputs.run_id,
        artifact_path="autogluon_model_artifacts",
        path=models_dir,
    )
    version = mlflow_utils.register_model(
        model_uri=train_outputs.model_uri, name=model_name, run_id=train_outputs.run_id
    )
    mlflow_utils.end_run(train_outputs.run_id)
    logger.info("Model registered: %s version %s", model_name, version)
    return RegisterOutputs(model_name, version)
