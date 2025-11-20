from typing import Any, Dict, NamedTuple

from src import zenml_patches  # noqa: F401
from zenml import step

from src.training.train_autogluon import train_autogluon
from src.utils.logger import get_logger

logger = get_logger(__name__)


class TrainOutputs(NamedTuple):
    run_id: str
    model_uri: str


@step
def train_step(
    processed_path: str,
    label_column: str,
    presets: str,
    time_limit: int,
    eval_metric: str,
    models_dir: str,
    experiment_name: str,
    hyperparameters: Dict[str, Any] | None = None,
) -> TrainOutputs:
    """Train step returning predictor handle and artifact paths."""
    predictor, leaderboard_path, fi_path, run_id, model_uri = train_autogluon(
        processed_path,
        label_column,
        presets,
        time_limit,
        eval_metric,
        models_dir,
        experiment_name,
        hyperparameters=hyperparameters,
    )
    logger.info("Training step completed.")
    return TrainOutputs(run_id, model_uri)
