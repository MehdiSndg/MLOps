from typing import Dict, NamedTuple

from src import zenml_patches  # noqa: F401
from zenml import step

from src.steps.train_step import TrainOutputs
from src.training.evaluate import evaluate_model
from src.utils.logger import get_logger

logger = get_logger(__name__)


class EvaluateOutputs(NamedTuple):
    metrics: Dict[str, float]
    metrics_path: str


@step
def evaluate_step(
    train_outputs: TrainOutputs,
    models_dir: str,
    processed_path: str,
    label_column: str,
    experiment_name: str,
    output_dir: str,
) -> EvaluateOutputs:
    """Evaluate model and return metrics path."""
    from autogluon.tabular import TabularPredictor

    predictor = TabularPredictor.load(models_dir)
    metrics, metrics_path = evaluate_model(
        predictor=predictor,
        processed_path=processed_path,
        label_column=label_column,
        experiment_name=experiment_name,
        output_dir=output_dir,
        run_id=train_outputs.run_id,
    )
    logger.info("Evaluation step metrics: %s", metrics)
    return EvaluateOutputs(metrics, metrics_path)
