from src import zenml_patches  # noqa: F401
from zenml import pipeline

from src.steps.data_loader_step import load_data_step
from src.steps.preprocess_step import preprocess_step
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


@pipeline
def data_pipeline(
    raw_path: str,
    label_column: str,
    test_size: float,
    random_state: int,
    processed_path: str,
):
    df = load_data_step(raw_path=raw_path)
    return preprocess_step(
        df=df,
        label_column=label_column,
        test_size=test_size,
        random_state=random_state,
        processed_path=processed_path,
    )


def run_data_pipeline(config_path: str = "src/config/config.yaml") -> str:
    cfg = load_config(config_path)
    paths = cfg["paths"]
    preprocess_cfg = cfg["preprocess"]

    data_flow = data_pipeline(
        raw_path=paths["raw_data"],
        label_column=cfg["training"]["label_column"],
        test_size=preprocess_cfg["test_size"],
        random_state=preprocess_cfg["random_state"],
        processed_path=paths["processed_data"],
    )
    processed_path = paths["processed_data"]
    logger.info("Data pipeline completed: %s", processed_path)
    return processed_path
