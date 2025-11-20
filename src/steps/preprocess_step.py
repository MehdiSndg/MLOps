from src import zenml_patches  # noqa: F401
import os
import pandas as pd
from zenml import step

from src.data.preprocess import preprocess_data
from src.utils.logger import get_logger

logger = get_logger(__name__)


@step
def preprocess_step(
    df: pd.DataFrame,
    label_column: str,
    test_size: float,
    random_state: int,
    processed_path: str,
) -> str:
    """ZenML step to preprocess data and persist processed CSV."""
    processed = preprocess_data(df, label_column, test_size, random_state)
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    processed.to_csv(processed_path, index=False)
    logger.info("Processed data saved to %s", processed_path)
    return processed_path
