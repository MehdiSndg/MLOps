from src import zenml_patches  # noqa: F401
import pandas as pd
from zenml import step

from src.data.load_data import load_raw_data
from src.utils.logger import get_logger

logger = get_logger(__name__)


@step
def load_data_step(raw_path: str) -> pd.DataFrame:
    """ZenML step to load raw data."""
    df = load_raw_data(raw_path)
    logger.info("Data loaded in step with shape %s", df.shape)
    return df
