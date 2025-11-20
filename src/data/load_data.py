import os
import pandas as pd
from sklearn.datasets import load_breast_cancer

from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_raw_data(raw_path: str) -> pd.DataFrame:
    """Load raw data from CSV; if missing, create a demo dataset."""
    if not os.path.exists(raw_path):
        logger.info("Raw data not found at %s. Creating demo dataset.", raw_path)
        data = load_breast_cancer(as_frame=True)
        df = data.frame
        df.rename(columns={"target": "target"}, inplace=True)
        os.makedirs(os.path.dirname(raw_path), exist_ok=True)
        df.to_csv(raw_path, index=False)
    df = pd.read_csv(raw_path)
    logger.info("Loaded raw data with shape %s", df.shape)
    return df
