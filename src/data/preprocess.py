import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.logger import get_logger

logger = get_logger(__name__)


def preprocess_data(df: pd.DataFrame, label_column: str, test_size: float, random_state: int) -> pd.DataFrame:
    """Basic preprocessing: drop duplicates, fill numeric NaNs, split marker flag."""
    df = df.drop_duplicates().reset_index(drop=True)
    numeric_cols = df.select_dtypes(include="number").columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    if label_column not in df.columns:
        raise ValueError(f"Label column {label_column} not found in data.")
    df["split"] = "train"
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df[label_column]
    )
    train_df["split"] = "train"
    test_df["split"] = "test"
    processed = pd.concat([train_df, test_df]).reset_index(drop=True)
    logger.info("Processed data shape: %s", processed.shape)
    return processed
