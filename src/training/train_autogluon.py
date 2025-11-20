import os
from pathlib import Path

import pandas as pd
from autogluon.tabular import TabularPredictor

from src.utils import mlflow_utils
from src.utils.logger import get_logger

logger = get_logger(__name__)


def train_autogluon(
    processed_path: str,
    label_column: str,
    presets: str,
    time_limit: int,
    eval_metric: str,
    models_dir: str,
    experiment_name: str,
    hyperparameters=None,
):
    """Train AutoGluon TabularPredictor and log artifacts to MLflow."""
    df = pd.read_csv(processed_path)
    train_df = df[df["split"] == "train"].drop(columns=["split"])
    if label_column not in train_df.columns:
        raise ValueError(f"Label column {label_column} missing from processed data.")

    Path(models_dir).mkdir(parents=True, exist_ok=True)
    run = mlflow_utils.start_run(experiment_name, run_name="autogluon-train")
    run_id = run.info.run_id
    mlflow_utils.log_params(
        {
            "label_column": label_column,
            "presets": presets,
            "time_limit": time_limit,
            "eval_metric": eval_metric,
        }
    )

    predictor = TabularPredictor(
        label=label_column, eval_metric=eval_metric, path=models_dir
    ).fit(
        train_df,
        presets=presets,
        time_limit=time_limit,
        hyperparameters=hyperparameters,
    )

    leaderboard = predictor.leaderboard(train_df, silent=True)
    leaderboard_path = os.path.join(models_dir, "leaderboard.csv")
    leaderboard.to_csv(leaderboard_path, index=False)
    mlflow_utils.log_artifact(leaderboard_path)

    feature_importance = predictor.feature_importance(train_df)
    fi_path = os.path.join(models_dir, "feature_importance.csv")
    feature_importance.to_csv(fi_path)
    mlflow_utils.log_artifact(fi_path)

    best_row = leaderboard.iloc[0]
    best_score = float(best_row["score_val"])
    mlflow_utils.log_metrics_to_run(run_id, {"best_score": best_score})

    model_uri = mlflow_utils.log_autogluon_model(run_id, predictor, artifact_path="model")
    logger.info("Training completed. Best model: %s", best_row["model"])
    return predictor, leaderboard_path, fi_path, run_id, model_uri
