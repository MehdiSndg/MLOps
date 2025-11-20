import json
import os

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.utils import mlflow_utils
from src.utils.logger import get_logger

logger = get_logger(__name__)


def evaluate_model(
    predictor,
    processed_path: str,
    label_column: str,
    experiment_name: str,
    output_dir: str,
    run_id: str | None = None,
):
    """Evaluate AutoGluon predictor on test split and log metrics."""
    df = pd.read_csv(processed_path)
    test_df = df[df["split"] == "test"].drop(columns=["split"])

    y_true = test_df[label_column]
    X_test = test_df.drop(columns=[label_column])
    y_pred = predictor.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }

    if run_id:
        mlflow_utils.log_metrics_to_run(run_id, metrics)
    else:
        mlflow_utils.log_metrics(metrics)

    metrics_path = os.path.join(output_dir, "evaluation_metrics.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2)

    if run_id:
        mlflow_utils.log_artifacts_to_run(run_id, artifact_path="evaluation", path=output_dir)
    else:
        mlflow_utils.log_artifact(metrics_path)

    logger.info("Evaluation metrics: %s", metrics)
    return metrics, metrics_path
