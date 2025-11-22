"""Log Garak security scan metrics to MLflow."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv
import mlflow


def _is_number(val: Any) -> bool:
    return isinstance(val, (int, float)) and not math.isnan(val)


def _flatten(prefix: str, obj: Any, out: Dict[str, float]) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            _flatten(key, v, out)
    elif isinstance(obj, list):
        for idx, v in enumerate(obj):
            key = f"{prefix}[{idx}]"
            _flatten(key, v, out)
    elif _is_number(obj):
        out[prefix] = float(obj)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Log Garak metrics to MLflow.")
    parser.add_argument(
        "--report",
        type=str,
        default="garak_report.json",
        help="Path to Garak JSON report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_path = Path(args.report)
    if not report_path.exists():
        print(f"[WARN] Garak report not found at {report_path}, skipping MLflow logging.")
        return

    load_dotenv()
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    experiment = os.getenv("MLFLOW_EXPERIMENT", "mlops_experiment")

    with report_path.open("r", encoding="utf-8") as f:
        report = json.load(f)

    metrics: Dict[str, float] = {}
    _flatten("", report, metrics)

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)

    with mlflow.start_run(run_name="garak_security_scan"):
        mlflow.log_param("garak_report_path", str(report_path))
        if metrics:
            mlflow.log_metrics(metrics)
        mlflow.log_dict(report, "garak_report.json")
        print(f"[INFO] Logged {len(metrics)} metrics to MLflow at {tracking_uri}.")


if __name__ == "__main__":
    main()
