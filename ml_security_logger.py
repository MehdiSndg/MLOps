"""Log Garak security scan metrics to MLflow."""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List

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
    parser = argparse.ArgumentParser(description="Log security metrics to MLflow.")
    parser.add_argument(
        "--reports",
        nargs="+",
        default=[
            "bandit_report.json",
            "safety_report.json",
            "presidio_report.json",
            "trivy_report.json",
        ],
        help="List of JSON report paths.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="security_controls",
        help="Name for MLflow run.",
    )
    return parser.parse_args()


def load_report(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def collect_metrics(reports: List[Path]) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for report in reports:
        if not report.exists():
            print(f"[WARN] Report missing: {report}; skipping.")
            continue
        data = load_report(report)
        _flatten(report.stem, data, metrics)
    return metrics


def main() -> None:
    args = parse_args()
    report_paths = [Path(r) for r in args.reports]

    load_dotenv()
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    experiment = os.getenv("MLFLOW_EXPERIMENT", "mlops_experiment")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)

    metrics = collect_metrics(report_paths)

    with mlflow.start_run(run_name=args.run_name):
        mlflow.log_param("security_reports", ",".join(str(p) for p in report_paths))
        if metrics:
            mlflow.log_metrics(metrics)
        for path in report_paths:
            if path.exists():
                mlflow.log_artifact(str(path), artifact_path="security_reports")
        print(f"[INFO] Logged {len(metrics)} security metrics to MLflow at {tracking_uri}.")


if __name__ == "__main__":
    main()
