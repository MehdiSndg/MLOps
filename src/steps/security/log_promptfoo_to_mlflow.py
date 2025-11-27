"""Log Promptfoo security test results to MLflow."""
from __future__ import annotations

import json
import os
import requests
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv
import mlflow


def summarize(results: Dict[str, Any]) -> Dict[str, float]:
    """Extract basic pass/fail counts from promptfoo JSON output."""
    metrics: Dict[str, float] = {}
    tests = results.get("tests") or results.get("results") or []

    total = len(tests)
    passed = 0
    failed = 0
    for item in tests:
        status = item.get("status") or item.get("result") or item.get("outcome")
        if isinstance(status, str) and status.lower() in ("pass", "passed", "success"):
            passed += 1
        elif isinstance(status, str) and status.lower() in ("fail", "failed", "failure"):
            failed += 1
    if total and not (passed or failed):
        # Fallback: count heuristics if available
        for item in tests:
            if item.get("pass") is True:
                passed += 1
            elif item.get("pass") is False:
                failed += 1

    metrics["promptfoo_total"] = float(total)
    metrics["promptfoo_passed"] = float(passed)
    metrics["promptfoo_failed"] = float(failed)
    if total:
        metrics["promptfoo_pass_rate"] = float(passed) / float(total)
        metrics["promptfoo_fail_rate"] = float(failed) / float(total)
    return metrics


def main() -> None:
    load_dotenv()
    report_path = Path("promptfoo-results.json")
    if not report_path.exists():
        print(f"[WARN] promptfoo report not found: {report_path}")
        return

    with report_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    metrics = summarize(data)

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    experiment = os.getenv("MLFLOW_EXPERIMENT", "mlops_experiment")
    try:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment)
    except (requests.exceptions.RequestException, mlflow.exceptions.MlflowException):
        tracking_uri = "file:./mlruns"
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment)

    with mlflow.start_run(run_name="promptfoo_security_tests"):
        mlflow.log_param("promptfoo_report_path", str(report_path))
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(report_path), artifact_path="promptfoo")
        print(f"[INFO] Logged promptfoo results to MLflow at {tracking_uri}")


if __name__ == "__main__":
    main()
