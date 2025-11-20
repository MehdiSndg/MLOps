"""OWASP-inspired data and adversarial checks."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor

PII_PATTERNS = {
    "email": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "phone": re.compile(r"\+?\d{1,3}[- ]?\(?\d{2,3}\)?[- ]?\d{3}[- ]?\d{2,4}"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
}


@dataclass
class DataSecurityResult:
    pii_matches: Dict[str, int]
    missing_values: int
    anomaly_rows: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pii_matches": {k: int(v) for k, v in self.pii_matches.items()},
            "missing_values": int(self.missing_values),
            "anomaly_rows": int(self.anomaly_rows),
        }


def run_data_security_checks(processed_path: str) -> DataSecurityResult:
    """Run simple OWASP-aligned checks on processed data."""
    df = pd.read_csv(processed_path)
    pii_counts = {name: 0 for name in PII_PATTERNS}
    for col in df.columns:
        if df[col].dtype == object:
            series = df[col].astype(str)
            for name, pattern in PII_PATTERNS.items():
                pii_counts[name] += series.str.contains(pattern, na=False).sum()

    missing = int(df.isna().sum().sum())
    numeric_df = df.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        zscores = np.abs((numeric_df - numeric_df.mean()) / (numeric_df.std(ddof=0) + 1e-9))
        anomaly_rows = int((zscores > 4).any(axis=1).sum())
    else:
        anomaly_rows = 0

    return DataSecurityResult(pii_matches=pii_counts, missing_values=missing, anomaly_rows=anomaly_rows)


def run_adversarial_noise_test(
    models_dir: str,
    processed_path: str,
    label_column: str,
    sample_size: int = 32,
    noise_scale: float = 0.02,
) -> Dict[str, Any]:
    """Apply tiny Gaussian noise to numeric features and observe prediction drift."""
    predictor = TabularPredictor.load(models_dir)
    df = pd.read_csv(processed_path)
    if label_column in df:
        df = df.drop(columns=[label_column])

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    sample_df = df.sample(min(sample_size, len(df)), random_state=42).copy()
    clean_preds = predictor.predict(sample_df)

    if numeric_cols:
        noise = np.random.normal(0, noise_scale, size=sample_df[numeric_cols].shape)
        sample_df[numeric_cols] = sample_df[numeric_cols] + noise
    noisy_preds = predictor.predict(sample_df)

    changed = (clean_preds != noisy_preds).sum()
    change_ratio = float(changed) / float(len(clean_preds))
    return {
        "tested_rows": len(clean_preds),
        "changed_predictions": int(changed),
        "change_ratio": change_ratio,
        "numeric_columns_tested": len(numeric_cols),
        "clean_sample": clean_preds.head(min(5, len(clean_preds))).tolist(),
        "noisy_sample": noisy_preds.head(min(5, len(noisy_preds))).tolist(),
    }


def save_adversarial_examples(
    output_path: Path, clean_preds: pd.Series, noisy_preds: pd.Series, metadata: Dict[str, Any]
) -> None:
    payload = {
        "metadata": metadata,
        "clean_predictions": clean_preds.tolist(),
        "noisy_predictions": noisy_preds.tolist(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
