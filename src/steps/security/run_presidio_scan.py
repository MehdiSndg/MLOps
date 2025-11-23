"""Run a lightweight Presidio scan on processed data."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Any

import pandas as pd
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider

DEFAULT_CONFIG = {
    "nlp_engine_name": "spacy",
    "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
}


def build_analyzer() -> AnalyzerEngine:
    provider = NlpEngineProvider(nlp_configuration=DEFAULT_CONFIG)
    engine = provider.create_engine()
    return AnalyzerEngine(nlp_engine=engine, supported_languages=["en"])


def analyze_dataframe(df: pd.DataFrame, analyzer: AnalyzerEngine, sample_size: int) -> Dict[str, Any]:
    counter: Counter[str] = Counter()
    detailed: List[Dict[str, Any]] = []
    subset = df.head(sample_size)

    for idx, (_, row) in enumerate(subset.iterrows(), start=1):
        text = " | ".join(row.astype(str).tolist())
        results = analyzer.analyze(text=text, language="en")
        for res in results:
            counter[res.entity_type] += 1
            detailed.append(
                {
                    "row": idx,
                    "entity": res.entity_type,
                    "score": res.score,
                    "start": res.start,
                    "end": res.end,
                }
            )

    return {
        "total_rows_scanned": int(len(subset)),
        "entity_counts": dict(counter),
        "detections": detailed,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Presidio PII scan on processed data.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/processed.csv"),
        help="Path to processed CSV data.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("presidio_report.json"),
        help="Output report path.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=200,
        help="Number of rows to sample for scanning.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        print(f"[WARN] Presidio input {args.input} not found; writing empty report.")
        report = {"error": "input_not_found", "path": str(args.input)}
    else:
        df = pd.read_csv(args.input)
        analyzer = build_analyzer()
        report = analyze_dataframe(df, analyzer, args.sample_size)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[INFO] Presidio report written to {args.output}")


if __name__ == "__main__":
    main()
