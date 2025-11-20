"""CLI wrapper to execute ZenML pipelines from script context."""
from __future__ import annotations

import argparse

from src.pipelines.data_pipeline import run_data_pipeline
from src.pipelines.train_pipeline import run_train_pipeline
from src.pipelines.deploy_pipeline import run_deploy_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ZenML pipelines")
    parser.add_argument(
        "pipeline",
        choices=["data", "train", "deploy"],
        help="Which ZenML pipeline to execute.",
    )
    parser.add_argument(
        "--config",
        default="src/config/config.yaml",
        help="Path to config file.",
    )
    args = parser.parse_args()

    if args.pipeline == "data":
        run_data_pipeline(config_path=args.config)
    elif args.pipeline == "train":
        run_train_pipeline(config_path=args.config)
    else:
        run_deploy_pipeline(config_path=args.config)


if __name__ == "__main__":
    main()

