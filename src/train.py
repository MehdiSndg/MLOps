"""Entry point to trigger the full training pipeline via run_pipelines.py."""
from __future__ import annotations

import sys

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from run_pipelines import main as run_all


def main() -> None:
    """Execute the default pipeline (data + train + deploy + security)."""
    sys.argv = ["run_pipelines.py", "--pipeline", "all"]
    run_all()


if __name__ == "__main__":
    main()
