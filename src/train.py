"""Entry point to trigger the full training pipeline via run_pipelines.py."""
from __future__ import annotations

import sys

from run_pipelines import main as run_all


def main() -> None:
    """Execute the default pipeline (data + train + deploy + security)."""
    sys.argv = ["run_pipelines.py", "--pipeline", "all"]
    run_all()


if __name__ == "__main__":
    main()
