import logging
import os
from typing import Optional


def _log_level() -> int:
    """Translate LOG_LEVEL env to logging level with INFO fallback."""
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    return getattr(logging, level, logging.INFO)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a configured logger shared across the project."""
    logger = logging.getLogger(name if name else __name__)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(_log_level())
        logger.propagate = False
    else:
        logger.setLevel(_log_level())
    return logger
