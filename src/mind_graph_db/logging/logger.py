"""Logging infrastructure for Mind Graph DB."""

import logging
import sys
from typing import Optional, TextIO

from mind_graph_db.config.settings import get_settings

DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def configure_logging(
    level: Optional[str] = None,
    log_format: str = DEFAULT_FORMAT,
    stream: Optional[TextIO] = None,
) -> None:

    """Configure system-wide logger format and level.

    Args:
        level: Log level string (e.g., 'DEBUG', 'INFO', 'WARNING', 'ERROR'). If None, uses settings.
        log_format: Logging format string.
        stream: Output stream (defaults to sys.stdout).
    """
    if level is None:
        settings = get_settings()
        level = settings.log_level

    numeric_level = getattr(logging, level.upper(), logging.INFO)

    handler = logging.StreamHandler(stream or sys.stdout)
    handler.setFormatter(logging.Formatter(log_format))

    root_logger = logging.getLogger("mind_graph_db")
    root_logger.setLevel(numeric_level)

    # Avoid adding duplicate handlers
    root_logger.handlers.clear()
    root_logger.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Get a child logger under the 'mind_graph_db' namespace.

    Args:
        name: Name of the module or component.

    Returns:
        logging.Logger instance.
    """
    if name.startswith("mind_graph_db."):
        return logging.getLogger(name)
    return logging.getLogger(f"mind_graph_db.{name}")
