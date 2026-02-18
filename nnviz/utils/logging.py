"""
Logging Setup
=============
Structured logging with rich formatting for console output.
"""

from __future__ import annotations

import logging
import sys


def setup_logging(level: int = logging.INFO, name: str = "nnviz") -> logging.Logger:
    """Configure structured logging for the nnviz package.

    Args:
        level: Logging level (e.g., logging.INFO, logging.DEBUG).
        name: Root logger name.

    Returns:
        The configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        return logger  # Already configured

    # Console handler with rich formatting (if available)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    try:
        from rich.logging import RichHandler
        handler = RichHandler(
            level=level,
            show_path=False,
            markup=True,
            rich_tracebacks=True,
        )
        formatter = logging.Formatter("%(message)s")
    except ImportError:
        formatter = logging.Formatter(
            "%(asctime)s │ %(levelname)-8s │ %(name)s │ %(message)s",
            datefmt="%H:%M:%S",
        )

    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger
