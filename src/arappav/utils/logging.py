"""Logging and seeding utilities."""

from __future__ import annotations

import logging
import random
import sys

import numpy as np
import torch


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure root logger with a consistent format.

    Args:
        level: Logging level.

    Returns:
        The root logger.
    """
    logger = logging.getLogger("arappav")
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(level)
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
