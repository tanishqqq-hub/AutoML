"""
Centralized logging configuration for AutoML Studio.

Provides a get_logger factory that returns a consistently
formatted logger with console and rotating file handlers.
"""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from types import SimpleNamespace

from src.utils.common import ensure_dir


def get_logger(name: str, config: SimpleNamespace) -> logging.Logger:
    """
    Create or retrieve a configured logger instance.

    The logger writes logs to both console and a rotating file handler.
    Handlers are added only once to prevent duplicate log messages.

    Parameters
    ----------
    name : str
        Name of the logger (typically __name__ of the module).
    config : SimpleNamespace
        Loaded configuration object.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """

    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    level = getattr(logging, config.logging.level.upper(), logging.INFO)
    logger.setLevel(level)

    log_file = Path(config.logging.file)
    ensure_dir(log_file.parent)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
    )
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    logger.propagate = False

    return logger