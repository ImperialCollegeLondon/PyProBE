"""Module for configuring logging for the PyProBE package."""

import logging
from typing import Optional


def configure_logging(
    level: str | int = logging.WARNING,
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    log_file: Optional[str] = None,
) -> None:
    """Configure the logging level, format, and handlers for the PyProBE package.

    Args:
        level: The logging level.
        format: The logging format.
        log_file: The log file to write to. By default, no file is written.
    """
    # Create a root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Clear existing handlers
    root_logger.handlers = []

    # Create a console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter(format))
    root_logger.addHandler(console_handler)

    # Optionally create a file handler
    if log_file is not None:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(format))
        root_logger.addHandler(file_handler)
