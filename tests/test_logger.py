"""Tests for the logger module."""
import logging

from pyprobe.logger import configure_logging


def test_configure_logging_default():
    """Test logging configuration with default parameters."""
    configure_logging()
    root_logger = logging.getLogger()
    assert len(root_logger.handlers) == 1
    assert isinstance(root_logger.handlers[0], logging.StreamHandler)
    assert root_logger.level == logging.WARNING


def test_configure_logging_custom_level():
    """Test logging configuration with a custom log level."""
    configure_logging(level=logging.DEBUG)
    root_logger = logging.getLogger()
    assert root_logger.level == logging.DEBUG


def test_configure_logging_custom_format():
    """Test logging configuration with a custom format."""
    custom_format = "%(levelname)s - %(message)s"
    configure_logging(format=custom_format)
    root_logger = logging.getLogger()
    assert isinstance(root_logger.handlers[0].formatter, logging.Formatter)
    assert root_logger.handlers[0].formatter._fmt == custom_format


def test_configure_logging_log_file(tmp_path):
    """Test logging configuration with a log file."""
    log_file = tmp_path / "test.log"
    configure_logging(log_file=str(log_file))
    root_logger = logging.getLogger()
    assert len(root_logger.handlers) == 2
    assert any(
        isinstance(handler, logging.FileHandler) for handler in root_logger.handlers
    )
    assert any(
        isinstance(handler, logging.StreamHandler) for handler in root_logger.handlers
    )
