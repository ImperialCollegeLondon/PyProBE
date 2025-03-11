"""Tests for the utils module."""

import inspect
from typing import Any

import pytest
from pydantic import BaseModel, Field

from pyprobe import utils
from pyprobe.utils import PyProBEValidationError, catch_pydantic_validation


def test_flatten():
    """Test flattening lists."""
    lst = [[1, 2, 3], [4, 5], 6]
    flat_list = utils.flatten_list(lst)
    assert flat_list == [1, 2, 3, 4, 5, 6]


class TestModel(BaseModel):
    """Test model for validation."""

    value: int = Field(gt=0)
    name: str = Field(min_length=3)


@catch_pydantic_validation
def sample_func(value: Any, name: Any) -> None:
    """Test docstring."""
    TestModel(value=value, name=name)


def test_catch_pydantic_validation_single_error():
    """Test single validation error handling."""
    with pytest.raises(PyProBEValidationError) as exc_info:
        sample_func(-1, "test")

    assert "value" in str(exc_info.value)
    assert "greater than 0" in str(exc_info.value).lower()


def test_catch_pydantic_validation_multiple_errors():
    """Test multiple validation errors handling."""
    with pytest.raises(PyProBEValidationError) as exc_info:
        sample_func(-1, "a")

    error_msg = str(exc_info.value)
    assert "value" in error_msg
    assert "name" in error_msg
    assert "greater than 0" in error_msg.lower()
    assert "at least 3 characters" in error_msg.lower()


def test_catch_pydantic_validation_preserves_signature():
    """Test decorator preserves function signature."""

    @catch_pydantic_validation
    def sample_func(x: int, y: str = "test") -> str:
        """Test docstring."""
        return y * x

    sig = inspect.signature(sample_func)
    assert "x" in sig.parameters
    assert "y" in sig.parameters
    assert sig.return_annotation is str
    assert sample_func.__name__ == "sample_func"
    assert sample_func.__doc__ == "Test docstring."


def test_set_log_level_default(mocker):
    """Test that set_log_level uses ERROR as default."""
    # Arrange
    mock_remove = mocker.patch("pyprobe.utils.logger.remove")
    mock_add = mocker.patch("pyprobe.utils.logger.add")

    # Act
    utils.set_log_level()

    # Assert
    mock_remove.assert_called_once()
    mock_add.assert_called_once()
    # Check the level parameter is "ERROR"
    _, kwargs = mock_add.call_args
    assert kwargs["level"] == "ERROR"


def test_set_log_level_specific_levels(mocker):
    """Test set_log_level with different valid log levels."""
    # Arrange
    mock_remove = mocker.patch("pyprobe.utils.logger.remove")
    mock_add = mocker.patch("pyprobe.utils.logger.add")
    valid_levels = ["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"]

    # Act & Assert
    for level in valid_levels:
        mock_remove.reset_mock()
        mock_add.reset_mock()

        utils.set_log_level(level)

        mock_remove.assert_called_once()
        mock_add.assert_called_once()
        _, kwargs = mock_add.call_args
        assert kwargs["level"] == level


def test_set_log_level_case_insensitive(mocker):
    """Test set_log_level handles lowercase input correctly."""
    # Arrange
    mock_remove = mocker.patch("pyprobe.utils.logger.remove")
    mock_add = mocker.patch("pyprobe.utils.logger.add")

    # Act
    utils.set_log_level("debug")

    # Assert
    mock_remove.assert_called_once()
    mock_add.assert_called_once()
    # Check the level parameter is correctly uppercased
    _, kwargs = mock_add.call_args
    assert kwargs["level"] == "DEBUG"


def test_set_log_level_format(mocker):
    """Test set_log_level uses correct format string."""
    # Arrange
    mock_remove = mocker.patch("pyprobe.utils.logger.remove")
    mock_add = mocker.patch("pyprobe.utils.logger.add")
    expected_format = (
        "<green>{time:HH:mm:ss}</green> | <level>{level}</level> | "
        "<cyan>{name}:{function}:{line}</cyan> - <level>{message}</level>"
        " | Context: {extra}"
    )

    # Act
    utils.set_log_level("INFO")

    # Assert
    mock_remove.assert_called_once()
    mock_add.assert_called_once()
    # Verify format and colorize parameters
    _, kwargs = mock_add.call_args
    assert kwargs["format"] == expected_format
    assert kwargs["colorize"] is True
