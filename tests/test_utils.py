"""Tests for the utils module."""

import pytest

from pyprobe import utils
from pyprobe.utils import validate_timezone


def test_flatten():
    """Test flattening lists."""
    lst = [[1, 2, 3], [4, 5], 6]
    flat_list = utils.flatten_list(lst)
    assert flat_list == [1, 2, 3, 4, 5, 6]


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


class TestValidateTimezone:
    """Tests for validate_timezone."""

    def test_valid_timezones(self):
        """validate_timezone returns the string unchanged for valid IANA names."""
        assert validate_timezone("UTC") == "UTC"
        assert validate_timezone("Europe/London") == "Europe/London"
        assert validate_timezone("America/New_York") == "America/New_York"
        assert validate_timezone("Asia/Tokyo") == "Asia/Tokyo"

    def test_invalid_timezones_raise(self):
        """validate_timezone raises ValueError for unrecognised timezone strings."""
        with pytest.raises(ValueError, match="Invalid timezone"):
            validate_timezone("Invalid/Timezone")

        with pytest.raises(ValueError, match="Invalid timezone"):
            validate_timezone("NotATimezone")

        with pytest.raises(ValueError, match="Invalid timezone"):
            validate_timezone("GMT+5")


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
