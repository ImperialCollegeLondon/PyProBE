"""Test that the version is correct."""

import tomllib
from pathlib import Path

import pytest
from loguru import logger  # noqa: F401

import pyprobe


@pytest.fixture
def expected_version():
    """Fixture to load the expected version from pyproject.toml."""
    project_root = Path(__file__).parent.parent
    pyproject_path = project_root / "pyproject.toml"
    with open(pyproject_path, "rb") as f:
        pyproject_data = tomllib.load(f)
    return pyproject_data["project"]["version"]


def test_version_consistency_pyproject(expected_version):
    """Test that __version__ matches the version in pyproject.toml."""
    assert pyprobe._version._get_version() == expected_version
    assert pyprobe.__version__ == expected_version


def test_version_consistency_importlib(mocker):
    """Test that __version__ can be retrieved from package metadata."""
    mocker.patch("builtins.open", side_effect=FileNotFoundError)
    mocker.patch("pyprobe._version.version", return_value="1.0.0")
    assert pyprobe._version._get_version() == "1.0.0"


def test_version_consistency_all_fail(mocker):
    """Test that __version__ falls back to 'unknown' if both methods fail."""
    mocker.patch("builtins.open", side_effect=FileNotFoundError)
    mocker.patch("pyprobe._version.version", side_effect=ImportError)
    assert pyprobe._version._get_version() == "unknown"
