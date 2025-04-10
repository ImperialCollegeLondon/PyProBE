"""Test that the version is correct."""

import tomllib
from pathlib import Path

from loguru import logger  # noqa: F401

import pyprobe


def test_version_consistency():
    """Test that __version__ matches the version in pyproject.toml."""
    # Find the project root and pyproject.toml file
    project_root = Path(__file__).parent.parent
    pyproject_path = project_root / "pyproject.toml"

    # Load and parse pyproject.toml
    with open(pyproject_path, "rb") as f:
        pyproject_data = tomllib.load(f)

    # Extract version from pyproject.toml
    expected_version = pyproject_data["project"]["version"]
    assert pyprobe.__version__ == expected_version
