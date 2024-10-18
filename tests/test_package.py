"""Test package-level functionality."""

import toml

import pyprobe


def test_version():
    """Test version."""
    assert pyprobe.__version__ == toml.load("pyproject.toml")["project"]["version"]
