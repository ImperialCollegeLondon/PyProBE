"""Version information."""

import tomllib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

# Try to get version from pyproject.toml first (for development)
pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
try:
    with open(pyproject_path, "rb") as f:
        pyproject_data = tomllib.load(f)
    __version__ = pyproject_data["project"]["version"]
except FileNotFoundError:
    # Fallback to installed package metadata if pyproject.toml not available
    try:
        __version__ = version("PyProBE-Data")
    except (ImportError, PackageNotFoundError):
        __version__ = "unknown"
