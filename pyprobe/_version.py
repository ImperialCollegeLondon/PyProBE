"""Version information."""

import tomllib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def _get_version() -> str:
    """Get the version of the pyprobe."""
    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
    try:
        with open(pyproject_path, "rb") as f:
            pyproject_data = tomllib.load(f)
        return pyproject_data["project"]["version"]
    except FileNotFoundError:
        try:
            return version("PyProBE-Data")
        except (ImportError, PackageNotFoundError):
            return "unknown"


__version__ = _get_version()
