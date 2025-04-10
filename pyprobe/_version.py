"""Version information."""

import tomllib
from pathlib import Path

# Get version from pyproject.toml
pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
with open(pyproject_path, "rb") as f:
    pyproject_data = tomllib.load(f)
__version__ = pyproject_data["project"]["version"]
