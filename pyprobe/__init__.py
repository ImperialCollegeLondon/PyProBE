"""The PyProBE package."""
import os

import toml

from .cell import Cell, load_archive, make_cell_list  # noqa: F401
from .dashboard import launch_dashboard  # noqa: F401
from .plot import Plot  # noqa: F401
from .result import Result  # noqa: F401

pyproject_path = os.path.join(os.path.dirname(__file__), "..", "pyproject.toml")
pyproject_data = toml.load(pyproject_path)
__version__ = pyproject_data["project"]["version"]
