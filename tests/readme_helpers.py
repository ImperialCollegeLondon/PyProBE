"""Helpers for attaching a legacy README to a loaded procedure in tests."""

from pathlib import Path

from pyprobe.filters import Procedure
from pyprobe.readme_processor import process_readme


def attach_readme(procedure: Procedure, readme_path: str | Path) -> Procedure:
    """Attach a legacy README's experiment definitions to *procedure*.

    Args:
        procedure: The procedure to update.
        readme_path: Path to a README.yaml file.

    Returns:
        Procedure: *procedure*, with its experiment definitions populated.
    """
    procedure.readme_dict = process_readme(str(readme_path)).experiment_dict
    procedure._populate_step_descriptions()  # noqa: SLF001
    return procedure
