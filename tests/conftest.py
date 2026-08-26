"""Module containing pytest fixtures."""

import polars as pl
import pytest
from _pytest.logging import LogCaptureFixture
from loguru import logger

from pyprobe.cell import Cell
from pyprobe.filters import Procedure
from tests.metadata_helpers import build_metadata
from tests.readme_helpers import attach_readme


@pytest.fixture
def caplog(caplog: LogCaptureFixture):
    """Pytest fixture for capturing log messages."""
    handler_id = logger.add(
        caplog.handler,
        format="{message}",
        level=0,
        filter=lambda record: record["level"].no >= caplog.handler.level,
        enqueue=False,  # Set to 'True' if your test is spawning child processes.
    )
    yield caplog
    logger.remove(handler_id)


@pytest.fixture
def info_fixture():
    """Pytest fixture for a simple cell metadata record."""
    return build_metadata(Name="Test_Cell")


@pytest.fixture
def lazyframe_fixture():
    """Pytest fixture for example lazyframe."""
    return pl.scan_parquet("tests/sample_data/neware/sample_data_neware.bdf.parquet")


@pytest.fixture
def sample_data_neware_parquet():
    """Pytest fixture for sample neware parquet file path."""
    return "tests/sample_data/neware/sample_data_neware.bdf.parquet"


@pytest.fixture
def titles_fixture():
    """Pytest fixture for example data titles."""
    return [
        "Initial Charge",
        "Break-in Cycles",
        "Discharge Pulses",
    ]


@pytest.fixture
def steps_fixture():
    """Pytest fixture for example steps."""
    return [[1, 2, 3], [4, 5, 6, 7], [9, 10, 11, 12]]


@pytest.fixture
def cycles_fixture():
    """Pytest fixture for example cycles."""
    return [[0], [0, 1, 2, 3, 4], [4, 5, 6, 7, 8, 9, 10, 11, 12, 13]]


@pytest.fixture
def step_names_fixture():
    """Pytest fixture for example test names."""
    return [
        None,
        "Rest",
        "CCCV Chg",
        "Rest",
        "CC DChg",
        "Rest",
        "CCCV Chg",
        "Rest",
        None,
        "Rest",
        "CC DChg",
        "Rest",
        "Rest",
    ]


@pytest.fixture
def step_descriptions_fixture():
    """Pytest fixture for example step descriptions."""
    return {
        "Step": [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12],
        "Description": [
            "Rest for 4 hours",
            "Charge at 4mA until 4.2 V, Hold at 4.2 V until 0.04 A",
            "Rest for 2 hours",
            "Discharge at 4 mA until 3 V",
            "Rest for 2 hours",
            "Charge at 4 mA until 4.2 V, Hold at 4.2 V until 0.04 A",
            "Rest for 2 hours",
            "Rest for 10 seconds",
            "Discharge at 20 mA for 0.2 hours or until 3 V",
            "Rest for 30 minutes",
            "Rest for 1.5 hours",
        ],
    }


@pytest.fixture
def cell_fixture(sample_data_neware_parquet):
    """Pytest fixture for example cell."""
    c = Cell()
    proc = attach_readme(
        Procedure.load(sample_data_neware_parquet),
        "tests/sample_data/neware/README.yaml",
    )
    c.add_procedure("Sample", proc)
    return c


@pytest.fixture
def procedure_fixture():
    """Pytest fixture for example procedure."""
    return attach_readme(
        Procedure.load("tests/sample_data/neware/sample_data_neware.bdf.parquet"),
        "tests/sample_data/neware/README.yaml",
    )


@pytest.fixture(scope="function")
def BreakinCycles_fixture(procedure_fixture):
    """Pytest fixture for example cycling experiment."""
    return procedure_fixture.experiment("Break-in Cycles")


@pytest.fixture
def cycling_frame() -> pl.DataFrame:
    """Pytest fixture for a frame with the required BDF columns and three steps."""
    return pl.DataFrame(
        {
            "Test Time / s": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "Current / A": [1.0, 1.0, -1.0, -1.0, 0.0, 0.0],
            "Voltage / V": [3.7, 3.8, 3.6, 3.5, 3.55, 3.56],
            "Step ID": [1, 1, 2, 2, 3, 3],
        },
    )


@pytest.fixture
def procedure(cycling_frame: pl.DataFrame) -> Procedure:
    """Pytest fixture for a procedure over the cycling frame."""
    return Procedure.load(cycling_frame)
