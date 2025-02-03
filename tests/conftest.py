"""Module containing pytest fixtures."""

import polars as pl
import pytest

from pyprobe.cell import Cell


@pytest.fixture
def info_fixture():
    """Pytest fixture for simple cell info."""
    return {"Name": "Test_Cell"}


@pytest.fixture
def lazyframe_fixture():
    """Pytest fixture for example lazyframe."""
    return pl.scan_parquet("tests/sample_data/neware/sample_data_neware_ref.parquet")


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
def cell_fixture(info_fixture):
    """Pytest fixture for example cell."""
    cell = Cell(info=info_fixture)
    cell.add_procedure(
        "Sample", "tests/sample_data/neware/", "sample_data_neware.parquet"
    )
    return cell


@pytest.fixture
def procedure_fixture(info_fixture):
    """Pytest fixture for example procedure."""
    cell = Cell(info=info_fixture)
    cell.add_procedure(
        "Sample", "tests/sample_data/neware/", "sample_data_neware.parquet"
    )
    return cell.procedure["Sample"]


@pytest.fixture(scope="function")
def BreakinCycles_fixture(procedure_fixture):
    """Pytest fixture for example cycling experiment."""
    return procedure_fixture.experiment("Break-in Cycles")
