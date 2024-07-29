"""Module containing pytest fixtures."""
import polars as pl
import pytest

from pyprobe.cell import Cell


@pytest.fixture(scope="module")
def info_fixture():
    """Pytest fixture for simple cell info."""
    return {"Name": "Test_Cell"}


@pytest.fixture(scope="module")
def lazyframe_fixture():
    """Pytest fixture for example lazyframe."""
    return pl.scan_parquet("tests/sample_data/neware/sample_data_neware_ref.parquet")


@pytest.fixture(scope="module")
def titles_fixture():
    """Pytest fixture for example data titles."""
    return [
        "Initial Charge",
        "Break-in Cycles",
        "Discharge Pulses",
    ]


@pytest.fixture(scope="module")
def steps_fixture():
    """Pytest fixture for example steps."""
    return [[1, 2, 3], [4, 5, 6, 7], [9, 10, 11, 12]]


@pytest.fixture(scope="module")
def cycles_fixture():
    """Pytest fixture for example cycles."""
    return [[0], [0, 1, 2, 3, 4], [4, 5, 6, 7, 8, 9, 10, 11, 12, 13]]


@pytest.fixture(scope="module")
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


@pytest.fixture(scope="module")
def procedure_fixture(info_fixture):
    """Pytest fixture for example procedure."""
    cell = Cell(info_fixture)
    cell.add_procedure(
        "Sample", "tests/sample_data/neware/", "sample_data_neware.parquet"
    )
    return cell.procedure["Sample"]


@pytest.fixture(scope="module")
def BreakinCycles_fixture(procedure_fixture):
    """Pytest fixture for example cycling experiment."""
    return procedure_fixture.experiment("Break-in Cycles")
