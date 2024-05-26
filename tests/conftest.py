"""Module containing pytest fixtures."""
import polars as pl
import pytest

from pyprobe.procedure import Procedure


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
    return {
        "Initial Charge": "SOC Reset",
        "Break-in Cycles": "Cycling",
        "Discharge Pulses": "Pulsing",
    }


@pytest.fixture(scope="module")
def steps_fixture():
    """Pytest fixture for example steps."""
    return [[1, 2, 3], [4, 5, 6, 7], [9, 10, 11, 12]]


@pytest.fixture(scope="module")
def cycles_fixture():
    """Pytest fixture for example cycles."""
    return [[1], [1, 2, 3, 4, 5], [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]]


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
    return Procedure(
        "tests/sample_data/neware/sample_data_neware.parquet", info_fixture
    )


@pytest.fixture(scope="module")
def BreakinCycles_fixture(procedure_fixture):
    """Pytest fixture for example cycling experiment."""
    return procedure_fixture.experiment("Break-in Cycles")


@pytest.fixture(scope="module")
def Pulsing_fixture(procedure_fixture):
    """Pytest fixture for example pulsing experiment."""
    return procedure_fixture.experiment("Discharge Pulses")
