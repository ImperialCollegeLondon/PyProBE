import pytest
import polars as pl
import numpy as np
from datetime import datetime, timedelta
from pybatdata.preprocessor import Preprocessor
from pybatdata.procedure import Procedure

@pytest.fixture(scope='module')
def preprocessor_fixture():
    folderpath = "tests"
    test_name = "sample_data_neware"
    cycler = "Neware"
    return Preprocessor(folderpath, test_name, cycler)

@pytest.fixture(scope='module')
def lazyframe():
    # preprocessor_fixture.write_parquet("tests/sample_data_neware/sample_data_neware.xlsx", "tests/sample_data_neware/sample_data_neware.parquet")
    return pl.scan_parquet("tests/sample_data_neware/sample_data_neware.parquet")

@pytest.fixture(scope='module')
def procedure_fixture(lazyframe, preprocessor_fixture):
    return Procedure(lazyframe, preprocessor_fixture.titles, preprocessor_fixture.cycles, preprocessor_fixture.steps, preprocessor_fixture.step_names)

@pytest.fixture(scope='module')
def BreakinCycles_fixture(procedure_fixture):
    return procedure_fixture.experiment('Break-in Cycles')