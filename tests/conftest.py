import pytest
import polars as pl
import numpy as np

@pytest.fixture(scope='module')
def dataframe():
    df = pl.read_csv('sample_data/sample_data_neware.csv')
    return df

@pytest.fixture(scope='module')
def lazyframe():
    return dataframe.lazy()

