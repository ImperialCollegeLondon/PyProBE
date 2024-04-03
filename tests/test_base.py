import pytest
import polars as pl
from pybatdata.base import Base

@pytest.fixture
def base_instance(lazyframe, preprocessor_fixture):
    return Base(lazyframe, preprocessor_fixture.cycles, preprocessor_fixture.steps, preprocessor_fixture.step_names)

def test_init(base_instance):
    assert isinstance(base_instance, Base)
    assert base_instance.cycles_idx is not None
    assert base_instance.steps_idx is not None
    assert base_instance.step_names is not None
    assert base_instance.lazyframe is not None

def test_zero_capacity(base_instance):
    base_instance.zero_capacity()
    assert base_instance.RawData["Capacity (Ah)"][0] == 0


def test_RawData(base_instance):
    assert base_instance._raw_data is None
    raw_data = base_instance.RawData
    assert base_instance._raw_data is not None
    assert isinstance(raw_data, pl.DataFrame)

def test_get_conditions(base_instance):
    column = "Step"
    indices = [1, 2, 3]
    conditions = base_instance.get_conditions(column, indices)
    assert isinstance(conditions, pl.Expr)

def test_flatten(base_instance):
    lst = [[1, 2, 3], [4, 5], 6]
    flat_list = base_instance.flatten(lst)
    assert flat_list == [1, 2, 3, 4, 5, 6]