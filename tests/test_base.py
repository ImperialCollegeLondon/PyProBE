import pytest
import polars as pl
from polars.testing import assert_series_equal
from pybatdata.base import Base

@pytest.fixture
def base_instance(lazyframe_fixture,
                  cycles_fixture,
                  steps_fixture,
                  step_names_fixture,):
    return Base(lazyframe_fixture,
                cycles_fixture,
                steps_fixture,
                step_names_fixture,)

def test_init(base_instance):
    assert isinstance(base_instance, Base)
    assert base_instance.cycles_idx is not None
    assert base_instance.steps_idx is not None
    assert base_instance.step_names is not None
    assert base_instance.lazyframe is not None

def test_zero_capacity(base_instance):
    base_instance._set_zero_capacity()
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

def test_mA_units(base_instance):
    base_instance._create_mA_units()
    assert_series_equal(base_instance.RawData["Current (mA)"], 
                        1000*base_instance.RawData["Current (A)"],
                        check_names=False)
    assert_series_equal(base_instance.RawData["Capacity (mAh)"], 
                        1000*base_instance.RawData["Capacity (Ah)"],
                        check_names=False)
    
def test_capacity_throughput(BreakinCycles_fixture):
    step_capacity = BreakinCycles_fixture.cycle(0).charge(0).capacity
    step_capacity_throughput = BreakinCycles_fixture.cycle(0).charge(0).RawData["Capacity Throughput (Ah)"].tail(1).to_list()[0]
    assert step_capacity == step_capacity_throughput