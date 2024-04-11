import pytest
import polars as pl
from polars.testing import assert_series_equal
from pybatdata.base import Base
import math

@pytest.fixture
def base_instance(lazyframe_fixture, info_fixture):
    return Base(lazyframe_fixture, info_fixture)

def test_init(base_instance):
    assert isinstance(base_instance, Base)
    assert base_instance.lazyframe is not None

# def test_zero_capacity(base_instance):
#     base_instance._set_zero_capacity()
#     assert base_instance.data["Capacity [Ah]"][0] == 0

# def test_mA_units(base_instance):
#     base_instance._create_mA_units()
#     assert_series_equal(base_instance.data["Current [mA]"], 
#                         1000*base_instance.data["Current [A]"],
#                         check_names=False)
#     assert_series_equal(base_instance.data["Capacity (mAh)"], 
#                         1000*base_instance.data["Capacity [Ah]"],
#                         check_names=False)
    
def test_capacity_throughput(BreakinCycles_fixture):
    step_capacity = BreakinCycles_fixture.cycle(0).charge(0).capacity
    step_capacity_throughput = BreakinCycles_fixture.cycle(0).charge(0).data["Capacity Throughput [Ah]"].tail(1).to_list()[0]
    assert math.isclose(step_capacity, step_capacity_throughput)

def test_get_events(base_instance):
    result = base_instance._get_events(base_instance.lazyframe)

    # Check that the result has the expected columns
    expected_columns = ['_cycle', '_cycle_reversed', '_step', '_step_reversed']
    assert all(column in result.columns for column in expected_columns)
    result = result.collect()
    # Check that '_cycle' contains only values from 0 to 13
    assert result['_cycle'].min() == 0
    assert result['_cycle'].max() == 13

    # Check that '_step' contains only values from 0 to 61
    assert result['_step'].min() == 0
    assert result['_step'].max() == 61

    # Check that '_cycle_reversed' contains only values from -14 to -1
    assert result['_cycle_reversed'].min() == -14
    assert result['_cycle_reversed'].max() == -1

    # Check that '_step_reversed' contains only values from -62 to -1
    assert result['_step_reversed'].min() == -62
    assert result['_step_reversed'].max() == -1
