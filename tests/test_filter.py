from pybatdata.filter import Filter

def test_get_events(lazyframe_fixture):
    result = Filter._get_events(lazyframe_fixture)

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