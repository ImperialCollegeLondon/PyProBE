"""Tests for the dashboard module."""

import subprocess
from unittest.mock import MagicMock, patch

import pandas as pd
import polars as pl
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from pyprobe.dashboard import (
    _dataframe_with_selections,
    _get_procedure_names,
    launch_dashboard,
)


@pytest.fixture
def mock_cell():
    """Create a mock Cell object."""
    cell = MagicMock()
    cell.info = {"id": "test_cell"}
    cell.procedure = {"test_proc": MagicMock()}
    return cell


@pytest.fixture
def cell_list(mock_cell):
    """Create a list of mock Cell objects."""
    return [mock_cell, MagicMock()]


def test_pickle_dump(cell_list, mocker):
    """Test that cell list is properly pickled with correct context handling."""
    mock_open = mocker.mock_open()

    with (
        patch("builtins.open", mock_open),
        patch("pickle.dump") as mock_dump,
        patch("subprocess.Popen"),
    ):
        launch_dashboard(cell_list)

        mock_open.assert_called_once_with("dashboard_data.pkl", "wb")
        mock_dump.assert_called_once_with(cell_list, mock_open())


def test_windows_launch(cell_list):
    """Test Windows subprocess launch."""
    with patch("platform.system", return_value="Windows"):
        with patch("subprocess.Popen") as mock_popen:
            with patch("builtins.open"):
                with patch("pickle.dump"):
                    launch_dashboard(cell_list)

                    mock_popen.assert_called_once()
                    args = mock_popen.call_args[0][0]

                    assert args[0:5] == ["cmd", "/c", "start", "/B", "streamlit"]
                    assert "dashboard.py" in args[6]
                    assert args[-3:] == [">", "nul", "2>&1"]
                    assert mock_popen.call_args[1]["shell"] is True


def test_darwin_launch(cell_list):
    """Test Darwin/MacOS subprocess launch."""
    with patch("platform.system", return_value="Darwin"):
        with patch("subprocess.Popen") as mock_popen:
            with patch("builtins.open"):
                with patch("pickle.dump"):
                    launch_dashboard(cell_list)

                    mock_popen.assert_called_once()
                    args = mock_popen.call_args[0][0]

                    assert args[0:3] == ["nohup", "streamlit", "run"]
                    assert "dashboard.py" in args[3]
                    assert mock_popen.call_args[1]["stdout"] is subprocess.DEVNULL
                    assert mock_popen.call_args[1]["stderr"] is subprocess.STDOUT


def test_other_platform(cell_list):
    """Test that no subprocess is launched on unsupported platforms."""
    with patch("platform.system", return_value="Linux"):
        with patch("subprocess.Popen") as mock_popen:
            with patch("builtins.open"):
                with patch("pickle.dump"):
                    launch_dashboard(cell_list)
                    mock_popen.assert_not_called()


def test_empty_cell_list():
    """Test handling of empty cell list."""
    with patch("builtins.open"):
        with patch("pickle.dump") as mock_dump:
            with patch("subprocess.Popen"):
                launch_dashboard([])
                mock_dump.assert_called_once_with([], mock_dump.call_args[0][1])


@pytest.fixture
def sample_df():
    """Create a sample Polars DataFrame for testing."""
    return pl.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})


def test_dataframe_with_selections(sample_df):
    """Test basic functionality of _dataframe_with_selections."""
    with patch("streamlit.sidebar.data_editor") as mock_editor:
        # Mock the returned DataFrame from st.data_editor
        mock_edited_df = pd.DataFrame(
            {"Select": [True, False, True], "col1": [1, 2, 3], "col2": ["a", "b", "c"]}
        )
        mock_editor.return_value = mock_edited_df

        result = _dataframe_with_selections(sample_df)

        assert result == [0, 2]
        mock_editor.assert_called_once()

        empty_df = pl.DataFrame()
        mock_editor.return_value = pd.DataFrame({"Select": []})
        result = _dataframe_with_selections(empty_df)
        assert result == []

        mock_edited_df = pd.DataFrame(
            {
                "Select": [False, False, False],
                "col1": [1, 2, 3],
                "col2": ["a", "b", "c"],
            }
        )
        mock_editor.return_value = mock_edited_df

        result = _dataframe_with_selections(sample_df)
        assert result == []

        mock_edited_df = pd.DataFrame(
            {"Select": [True, True, True], "col1": [1, 2, 3], "col2": ["a", "b", "c"]}
        )
        mock_editor.return_value = mock_edited_df

        result = _dataframe_with_selections(sample_df)
        assert result == [0, 1, 2]

        mock_edited_df = pd.DataFrame(
            {"Select": [False, True, False], "col1": [1, 2, 3], "col2": ["a", "b", "c"]}
        )
        mock_editor.return_value = mock_edited_df

        result = _dataframe_with_selections(sample_df)
        assert result == [1]


def test_get_procedure_names():
    """Test with empty cell list or empty indices."""
    cell_list = []
    assert _get_procedure_names(cell_list, []) == []

    # Test with single cell having one procedure.
    cell = MagicMock()
    cell.procedure = {"proc1": MagicMock()}
    assert _get_procedure_names([cell], [0]) == ["proc1"]

    """Test finding common procedures across multiple cells."""
    cell1 = MagicMock()
    cell2 = MagicMock()
    cell1.procedure = {"proc1": MagicMock(), "proc2": MagicMock()}
    cell2.procedure = {"proc1": MagicMock(), "proc3": MagicMock()}
    cell_list = [cell1, cell2]
    assert _get_procedure_names(cell_list, [0, 1]) == ["proc1"]

    """Test when cells have no common procedures."""
    cell1 = MagicMock()
    cell2 = MagicMock()
    cell1.procedure = {"proc1": MagicMock()}
    cell2.procedure = {"proc2": MagicMock()}
    cell_list = [cell1, cell2]
    assert _get_procedure_names(cell_list, [0, 1]) == []

    """Test with cells having multiple common procedures."""
    cell1 = MagicMock()
    cell2 = MagicMock()
    cell1.procedure = {"proc1": MagicMock(), "proc2": MagicMock(), "proc3": MagicMock()}
    cell2.procedure = {"proc1": MagicMock(), "proc2": MagicMock(), "proc4": MagicMock()}
    cell_list = [cell1, cell2]
    assert sorted(_get_procedure_names(cell_list, [0, 1])) == ["proc1", "proc2"]

    """Test when one cell has no procedures."""
    cell1 = MagicMock()
    cell2 = MagicMock()
    cell1.procedure = {"proc1": MagicMock()}
    cell2.procedure = {}
    cell_list = [cell1, cell2]
    assert _get_procedure_names(cell_list, [0, 1]) == []

    """Test behavior with invalid indices."""
    cell = MagicMock()
    cell.procedure = {"proc1": MagicMock()}
    with pytest.raises(IndexError):
        _get_procedure_names([cell], [1])


def filters_mini_app():
    """Mini app for testing _get_data_filters."""
    from unittest.mock import MagicMock

    import streamlit as st

    from pyprobe.dashboard import _get_data_filters

    mock_cell = MagicMock()
    mock_procedure = MagicMock()
    mock_procedure.experiment_names = ["exp1", "exp2"]
    mock_cell.procedure = {"proc1": mock_procedure}
    selected_procedure, selected_experiment_tuple, cycle_step_input = _get_data_filters(
        [mock_cell], [0], ["proc1"]
    )
    st.session_state.selected_procedure = selected_procedure
    st.session_state.selected_experiment_tuple = selected_experiment_tuple
    st.session_state.cycle_step_input = cycle_step_input


def test_get_data_filters():
    """Test Streamlit widget interactions directly using AppTest."""
    at = AppTest.from_function(filters_mini_app)
    at.run()

    # Verify widgets were created
    selectbox = at.sidebar.selectbox[0]
    assert selectbox.label == "Select a procedure"
    assert selectbox.options == ["proc1"]

    multiselect = at.sidebar.multiselect[0]
    assert multiselect.label == "Select an experiment"
    assert multiselect.options == ["exp1", "exp2"]

    text_input = at.sidebar.text_input[0]
    assert (
        text_input.label
        == 'Enter the cycle and step numbers (e.g., "cycle(1).step(2)")'
    )

    selectbox.select("proc1")
    multiselect.select("exp1")
    text_input.input("cycle(1)")
    at.run()

    # Verify return values
    assert selectbox.value == "proc1"
    assert multiselect.value == ["exp1"]
    assert text_input.value == "cycle(1)"

    assert at.session_state.selected_procedure == "proc1"
    assert at.session_state.selected_experiment_tuple == ("exp1",)
    assert at.session_state.cycle_step_input == "cycle(1)"


def graph_inputs_mini_app():
    """Mini app for testing _get_graph_inputs."""
    from unittest.mock import MagicMock

    import polars as pl
    import streamlit as st

    from pyprobe.dashboard import _get_graph_inputs

    info = pl.DataFrame(
        {"id": ["cell1", "cell2"], "temperature": [25, 30], "type": ["A", "B"]}
    )

    mock_cell = MagicMock()
    mock_cell.info = {"id": "cell1", "temperature": 25, "type": "A"}
    cell_list = [mock_cell]

    x, y, sec_y, identifier, names = _get_graph_inputs(info, cell_list, [0])
    st.session_state.x = x
    st.session_state.y = y
    st.session_state.sec_y = sec_y
    st.session_state.identifier = identifier
    st.session_state.names = names


def test_get_graph_inputs_widgets():
    """Test widget creation and interactions for _get_graph_inputs."""
    at = AppTest.from_function(graph_inputs_mini_app)
    at.run()

    # Verify all columns created
    col1, col2, col3, col4, col5 = at.columns[0:5]

    # Test filter stage selectbox
    filter_stage = col1.selectbox[0]
    assert filter_stage.label == "Filter stage"
    assert filter_stage.options == ["", "Experiment", "Cycle", "Step"]
    assert filter_stage.index == 0

    # Test x-axis selectbox
    x_axis = col2.selectbox[0]
    assert x_axis.label == "x axis"
    assert x_axis.options == [
        "Time [s]",
        "Time [min]",
        "Time [hr]",
        "Capacity [Ah]",
        "Capacity [mAh]",
        "Capacity Throughput [Ah]",
    ]
    assert x_axis.index == 0

    # Test y-axis selectbox
    y_axis = col3.selectbox[0]
    assert y_axis.label == "y axis"
    assert y_axis.options == [
        "Voltage [V]",
        "Current [A]",
        "Current [mA]",
        "Capacity [Ah]",
        "Capacity [mAh]",
    ]
    assert y_axis.index == 1

    # Test secondary y-axis selectbox
    secondary_y = col4.selectbox[0]
    assert secondary_y.label == "Secondary y axis"
    assert secondary_y.options == ["None"] + [
        "Voltage [V]",
        "Current [A]",
        "Current [mA]",
        "Capacity [Ah]",
        "Capacity [mAh]",
    ]
    assert secondary_y.index == 0

    # Test cell identifier selectbox
    cell_id = col5.selectbox[0]
    assert cell_id.label == "Legend label"
    assert set(cell_id.options) == {"id", "temperature", "type"}


def test_get_graph_inputs_interactions():
    """Test user interactions and return values."""
    at = AppTest.from_function(graph_inputs_mini_app)
    at.run()

    # Make selections
    col1, col2, col3, col4, col5 = at.columns[0:5]

    filter_stage = col1.selectbox[0]
    x_axis = col2.selectbox[0]
    y_axis = col3.selectbox[0]
    secondary_y = col4.selectbox[0]
    cell_id = col5.selectbox[0]

    filter_stage.select("Experiment")
    x_axis.select("Time [min]")
    y_axis.select("Voltage [V]")
    secondary_y.select("Current [A]")
    cell_id.select("id")
    # Verify return values
    at.run()
    assert at.session_state.x == "Experiment Time [min]"
    assert at.session_state.y == "Voltage [V]"
    assert at.session_state.sec_y == "Current [A]"
    assert at.session_state.identifier == "id"
    assert at.session_state.names == ["cell1"]
