"""Tests for the dashboard module."""

import subprocess
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import polars as pl
import polars.testing as pl_testing
import pytest
from streamlit.testing.v1 import AppTest

from pyprobe.dashboard import (
    _Dashboard,
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


def test_dashboard_init(mock_cell):
    """Test Dashboard initialization."""
    dashboard = _Dashboard([mock_cell])
    assert dashboard.cell_list == [mock_cell]
    assert isinstance(dashboard.info, pl.DataFrame)


def test_dashboard_get_info(mock_cell):
    """Test info DataFrame creation."""
    mock_cell.info = {"id": "test1", "value": 10}
    mock_cell2 = MagicMock()
    mock_cell2.info = {"id": "test2", "value": 20}

    dashboard = _Dashboard([mock_cell, mock_cell2])
    expected_df = pl.DataFrame(
        [{"id": "test1", "value": 10}, {"id": "test2", "value": 20}]
    )
    pl_testing.assert_frame_equal(
        dashboard.get_info([mock_cell, mock_cell2]),
        expected_df,
        check_column_order=False,
    )


def testdataframe_with_selections():
    """Test DataFrame filtering based on selections."""
    data = pl.DataFrame({"id": ["test1", "test2"], "value": [10, 20]})
    df_with_selections = _Dashboard.dataframe_with_selections(data)
    assert "Select" in df_with_selections.columns
    assert not df_with_selections["Select"].to_numpy().all()


def test_select_cell_indices(mock_cell):
    """Test cell selection functionality."""
    mock_cell.info = {"id": "test1", "value": 10}
    mock_cell2 = MagicMock()
    mock_cell2.info = {"id": "test2", "value": 20}
    dashboard = _Dashboard([mock_cell, mock_cell2])

    with patch("streamlit.sidebar.data_editor") as mock_select:
        mock_select.return_value = pd.DataFrame(
            {"id": ["test1", "test2"], "value": [10, 20], "Select": [True, False]}
        )
        result = dashboard.select_cell_indices()
        assert result == [0]


def test_dashboard_get_common_procedures():
    """Test common procedures identification."""
    # Test with single cell having one procedure.
    cell1 = MagicMock()
    cell1.procedure = {"proc1": MagicMock(), "proc2": MagicMock()}
    dashboard = _Dashboard([cell1])
    dashboard.selected_indices = [0]
    assert dashboard.get_common_procedures() == ["proc1", "proc2"]

    """Test finding common procedures across multiple cells."""
    cell1 = MagicMock()
    cell2 = MagicMock()
    cell1.procedure = {"proc1": MagicMock(), "proc2": MagicMock()}
    cell2.procedure = {"proc1": MagicMock(), "proc3": MagicMock()}
    dashboard = _Dashboard([cell1, cell2])
    dashboard.selected_indices = [0, 1]
    assert dashboard.get_common_procedures() == ["proc1"]

    """Test when cells have no common procedures."""
    cell1 = MagicMock()
    cell2 = MagicMock()
    cell1.procedure = {"proc1": MagicMock()}
    cell2.procedure = {"proc2": MagicMock()}
    dashboard = _Dashboard([cell1, cell2])
    dashboard.selected_indices = [0, 1]
    assert dashboard.get_common_procedures() == []

    """Test with cells having multiple common procedures."""
    cell1 = MagicMock()
    cell2 = MagicMock()
    cell1.procedure = {"proc1": MagicMock(), "proc2": MagicMock(), "proc3": MagicMock()}
    cell2.procedure = {"proc1": MagicMock(), "proc2": MagicMock(), "proc4": MagicMock()}
    dashboard = _Dashboard([cell1, cell2])
    dashboard.selected_indices = [0, 1]
    assert dashboard.get_common_procedures() == ["proc1", "proc2"]

    """Test when one cell has no procedures."""
    cell1 = MagicMock()
    cell2 = MagicMock()
    cell1.procedure = {"proc1": MagicMock()}
    cell2.procedure = {}
    dashboard = _Dashboard([cell1, cell2])
    dashboard.selected_indices = [0, 1]
    assert dashboard.get_common_procedures() == []


def test_dashboard_select_experiment():
    """Test experiment selection functionality."""

    def select_exp_mini_app():
        from unittest.mock import MagicMock

        import streamlit as st  # noqa: F811

        from pyprobe.dashboard import _Dashboard

        cell = MagicMock()
        mock_procedure = MagicMock()
        mock_procedure.experiment_names = ["Exp1", "Exp2", "Exp3"]
        cell.procedure = {"Pro1": mock_procedure, "Pro2": MagicMock()}
        dashboard = _Dashboard([cell])
        dashboard.selected_procedure = "Pro1"
        dashboard.selected_indices = [0]
        dashboard.select_experiment()

        dashboard = _Dashboard([cell])
        dashboard.selected_indices = [0]
        dashboard.selected_procedure = None
        st.session_state.expected_blank_tuple = dashboard.select_experiment()

    at = AppTest.from_function(select_exp_mini_app)
    at.run(timeout=30)
    assert at.sidebar.multiselect[0].options == ["Exp1", "Exp2", "Exp3"]

    assert at.session_state.expected_blank_tuple == ()


def test_get_data(cell_fixture):
    """Test data retrieval functionality."""
    dashboard = _Dashboard([cell_fixture])
    dashboard.selected_indices = [0]
    dashboard.selected_experiments = ()
    dashboard.selected_procedure = "Sample"
    dashboard.cycle_step_input = False
    pl_testing.assert_frame_equal(
        dashboard.get_data()[0].data,
        cell_fixture.procedure["Sample"].data,
        check_column_order=False,
    )

    dashboard.selected_experiments = tuple(["Break-in Cycles"])
    pl_testing.assert_frame_equal(
        dashboard.get_data()[0].data,
        cell_fixture.procedure["Sample"].experiment("Break-in Cycles").data,
        check_column_order=False,
    )

    dashboard.cycle_step_input = "cycle(1).discharge(0)"
    pl_testing.assert_frame_equal(
        dashboard.get_data()[0].data,
        cell_fixture.procedure["Sample"]
        .experiment("Break-in Cycles")
        .cycle(1)
        .discharge(0)
        .data,
        check_column_order=False,
    )


def test_dashboard_run(cell_fixture):
    """Test running the dashboard."""

    def run_mini_app():
        from unittest.mock import patch

        import streamlit as st  # noqa: F811

        from pyprobe import Cell
        from pyprobe.dashboard import _Dashboard

        cell = Cell(
            info={
                "name": "test",
                "temperature": 25,
            }
        )
        cell.add_procedure(
            "Sample", "tests/sample_data/neware/", "sample_data_neware.parquet"
        )
        cell.add_procedure(
            "Sample 2", "tests/sample_data/neware/", "sample_data_neware.parquet"
        )

        dashboard = _Dashboard([cell])
        with patch.object(_Dashboard, "select_cell_indices", return_value=[0]):
            dashboard.run()

            st.session_state.figure = dashboard.fig

    at = AppTest.from_function(run_mini_app)
    at.run(timeout=30)

    assert at.title[0].value == "PyProBE Dashboard"
    assert at.sidebar.title[0].value == "Select data to plot"

    # Check procedure selection
    procedure_selector = at.sidebar.selectbox[0]
    assert procedure_selector.options == ["Sample", "Sample 2"]
    procedure_selector.select("Sample")
    at.run(timeout=30)

    filter_stage_select = at.selectbox[0]
    # Check plot
    assert filter_stage_select.options == ["", "Experiment", "Cycle", "Step"]
    assert at.selectbox[1].options == _Dashboard.x_options
    assert at.selectbox[2].options == _Dashboard.y_options
    assert at.selectbox[3].options == ["None"] + _Dashboard.y_options
    assert at.selectbox[4].options == ["name", "temperature"]

    filter_stage_select.select("")
    at.selectbox[1].select("Time [s]")
    at.selectbox[2].select("Voltage [V]")
    at.selectbox[3].select("None")
    at.selectbox[4].select("name")
    at.run(timeout=30)

    fig = at.session_state.figure
    assert fig["layout"]["xaxis"]["title"]["text"] == "Time [s]"
    assert fig["layout"]["yaxis"]["title"]["text"] == "Voltage [V]"
    np.testing.assert_array_equal(
        fig["data"][0]["x"], cell_fixture.procedure["Sample"].get("Time [s]")
    )
    np.testing.assert_array_equal(
        fig["data"][0]["y"], cell_fixture.procedure["Sample"].get("Voltage [V]")
    )

    # Check plot with multiple y axes
    at.selectbox[3].select("Current [A]")
    at.run(timeout=30)
    fig = at.session_state.figure
    assert fig["layout"]["xaxis"]["title"]["text"] == "Time [s]"
    assert fig["layout"]["yaxis"]["title"]["text"] == "Voltage [V]"
    assert fig["layout"]["yaxis2"]["title"]["text"] == "Current [A]"
    np.testing.assert_array_equal(
        fig["data"][1]["x"], cell_fixture.procedure["Sample"].get("Time [s]")
    )
    np.testing.assert_array_equal(
        fig["data"][1]["y"], cell_fixture.procedure["Sample"].get("Current [A]")
    )
    assert fig["data"][2]["name"] == "Current [A]"
    assert fig["data"][2]["line"]["color"] == "black"
    assert fig["data"][2]["line"]["dash"] == "dash"
    assert fig["data"][2]["x"] == (None,)
    assert fig["data"][2]["y"] == (None,)

    # Check unit conversion
    at.selectbox[1].select("Time [hr]")
    at.selectbox[3].select("Current [mA]")
    at.run(timeout=30)
    fig = at.session_state.figure
    np.testing.assert_allclose(
        fig["data"][1]["x"], cell_fixture.procedure["Sample"].get("Time [s]") / 3600
    )
    np.testing.assert_allclose(
        fig["data"][1]["y"], cell_fixture.procedure["Sample"].get("Current [A]") * 1000
    )

    # Check filtering by experiment
    experiment_selector = at.sidebar.multiselect[0]
    assert experiment_selector.options == [
        "Initial Charge",
        "Break-in Cycles",
        "Discharge Pulses",
    ]
    experiment_selector.select("Break-in Cycles")
    at.run(timeout=30)
    fig = at.session_state.figure
    np.testing.assert_array_equal(
        fig["data"][0]["x"],
        cell_fixture.procedure["Sample"].experiment("Break-in Cycles").get("Time [hr]"),
    )
    np.testing.assert_array_equal(
        fig["data"][0]["y"],
        cell_fixture.procedure["Sample"]
        .experiment("Break-in Cycles")
        .get("Voltage [V]"),
    )
    np.testing.assert_array_equal(
        fig["data"][1]["x"],
        cell_fixture.procedure["Sample"].experiment("Break-in Cycles").get("Time [hr]"),
    )
    np.testing.assert_array_equal(
        fig["data"][1]["y"],
        cell_fixture.procedure["Sample"]
        .experiment("Break-in Cycles")
        .get("Current [mA]"),
    )

    # check filtering by cycle and step
    at.sidebar.text_input[0].set_value("cycle(1).discharge(0)")
    at.run(timeout=30)
    fig = at.session_state.figure
    np.testing.assert_array_equal(
        fig["data"][0]["x"],
        cell_fixture.procedure["Sample"]
        .experiment("Break-in Cycles")
        .cycle(1)
        .discharge(0)
        .get("Time [hr]"),
    )
    np.testing.assert_array_equal(
        fig["data"][0]["y"],
        cell_fixture.procedure["Sample"]
        .experiment("Break-in Cycles")
        .cycle(1)
        .discharge(0)
        .get("Voltage [V]"),
    )
    np.testing.assert_array_equal(
        fig["data"][1]["x"],
        cell_fixture.procedure["Sample"]
        .experiment("Break-in Cycles")
        .cycle(1)
        .discharge(0)
        .get("Time [hr]"),
    )
    np.testing.assert_array_equal(
        fig["data"][1]["y"],
        cell_fixture.procedure["Sample"]
        .experiment("Break-in Cycles")
        .cycle(1)
        .discharge(0)
        .get("Current [mA]"),
    )

    at.selectbox[0].select("Cycle")
    at.selectbox[1].select("Capacity [Ah]")
    at.run(timeout=30)
    fig = at.session_state.figure
    np.testing.assert_array_equal(
        fig["data"][0]["x"],
        cell_fixture.procedure["Sample"]
        .experiment("Break-in Cycles")
        .cycle(1)
        .discharge(0)
        .get("Cycle Capacity [Ah]"),
    )
    np.testing.assert_array_equal(
        fig["data"][0]["y"],
        cell_fixture.procedure["Sample"]
        .experiment("Break-in Cycles")
        .cycle(1)
        .discharge(0)
        .get("Voltage [V]"),
    )
    np.testing.assert_array_equal(
        fig["data"][1]["x"],
        cell_fixture.procedure["Sample"]
        .experiment("Break-in Cycles")
        .cycle(1)
        .discharge(0)
        .get("Cycle Capacity [Ah]"),
    )
    np.testing.assert_array_equal(
        fig["data"][1]["y"],
        cell_fixture.procedure["Sample"]
        .experiment("Break-in Cycles")
        .cycle(1)
        .discharge(0)
        .get("Current [mA]"),
    )

    # printed_data = at.dataframe[0].value
    expected_df = (
        cell_fixture.procedure["Sample"]
        .experiment("Break-in Cycles")
        .cycle(1)
        .discharge(0)
        .data.select(
            [
                "Time [s]",
                "Step",
                "Current [A]",
                "Voltage [V]",
                "Capacity [Ah]",
            ]
        )
        .to_pandas()
    )
    assert at.dataframe[0].value.equals(expected_df)
