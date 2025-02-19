"""Tests for the neware module."""

from datetime import datetime

import polars as pl
import polars.testing as pl_testing

from pyprobe.cyclers.neware import Neware

from .test_basecycler import helper_read_and_process


def test_process_dataframe(mocker):
    """Test the neware method."""
    mock_dataframe = pl.LazyFrame(
        {
            "Date": [
                "2022-02-02 02:02:00.00",
                "2022-02-02 02:02:01.00",
                "2022-02-02 02:02:02.00",
                "2022-02-02 02:02:03.00",
                "2022-02-02 02:02:04.00",
                "2022-02-02 02:02:05.10",
            ],
            "Total Time": [
                "2022-02-02 02:02:00.00",
                "2022-02-02 02:02:01.00",
                "2022-02-02 02:02:02.00",
                "2022-02-02 02:02:03.00",
                "2022-02-02 02:02:04.10",
                "2022-02-02 02:02:05.00",
            ],
            "Step Index": [1, 2, 1, 2, 4, 5],
            "Current(mA)": [1, 2, -3, -4, 0, 0],
            "Voltage(V)": [4, 5, 6, 7, 8, 9],
            "Chg. Cap.(mAh)": [
                0,
                20,
                0,
                0,
                0,
                0,
            ],
            "DChg. Cap.(mAh)": [0, 0, 10, 20, 0, 0],
            "Capacity(mAh)": [0, 20, 10, 20, 0, 0],
            "T1(℃)": [25, 25, 25, 25, 25, 25],
        }
    )

    mocker.patch("os.path.exists", return_value=True)
    mocker.patch(
        "pyprobe.cyclers.neware.Neware.get_imported_dataframe",
        return_value=mock_dataframe,
    )
    neware_cycler = Neware(input_data_path="tests/sample_data/mock_dataframe.xlsx")

    pyprobe_dataframe = neware_cycler.get_pyprobe_dataframe()

    pyprobe_dataframe = pyprobe_dataframe.select(
        [
            "Time [s]",
            "Step",
            "Current [A]",
            "Voltage [V]",
            "Capacity [Ah]",
            "Temperature [C]",
        ]
    )
    expected_dataframe = pl.DataFrame(
        {
            "Time [s]": [0.0, 1.0, 2.0, 3.0, 4.0, 5.1],
            "Step": [1, 2, 1, 2, 4, 5],
            "Current [A]": [1e-3, 2e-3, -3e-3, -4e-3, 0, 0],
            "Voltage [V]": [4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            "Capacity [Ah]": [20.0e-3, 40.0e-3, 30.0e-3, 20.0e-3, 20.0e-3, 20.0e-3],
            "Temperature [C]": [25.0, 25.0, 25.0, 25.0, 25.0, 25.0],
        }
    )
    pl_testing.assert_frame_equal(pyprobe_dataframe, expected_dataframe)

    # Test with a dataframe that does not contain a Charge or Discharge Capacity column
    mock_dataframe = mock_dataframe.drop("Chg. Cap.(mAh)")
    mock_dataframe = mock_dataframe.drop("DChg. Cap.(mAh)")
    mocker.patch(
        "pyprobe.cyclers.neware.Neware.get_imported_dataframe",
        return_value=mock_dataframe,
    )
    neware_cycler = Neware(input_data_path="tests/sample_data/mock_dataframe.xlsx")
    pyprobe_dataframe = neware_cycler.get_pyprobe_dataframe()
    pyprobe_dataframe = pyprobe_dataframe.select(
        [
            "Time [s]",
            "Step",
            "Current [A]",
            "Voltage [V]",
            "Capacity [Ah]",
            "Temperature [C]",
        ]
    )
    pl_testing.assert_frame_equal(pyprobe_dataframe, expected_dataframe)

    # Test with a dataframe that does not contain a "Date" column
    mock_dataframe = mock_dataframe.drop("Date")
    mocker.patch(
        "pyprobe.cyclers.neware.Neware.get_imported_dataframe",
        return_value=mock_dataframe,
    )
    neware_cycler = Neware(input_data_path="tests/sample_data/mock_dataframe.xlsx")
    pyprobe_dataframe = neware_cycler.get_pyprobe_dataframe()
    pyprobe_dataframe = pyprobe_dataframe.select(
        [
            "Time [s]",
            "Step",
            "Current [A]",
            "Voltage [V]",
            "Capacity [Ah]",
            "Temperature [C]",
        ]
    )
    expected_dataframe = pl.DataFrame(
        {
            "Time [s]": [0.0, 1.0, 2.0, 3.0, 4.1, 5.0],
            "Step": [1, 2, 1, 2, 4, 5],
            "Current [A]": [1e-3, 2e-3, -3e-3, -4e-3, 0, 0],
            "Voltage [V]": [4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            "Capacity [Ah]": [20.0e-3, 40.0e-3, 30.0e-3, 20.0e-3, 20.0e-3, 20.0e-3],
            "Temperature [C]": [25.0, 25.0, 25.0, 25.0, 25.0, 25.0],
        }
    )
    pl_testing.assert_frame_equal(pyprobe_dataframe, expected_dataframe)


def test_read_and_process_neware(benchmark):
    """Test the full process of reading and processing a file."""
    neware_cycler = Neware(
        input_data_path="tests/sample_data/neware/sample_data_neware.xlsx"
    )
    last_row = pl.DataFrame(
        {
            "Date": [datetime(2024, 3, 6, 21, 39, 38, 591000)],
            "Time [s]": [562749.497],
            "Step": [12],
            "Event": [61],
            "Current [A]": [0.0],
            "Voltage [V]": [3.4513],
            "Capacity [Ah]": [0.022805],
        }
    )
    expected_events = set(range(62))
    expected_columns = [
        "Date",
        "Time [s]",
        "Step",
        "Event",
        "Current [A]",
        "Voltage [V]",
        "Capacity [Ah]",
    ]
    helper_read_and_process(
        benchmark, neware_cycler, last_row, expected_events, expected_columns
    )
