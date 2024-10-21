"""Tests for the Arbin cycler class."""

from pyprobe.cyclers.arbin import Arbin


def test_read_and_process(benchmark):
    """Test the full process of reading and processing a file."""
    arbin_cycler = Arbin(
        input_data_path="tests/sample_data/arbin/sample_data_arbin.csv"
    )

    def read_and_process():
        return arbin_cycler.pyprobe_dataframe

    pyprobe_dataframe = benchmark(read_and_process)
    expected_columns = [
        "Date",
        "Time [s]",
        "Step",
        "Cycle",
        "Event",
        "Current [A]",
        "Voltage [V]",
        "Capacity [Ah]",
        "Temperature [C]",
    ]
    assert set(pyprobe_dataframe.columns) == set(expected_columns)
    assert set(
        pyprobe_dataframe.select("Event").unique().collect().to_series().to_list()
    ) == set([0, 1, 2])
