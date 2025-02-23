"""Tests for the preprocess module."""

import os

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from pyprobe.cyclers import basecycler
from pyprobe.preprocess import process_cycler_data


def test_process_cycler_data(mocker):
    """Test the process_cycler_file method."""
    output_name = "test.parquet"

    cyclers = ["neware", "maccor", "biologic", "basytec", "arbin"]
    file_paths = [
        "tests/sample_data/neware/sample_data_neware.xlsx",
        "tests/sample_data/maccor/sample_data_maccor.csv",
        "tests/sample_data/biologic/sample_data_biologic_CA1.txt",
        "tests/sample_data/basytec/sample_data_basytec.txt",
        "tests/sample_data/arbin/sample_data_arbin.csv",
    ]

    for cycler, file in zip(cyclers, file_paths):
        process_patch = mocker.patch(
            f"pyprobe.cyclers.{cycler}.{cycler.capitalize()}.process"
        )
        process_cycler_data(
            cycler,
            file,
            output_name,
        )
        process_patch.assert_called_once()


def test_process_cycler_data_generic():
    """Test the process_generic_file method."""
    data_path = "tests/sample_data/test_generic_file.csv"
    df = pl.DataFrame(
        {
            "T [s]": [1.0, 2.0, 3.0],
            "V [V]": [4.0, 5.0, 6.0],
            "I [A]": [7.0, 8.0, 9.0],
            "Q [Ah]": [10.0, 11.0, 12.0],
            "Count": [1, 2, 3],
        },
    )

    column_importers = [
        basecycler.ConvertUnits("Time [s]", "T [*]"),
        basecycler.ConvertUnits("Voltage [V]", "V [*]"),
        basecycler.ConvertUnits("Current [A]", "I [*]"),
        basecycler.ConvertUnits("Capacity [Ah]", "Q [*]"),
        basecycler.CastAndRename("Step", "Count", pl.Int64),
    ]

    df.write_csv(data_path)

    process_cycler_data(
        cycler_type="generic",
        input_data_path=data_path,
        column_importers=column_importers,
    )
    expected_df = pl.DataFrame(
        {
            "Time [s]": [1.0, 2.0, 3.0],
            "Step": [1, 2, 3],
            "Event": [0, 1, 2],
            "Current [A]": [7.0, 8.0, 9.0],
            "Voltage [V]": [4.0, 5.0, 6.0],
            "Capacity [Ah]": [10.0, 11.0, 12.0],
        },
    )
    saved_df = pl.read_parquet(data_path.replace(".csv", ".parquet"))
    assert_frame_equal(expected_df, saved_df, check_column_order=False)

    with pytest.raises(ValueError):
        process_cycler_data(
            cycler_type="generic",
            input_data_path=data_path,
        )

    os.remove(data_path)
    os.remove(data_path.replace(".csv", ".parquet"))
