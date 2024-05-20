"""A module to load and process Biologic battery cycler data."""


import os
from datetime import datetime

import polars as pl

from pyprobe.cyclers.basecycler import BaseCycler
from pyprobe.unitconverter import UnitConverter


def read_file(filepath: str) -> pl.DataFrame:
    """Read a battery cycler file into a DataFrame.

    Args:
        filepath: The path to the file.
    """
    filename = os.path.basename(filepath)
    file_ext = os.path.splitext(filename)[1]

    with open(filepath, "r", encoding="iso-8859-1") as file:
        file.readline()  # Skip the first line
        second_line = file.readline().strip()  # Read the second line
    _, value = second_line.split(":")
    n_header_lines = int(value.strip())

    with open(filepath, "r", encoding="iso-8859-1") as file:
        for i in range(n_header_lines):
            line = file.readline()
            if "Acquisition started on" in line:
                start_time_line = line
                break
    _, value = start_time_line.split(" : ")
    start_time = datetime.strptime(value.strip(), "%m/%d/%Y %H:%M:%S.%f")

    match file_ext:
        case ".mpt":
            dataframe = pl.read_csv(
                filepath, skip_rows=n_header_lines - 1, separator="\t"
            )
        case _:
            raise ValueError(f"Unsupported file extension: {file_ext}")

    start = pl.DataFrame({"start": [start_time]})
    dataframe = dataframe.with_columns(
        (pl.col("time/s") * 1000000 + start).cast(pl.Datetime).alias("Date")
    )
    return dataframe


def process_dataframe(dataframe: pl.DataFrame) -> pl.DataFrame:
    """Process a DataFrame from battery cycler data.

    Args:
        dataframe: The DataFrame to process.

    Returns:
        pl.DataFrame: The dataframe in PyProBE format.
    """
    columns = dataframe.columns
    time = pl.col("time/s").alias("Time [s]")

    # Cycle and step
    cycle = pl.col("counter inc.").alias("Cycle")
    step = pl.col("Ns").alias("Step")

    # Measured data
    column_name_pattern = r"(.+)/(.+)"
    current = UnitConverter.search_columns(
        columns, "I", column_name_pattern, "Current"
    ).to_default()
    voltage = UnitConverter.search_columns(
        columns, "Ecell", column_name_pattern, "Voltage"
    ).to_default()

    make_charge_capacity = UnitConverter.search_columns(
        columns, "Q charge", column_name_pattern, "Capacity"
    ).to_default(keep_name=True)
    make_discharge_capacity = UnitConverter.search_columns(
        columns, "Q discharge", column_name_pattern, "Capacity"
    ).to_default(keep_name=True)

    dataframe = dataframe.with_columns(time, cycle, step, current, voltage)

    dataframe = dataframe.with_columns(make_charge_capacity, make_discharge_capacity)

    diff_charge_capacity = (
        pl.col("Q charge [Ah]").diff().clip(lower_bound=0).fill_null(strategy="zero")
    )
    diff_discharge_capacity = (
        pl.col("Q discharge [Ah]").diff().clip(lower_bound=0).fill_null(strategy="zero")
    )
    make_capacity = (
        (diff_charge_capacity - diff_discharge_capacity).cum_sum()
        + pl.col("Q charge [Ah]").max()
    ).alias("Capacity [Ah]")

    dataframe = dataframe.with_columns(make_capacity)
    return dataframe


biologic = BaseCycler(reader=read_file, processor=process_dataframe)
