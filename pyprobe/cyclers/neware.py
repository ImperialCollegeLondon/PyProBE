"""A module to load and process Neware battery cycler data."""

import glob
import os
import re
from typing import List, Tuple

import polars as pl

from pyprobe.cyclers.basecycler import BaseCycler
from pyprobe.unitconverter import UnitConverter


def read_file(filepath: str) -> pl.DataFrame:
    """Read a battery cycler file into a DataFrame.

    Args:
        filepath (str): The path to the file.

    Returns:
        pl.DataFrame: The DataFrame.
    """
    file = os.path.basename(filepath)
    file_ext = os.path.splitext(file)[1]
    match file_ext:
        case ".xlsx":
            return pl.read_excel(filepath, engine="calamine")
        case ".csv":
            return pl.read_csv(filepath)
        case _:
            raise ValueError(f"Unsupported file extension: {file_ext}")


def sort_key(filepath_tuple: Tuple[str, str]) -> int:
    """Sort key for the files.

    Args:
        filepath_tupe (Tuple): A tuple containing the file suffix
            and the full filepath.

    Returns:
        int: The integer in the filename.
    """
    filepath = filepath_tuple[0]
    match = re.search(r"(\d+)(?=\.)", filepath)
    return int(match.group()) if match else 0


def sort_files(file_list: List[str]) -> List[str]:
    """Sort a list of files by the integer in the filename.

    Args:
        file_list (List[str]): The list of files.

    Returns:
        List[str]: The sorted list of files.
    """
    common_substring = os.path.commonprefix(file_list)
    stripped_and_converted = [
        (file.replace(common_substring, ""), file) for file in file_list
    ]
    stripped_and_converted.sort(key=sort_key)

    return [file for _, file in stripped_and_converted]


def read_all_files(filepath: str) -> pl.DataFrame:
    """Read a battery cycler file into a DataFrame.

    Args:
        filepath (str): The path to the file.
    """
    files = glob.glob(filepath)
    files = sort_files(files)
    dataframes = [read_file(file) for file in files]
    return pl.concat(dataframes, how="vertical", rechunk=True)


def process_dataframe(dataframe: pl.DataFrame) -> pl.DataFrame:
    """Process a DataFrame from battery cycler data.

    Args:
        dataframe: The DataFrame to process.

    Returns:
        pl.DataFrame: The dataframe in PyProBE format.
    """
    columns = dataframe.columns

    if dataframe.dtypes[dataframe.columns.index("Date")] != pl.Datetime:
        date = pl.col("Date").str.to_datetime().alias("Date")
        dataframe = dataframe.with_columns(date)

    # Time
    time = (
        (pl.col("Date").diff().dt.total_microseconds().cum_sum() / 1e6)
        .fill_null(strategy="zero")
        .alias("Time [s]")
    )

    # Cycle and step
    step = pl.col("Step Index").alias("Step")

    # Measured data
    column_name_pattern = r"(.+)\((.+)\)"
    current = UnitConverter.search_columns(
        columns, "Current", column_name_pattern, "Current"
    ).to_default()
    voltage = UnitConverter.search_columns(
        columns, "Voltage", column_name_pattern, "Voltage"
    ).to_default()

    dataframe = dataframe.with_columns(time, step, current, voltage)

    make_charge_capacity = UnitConverter.search_columns(
        columns, "Chg. Cap.", column_name_pattern, "Capacity"
    ).to_default(keep_name=True)
    make_discharge_capacity = UnitConverter.search_columns(
        columns, "DChg. Cap.", column_name_pattern, "Capacity"
    ).to_default(keep_name=True)

    dataframe = dataframe.with_columns(make_charge_capacity, make_discharge_capacity)

    diff_charge_capacity = (
        pl.col("Chg. Cap. [Ah]").diff().clip(lower_bound=0).fill_null(strategy="zero")
    )
    diff_discharge_capacity = (
        pl.col("DChg. Cap. [Ah]").diff().clip(lower_bound=0).fill_null(strategy="zero")
    )
    make_capacity = (
        (diff_charge_capacity - diff_discharge_capacity).cum_sum()
        + pl.col("Chg. Cap. [Ah]").max()
    ).alias("Capacity [Ah]")

    dataframe = dataframe.with_columns(make_capacity)
    return dataframe


neware = BaseCycler(read_all_files, process_dataframe)
