"""A module to load and process Biologic battery cycler data."""


import os

import polars as pl


def read_file(filepath: str) -> pl.DataFrame:
    """Read a battery cycler file into a DataFrame.

    Args:
        filepath: The path to the file.
    """
    filename = os.path.basename(filepath)
    file_ext = os.path.splitext(filename)[1]

    with open(filepath, "r", encoding="shift_jis") as file:
        file.readline()  # Skip the first line
        second_line = file.readline().strip()  # Read the second line
    _, value = second_line.split(":")
    n_header_lines = int(value.strip())

    match file_ext:
        case ".mpt":
            return pl.read_csv(filepath, skip_rows=n_header_lines - 1, separator="\t")
        case _:
            raise ValueError(f"Unsupported file extension: {file_ext}")
