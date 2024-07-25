"""A module to load and process Biologic battery cycler data."""


from datetime import datetime

import polars as pl

from pyprobe.cyclers.basecycler import BaseCycler


class Biologic(BaseCycler):
    """A class to load and process Biologic battery cycler data."""

    def __init__(self, input_data_path: str) -> None:
        """Create a Biologic cycler object.

        Args:
            input_data_path: The path to the input data.
        """
        super().__init__(
            input_data_path,
            common_suffix="_MB",
            column_name_pattern=r"(.+)/(.+)",
            column_dict={
                "Date": "Date",
                "Time": "time/s",
                "Step": "Ns",
                "Current": "I",
                "Voltage": "Ecell",
                "Charge Capacity": "Q charge",
                "Discharge Capacity": "Q discharge",
            },
        )

    @staticmethod
    def read_file(filepath: str) -> pl.DataFrame:
        """Read a battery cycler file into a DataFrame.

        Args:
            filepath: The path to the file.
        """
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

        columns_to_read = ["time/", "Ns", "I/", "Ecell/", "Q charge/", "Q discharge/"]

        all_columns = pl.scan_csv(
            filepath, skip_rows=n_header_lines - 1, separator="\t"
        ).columns
        selected_columns = []
        for substring in columns_to_read:
            found_columns = [col for col in all_columns if substring in col]
            selected_columns.extend(found_columns)

        dataframe = pl.read_csv(
            filepath,
            skip_rows=n_header_lines - 1,
            separator="\t",
            columns=selected_columns,
        )

        start = pl.DataFrame({"start": [start_time]})
        dataframe = dataframe.with_columns(
            (pl.col("time/s") * 1000000 + start).cast(pl.Datetime).alias("Date")
        )
        return dataframe

    @property
    def imported_dataframe(self) -> pl.DataFrame:
        """Read a battery cycler file into a DataFrame.

        Args:
            filepath: The path to the file.
        """
        for i in range(len(self.dataframe_list)):
            self.dataframe_list[i] = self.dataframe_list[i].with_columns(
                pl.col("Ns") + self.dataframe_list[i - 1]["Ns"].max() + 1
            )
        return pl.concat(self.dataframe_list, how="vertical", rechunk=True)

    @property
    def step(self) -> pl.Expr:
        """Identify and format the step column."""
        return (pl.col(self.column_dict["Step"]) + 1).alias("Step")
