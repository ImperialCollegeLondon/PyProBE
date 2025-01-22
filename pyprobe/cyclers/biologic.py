"""A module to load and process Biologic battery cycler data."""

import re
from datetime import datetime
from typing import List

import polars as pl

from pyprobe.cyclers.basecycler import BaseCycler


class Biologic(BaseCycler):
    """A class to load and process Biologic battery cycler data."""

    input_data_path: str
    column_dict: dict[str, str] = {
        "Date": "Date",
        "time/*": "Time [*]",
        "Ns": "Step",
        "I/*": "Current [*]",
        "<I>/*": "Current [*]",
        "Ecell/*": "Voltage [*]",
        "Q charge/*": "Charge Capacity [*]",
        "Q discharge/*": "Discharge Capacity [*]",
        "Temperature/*": "Temperature [*]",
    }

    @staticmethod
    def read_file(
        filepath: str, header_row_index: int = 0
    ) -> pl.DataFrame | pl.LazyFrame:
        """Read a battery cycler file into a DataFrame.

        Args:
            filepath: The path to the file.
            header_row_index: The index of the header row.

        Returns:
            pl.DataFrame | pl.LazyFrame: The DataFrame.
        """
        with open(filepath, "r", encoding="iso-8859-1") as file:
            file.readline()  # Skip the first line
            second_line = file.readline().strip()  # Read the second line
            if second_line.startswith("Nb header lines"):
                read_header_lines = True
            else:
                read_header_lines = False
        if read_header_lines:  # get the provided number of header lines
            _, value = second_line.split(":")
            n_header_lines = int(value.strip())
        else:
            n_header_lines = 1

        dataframe = pl.scan_csv(
            filepath, skip_rows=n_header_lines - 1, separator="\t", infer_schema=False
        )

        # check if the time column is in datetime format
        datetime_regex = r"^\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2}\.\d+$"
        date_string = dataframe.select("time/s").first().collect().item()
        if bool(re.match(datetime_regex, date_string)):
            dataframe = dataframe.with_columns(
                (
                    pl.col("time/s").str.strptime(
                        dtype=pl.Datetime, format="%m/%d/%Y %H:%M:%S%.f"
                    )
                ).alias("Date")
            )
            dataframe = dataframe.with_columns(
                (pl.col("Date").diff().dt.total_microseconds().cum_sum() / 1e6)
                .fill_null(strategy="zero")
                .cast(str)
                .alias("time/s")
            )
        # if the date column is not in datetime format try to retrieve date information
        # from header
        elif read_header_lines:
            with open(filepath, "r", encoding="iso-8859-1") as file:
                for i in range(n_header_lines):
                    line = file.readline()
                    if "Acquisition started on" in line:
                        start_time_line = line
                        break
            _, value = start_time_line.split(" : ")
            start_time = datetime.strptime(value.strip(), "%m/%d/%Y %H:%M:%S.%f")

            dataframe = dataframe.with_columns(
                (
                    pl.col("time/s").cast(pl.Float64).cast(pl.Duration)
                    + pl.lit(start_time)
                )
                .cast(str)
                .alias("Date")
            )

        return dataframe


class BiologicMB(Biologic):
    """A class to load and process Biologic Modulo Bat  battery cycler data."""

    def get_imported_dataframe(
        self, dataframe_list: List[pl.DataFrame]
    ) -> pl.DataFrame | pl.LazyFrame:
        """Read a battery cycler file into a DataFrame.

        Args:
            dataframe_list: The list of DataFrames to concatenate.

        Returns:
            The imported DataFrame.
        """
        df_list = []
        for i, df in enumerate(dataframe_list):
            df = df.with_columns(pl.lit(i).alias("MB File"))
            df_list.append(df)
        complete_df = pl.concat(df_list, how="diagonal")
        complete_df = self.apply_step_correction(complete_df)
        return complete_df

    @staticmethod
    def apply_step_correction(
        df: pl.DataFrame | pl.LazyFrame,
    ) -> pl.DataFrame | pl.LazyFrame:
        """Correct the step column.

        This method adds the maximum step number from the previous MB file to the step
        number of the following MB file so they monotonically increase.

        Args:
            df: The DataFrame to correct.

        Returns:
            pl.DataFrame: The corrected DataFrame.
        """
        # get the max step number for each MB file and add 1
        max_steps = df.group_by("MB File").agg(
            (pl.col("Ns").cast(pl.Int64).max() + 1).alias("Max_Step")
        )
        # sort the max steps by MB file
        max_steps = max_steps.sort("MB File")
        # get the cumulative sum of the max steps
        max_steps = max_steps.with_columns(pl.col("Max_Step").cum_sum())
        # add 1 to the MB file number to offset the join
        max_steps = max_steps.with_columns(pl.col("MB File") + 1)
        # join the max step number to the original dataframe and fill nulls with 0
        df_with_max_step = df.join(max_steps, on="MB File", how="left").fill_null(0)
        # add the max step number to the step number
        return df_with_max_step.with_columns(
            pl.col("Ns").cast(pl.Int64) + pl.col("Max_Step")
        )
