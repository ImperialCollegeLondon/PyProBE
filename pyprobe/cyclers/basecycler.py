"""A module to load and process battery cycler data."""

import polars as pl


class BaseCycler:
    """A class to load and process battery cycler data."""

    imported_dataframe: pl.DataFrame

    required_columns = [
        "Date",
        "Time [s]",
        "Cycle",
        "Step",
        "Event",
        "Current [A]",
        "Voltage [V]",
        "Capacity [Ah]",
    ]

    @staticmethod
    def get_cycle_and_event(dataframe: pl.DataFrame) -> pl.DataFrame:
        """Get the step and event columns from a DataFrame.

        Args:
            dataframe: The DataFrame to process.

        Returns:
            DataFrame: The DataFrame with the step and event columns.
        """
        cycle = (
            (pl.col("Step") - pl.col("Step").shift() < 0)
            .fill_null(strategy="zero")
            .cum_sum()
            .alias("Cycle")
            .cast(pl.Int64)
        )

        event = (
            (pl.col("Step") - pl.col("Step").shift() != 0)
            .fill_null(strategy="zero")
            .cum_sum()
            .alias("Event")
            .cast(pl.Int64)
        )
        return dataframe.with_columns(cycle, event)
