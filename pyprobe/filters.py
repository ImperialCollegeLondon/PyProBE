"""A module for the filtering classes."""
import os
import warnings
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import polars as pl
from pybamm import Experiment as PybammExperiment
from pydantic import Field

from pyprobe.rawdata import RawData, default_column_definitions

if TYPE_CHECKING:
    from pyprobe.typing import (  # , FilterToStepType
        FilterToCycleType,
        FilterToExperimentType,
    )


def _filter_numerical(
    dataframe: pl.LazyFrame | pl.DataFrame,
    column: str,
    indices: Tuple[Union[int, range], ...],
) -> pl.LazyFrame | pl.DataFrame:
    """Filter a polars Lazyframe or Dataframe by a numerical condition.

    Args:
        dataframe (pl.LazyFrame | pl.DataFrame): A LazyFrame or DataFrame to filter.
        column (str): The column to filter on.
        indices (Tuple[Union[int, range], ...]): A tuple of index values to filter by.

    Returns:
        pl.LazyFrame | pl.DataFrame: A filtered LazyFrame or DataFrame.

    Raises:
        ValueError: If indices are not all positive or all negative.
    """
    index_list = []
    for index in indices:
        if isinstance(index, range):
            index_list.extend(list(index))
        else:
            index_list.extend([index])

    if len(index_list) > 0:
        if all(item >= 0 for item in index_list):
            index_list = [item + 1 for item in index_list]
            return dataframe.filter(pl.col(column).rank("dense").is_in(index_list))
        elif all(item < 0 for item in index_list):
            index_list = [item * -1 for item in index_list]
            return dataframe.filter(
                pl.col(column).rank("dense", descending=True).is_in(index_list)
            )
        else:
            raise ValueError("Indices must be all positive or all negative.")
    else:
        return dataframe


def _step(
    filter: "FilterToCycleType",
    *step_numbers: Union[int, range],
    condition: Optional[pl.Expr] = None,
) -> "Step":
    """Return a step object. Filters to a numerical condition on the Event column.

    Args:
        filter (FilterToCycleType): A filter object that this method is called on.
        step_numbers (int | range):
            Variable-length argument list of step indices or a range object.
        condition (pl.Expr, optional):
            A polars expression to filter the step before applying the numerical filter.
            Defaults to None.

    Returns:
        Step: A step object.
    """
    if condition is not None:
        base_dataframe = _filter_numerical(
            filter.base_dataframe.filter(condition), "Event", step_numbers
        )
    else:
        base_dataframe = _filter_numerical(filter.base_dataframe, "Event", step_numbers)
    return Step(
        base_dataframe=base_dataframe,
        info=filter.info,
        column_definitions=filter.column_definitions,
    )


def _cycle(filter: "FilterToExperimentType", *cycle_numbers: Union[int]) -> "Cycle":
    """Return a cycle object. Filters on the Cycle column.

    Args:
        filter (FilterToExperimentType): A filter object that this method is called on.
        cycle_numbers (int | range):
            Variable-length argument list of cycle indices or a range object.

    Returns:
        Cycle: A cycle object.
    """
    lf_filtered = _filter_numerical(filter.base_dataframe, "Cycle", cycle_numbers)

    return Cycle(
        base_dataframe=lf_filtered,
        info=filter.info,
        column_definitions=filter.column_definitions,
    )


def _charge(filter: "FilterToCycleType", *charge_numbers: Union[int, range]) -> "Step":
    """Return a charge step.

    Args:
        filter (FilterToCycleType): A filter object that this method is called on.
        charge_numbers (int | range):
            Variable-length argument list of charge indices or a range object.

    Returns:
        Step: A charge step object.
    """
    condition = pl.col("Current [A]") > 0
    return filter.step(*charge_numbers, condition=condition)


def _discharge(
    filter: "FilterToCycleType",
    *discharge_numbers: Union[int, range],
) -> "Step":
    """Return a discharge step.

    Args:
        filter (FilterToCycleType): A filter object that this method is called on.
        discharge_numbers (int | range):
            Variable-length argument list of discharge indices or a range object.

    Returns:
        Step: A discharge step object.
    """
    condition = pl.col("Current [A]") < 0
    return filter.step(*discharge_numbers, condition=condition)


def _chargeordischarge(
    filter: "FilterToCycleType",
    *chargeordischarge_numbers: Union[int, range],
) -> "Step":
    """Return a charge or discharge step.

    Args:
        filter (FilterToCycleType): A filter object that this method is called on.
        chargeordischarge_numbers (int | range):
            Variable-length argument list of charge or discharge indices or a range
            object.

    Returns:
        Step: A charge or discharge step object.
    """
    condition = pl.col("Current [A]") != 0
    return filter.step(*chargeordischarge_numbers, condition=condition)


def _rest(filter: "FilterToCycleType", *rest_numbers: Union[int, range]) -> "Step":
    """Return a rest step object.

    Args:
        filter (FilterToCycleType): A filter object that this method is called on.
        rest_numbers (int | range):
            Variable-length argument list of rest indices or a range object.

    Returns:
        Step: A rest step object.
    """
    condition = pl.col("Current [A]") == 0
    return filter.step(*rest_numbers, condition=condition)


def _constant_current(
    filter: "FilterToCycleType",
    *constant_current_numbers: Union[int, range],
) -> "Step":
    """Return a constant current step object.

    Args:
        filter (FilterToCycleType): A filter object that this method is called on.
        constant_current_numbers (int | range):
            Variable-length argument list of constant current indices or a range object.

    Returns:
        Step: A constant current step object.
    """
    condition = (
        (pl.col("Current [A]") != 0)
        & (
            pl.col("Current [A]").abs()
            > 0.999 * pl.col("Current [A]").abs().round_sig_figs(4).mode()
        )
        & (
            pl.col("Current [A]").abs()
            < 1.001 * pl.col("Current [A]").abs().round_sig_figs(4).mode()
        )
    )
    return filter.step(*constant_current_numbers, condition=condition)


def _constant_voltage(
    filter: "FilterToCycleType",
    *constant_voltage_numbers: Union[int, range],
) -> "Step":
    """Return a constant voltage step object.

    Args:
        filter (FilterToCycleType): A filter object that this method is called on.
        constant_current_numbers (int | range):
            Variable-length argument list of constant voltage indices or a range object.

    Returns:
        Step: A constant voltage step object.
    """
    condition = (
        pl.col("Voltage [V]").abs()
        > 0.999 * pl.col("Voltage [V]").abs().round_sig_figs(4).mode()
    ) & (
        pl.col("Voltage [V]").abs()
        < 1.001 * pl.col("Voltage [V]").abs().round_sig_figs(4).mode()
    )
    return filter.step(*constant_voltage_numbers, condition=condition)


class Procedure(RawData):
    """A class for a procedure in a battery experiment."""

    titles: List[str]
    """The titles of the experiments in the procedure."""
    steps_idx: List[List[int]]
    """The indices of the steps in each experiment."""
    pybamm_experiment: Optional[PybammExperiment]
    """A PyBaMM experiment object for the whole procedure."""
    pybamm_experiment_list: List[Optional[PybammExperiment]]
    """A list of PyBaMM experiment objects for each experiment in the procedure."""

    base_dataframe: pl.LazyFrame | pl.DataFrame
    info: Dict[str, Optional[str | int | float]]
    column_definitions: Dict[str, str] = Field(
        default_factory=lambda: default_column_definitions.copy()
    )

    def model_post_init(self, __context: Any) -> None:
        """Create a procedure class."""
        self.zero_column(
            "Time [s]",
            "Procedure Time [s]",
            "Time elapsed since beginning of procedure.",
        )

        self.zero_column(
            "Capacity [Ah]",
            "Procedure Capacity [Ah]",
            "The net charge passed since beginning of procedure.",
        )

    step = _step
    cycle = _cycle
    charge = _charge
    discharge = _discharge
    chargeordischarge = _chargeordischarge
    rest = _rest
    constant_current = _constant_current
    constant_voltage = _constant_voltage

    def experiment(self, *experiment_names: str) -> "Experiment":
        """Return an experiment object from the procedure.

        Args:
            experiment_names (str):
                Variable-length argument list of experiment names.

        Returns:
            Experiment: An experiment object from the procedure.
        """
        steps_idx = []
        for experiment_name in experiment_names:
            if experiment_name not in self.titles:
                raise ValueError(f"{experiment_name} not in procedure.")
            experiment_number = self.titles.index(experiment_name)
            steps_idx.append(self.steps_idx[experiment_number])
        flattened_steps = self._flatten(steps_idx)
        conditions = [
            pl.col("Step").is_in(flattened_steps),
        ]
        lf_filtered = self.base_dataframe.filter(conditions)
        return Experiment(
            base_dataframe=lf_filtered,
            info=self.info,
            column_definitions=self.column_definitions,
        )

    @property
    def experiment_names(self) -> List[str]:
        """Return the names of the experiments in the procedure.

        Returns:
            List[str]: The names of the experiments in the procedure.
        """
        return list(self.titles)

    def add_external_data(
        self,
        filepath: str,
        importing_columns: List[str] | Dict[str, str],
        date_column_name: str = "Date",
    ) -> None:
        """Add data from another source to the procedure.

        The data must be timestamped, with a column that can be interpreted in
        DateTime format. The data will be interpolated to the procedure's time.

        Args:
            filepath (str): The path to the external file.
            importing_columns (List[str] | Dict[str, str]):
                The columns to import from the external file. If a list, the columns
                will be imported as is. If a dict, the keys are the columns in the data
                you want to import and the values are the columns you want to rename
                them to.
            date_column_name (str, optional):
                The name of the date column in the external data. Defaults to "Date".
        """
        external_data = self.load_external_file(filepath)
        if isinstance(importing_columns, dict):
            external_data = external_data.select(
                [date_column_name] + list(importing_columns.keys())
            )
            external_data = external_data.rename(importing_columns)
        elif isinstance(importing_columns, list):
            external_data = external_data.select([date_column_name] + importing_columns)
        self.add_new_data_columns(external_data, date_column_name)

    def load_external_file(self, filepath: str) -> pl.LazyFrame:
        """Load an external file into a LazyFrame.

        Supported file types are CSV, Parquet, and Excel. For maximum performance,
        consider using Parquet files. If you have an Excel file, consider converting
        it to CSV before loading.

        Args:
            filepath (str): The path to the external file.
        """
        file = os.path.basename(filepath)
        file_ext = os.path.splitext(file)[1]
        match file_ext:
            case ".csv":
                return pl.scan_csv(filepath)
            case ".parquet":
                return pl.scan_parquet(filepath)
            case ".xlsx":
                warnings.warn("Excel reading is slow. Consider converting to CSV.")
                return pl.read_excel(filepath)
            case _:
                raise ValueError(f"Unsupported file type: {file_ext}")

    @classmethod
    def _flatten(cls, lst: int | List[Any]) -> List[int]:
        """Flatten a list of lists into a single list.

        Args:
            lst (list): The list of lists to flatten.

        Returns:
            list: The flattened list.
        """
        if not isinstance(lst, list):
            return [lst]
        else:
            return [item for sublist in lst for item in cls._flatten(sublist)]


class Experiment(RawData):
    """A class for an experiment in a battery experimental procedure."""

    base_dataframe: pl.LazyFrame | pl.DataFrame
    info: Dict[str, Optional[str | int | float]]
    column_definitions: Dict[str, str] = Field(
        default_factory=lambda: default_column_definitions.copy()
    )

    def model_post_init(self, __context: Any) -> None:
        """Create an experiment class."""
        self.zero_column(
            "Time [s]",
            "Experiment Time [s]",
            "Time elapsed since beginning of experiment.",
        )

        self.zero_column(
            "Capacity [Ah]",
            "Experiment Capacity [Ah]",
            "The net charge passed since beginning of experiment.",
        )

    step = _step
    cycle = _cycle
    charge = _charge
    discharge = _discharge
    chargeordischarge = _chargeordischarge
    rest = _rest
    constant_current = _constant_current
    constant_voltage = _constant_voltage


class Cycle(RawData):
    """A class for a cycle in a battery experimental procedure."""

    base_dataframe: pl.LazyFrame | pl.DataFrame
    info: Dict[str, Optional[str | int | float]]
    column_definitions: Dict[str, str] = Field(
        default_factory=lambda: default_column_definitions.copy()
    )

    def model_post_init(self, __context: Any) -> None:
        """Create a cycle class."""
        self.zero_column(
            "Time [s]",
            "Cycle Time [s]",
            "Time elapsed since beginning of cycle.",
        )

        self.zero_column(
            "Capacity [Ah]",
            "Cycle Capacity [Ah]",
            "The net charge passed since beginning of cycle.",
        )

    step = _step
    charge = _charge
    discharge = _discharge
    chargeordischarge = _chargeordischarge
    rest = _rest
    constant_current = _constant_current
    constant_voltage = _constant_voltage


class Step(RawData):
    """A class for a step in a battery experimental procedure."""

    base_dataframe: pl.LazyFrame | pl.DataFrame
    info: Dict[str, Optional[str | int | float]]
    column_definitions: Dict[str, str] = Field(
        default_factory=lambda: default_column_definitions.copy()
    )

    def model_post_init(self, __context: Any) -> None:
        """Create a step class."""
        self.zero_column(
            "Time [s]",
            "Step Time [s]",
            "Time elapsed since beginning of step.",
        )

        self.zero_column(
            "Capacity [Ah]",
            "Step Capacity [Ah]",
            "The net charge passed since beginning of step.",
        )

    step = _step
    constant_current = _constant_current
    constant_voltage = _constant_voltage
