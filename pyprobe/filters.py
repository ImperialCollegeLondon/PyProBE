"""A module for the filtering classes."""
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import polars as pl

from pyprobe.rawdata import RawData
from pyprobe.typing import (  # , FilterToStepType
    FilterToCycleType,
    FilterToExperimentType,
)


def filter_numerical(
    dataframe: pl.LazyFrame | pl.DataFrame,
    column: str,
    indices: Tuple[Union[int, range], ...],
) -> pl.LazyFrame | pl.DataFrame:
    """Filter a polars Lazyframe or Dataframe by a numerical condition.

    Args:
        dataframe (pl.LazyFrame | pl.DataFrame): A LazyFrame object.
        column (str): The column to filter on.
        indices (Tuple[Union[int, range], ...]): A tuple of index values to filter by.

    Returns:
        pl.LazyFrame | pl.DataFrame: A filtered LazyFrame object.
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


def step(
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
        base_dataframe = filter_numerical(
            filter.base_dataframe.filter(condition), "Event", step_numbers
        )
    else:
        base_dataframe = filter_numerical(filter.base_dataframe, "Event", step_numbers)
    return Step(
        base_dataframe=base_dataframe,
        info=filter.info,
        column_definitions=filter.column_definitions,
    )


def cycle(filter: "FilterToExperimentType", *cycle_numbers: Union[int]) -> "Cycle":
    """Return a cycle object. Filters on the Cycle column.

    Args:
        filter (FilterToExperimentType): A filter object that this method is called on.
        cycle_numbers (int | range):
            Variable-length argument list of cycle indices or a range object.

    Returns:
        Cycle: A cycle object.
    """
    lf_filtered = filter_numerical(filter.base_dataframe, "Cycle", cycle_numbers)

    return Cycle(
        base_dataframe=lf_filtered,
        info=filter.info,
        column_definitions=filter.column_definitions,
    )


def charge(filter: "FilterToCycleType", *charge_numbers: Union[int, range]) -> "Step":
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


def discharge(
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


def chargeordischarge(
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


def rest(filter: "FilterToCycleType", *rest_numbers: Union[int, range]) -> "Step":
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


def constant_current(
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


def constant_voltage(
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


@dataclass(kw_only=True)
class Procedure(RawData):
    """A class for a procedure in a battery experiment.

    Args:
        base_dataframe (pl.LazyFrame | pl.DataFrame): The data for the procedure.
        info (Dict[str, str | int | float]): A dict containing test info.
        column_definitions (Dict[str, str], optional):
            A dict containing column definitions. Defaults to None.
        titles (Dict[str, str]): A dict containing the titles of the experiments.
        steps_idx (List[List[int]]): A list of lists containing the step indices for
            each experiment.

    Filtering attributes:
        step (Callable[..., int]):
            A method to return a step object from the procedure. See :func:`step`.
        cycle (Callable[..., int]):
            A method to return a cycle object from the procedure.
            See :func:`cycle`.
        charge (Callable[..., int]):
            A method to return a charge step object from the procedure.
            See :func:`charge`.
        discharge (Callable[..., int]):
            A method to return a discharge step object from the procedure.
            See :func:`discharge`.
        chargeordischarge (Callable[..., int]):
            A method to return a charge or discharge step object from the procedure.
            See :func:`chargeordischarge`.
        rest (Callable[..., int]):
            A method to return a rest step object from the procedure.
            See :func:`rest`.
        constant_current (Callable[..., int]):
            A method to return a constant current step object from the procedure.
            See :func:`constant_current`.
        constant_voltage (Callable[..., int]):
            A method to return a constant voltage step object from the procedure.
            See :func:`constant_voltage`.
    """

    titles: Dict[str, str]
    steps_idx: List[List[int]]

    def __post_init__(self) -> None:
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

    step = step
    cycle = cycle
    charge = charge
    discharge = discharge
    chargeordischarge = chargeordischarge
    rest = rest
    constant_current = constant_current
    constant_voltage = constant_voltage

    def experiment(self, *experiment_names: str) -> "Experiment":
        """Return an experiment object from the procedure.

        Args:
            experiment_names (str): Variable-length argument list of
                experiment names.

        Returns:
            Experiment: An experiment object from the procedure.
        """
        steps_idx = []
        for experiment_name in experiment_names:
            if experiment_name not in self.titles:
                raise ValueError(f"{experiment_name} not in procedure.")
            experiment_number = list(self.titles.keys()).index(experiment_name)
            steps_idx.append(self.steps_idx[experiment_number])
        flattened_steps = self.flatten(steps_idx)
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
        return list(self.titles.keys())

    def verify_yaml(self, readme_name: str) -> str:
        """Verify that the readme has YAML extension.

        Args:
            readme_name (str): The name of the README file.
        """
        # Get the file extension of output_filename
        _, ext = os.path.splitext(readme_name)

        # If the file extension is not .yaml, replace it with .yaml
        if ext != ".yaml":
            readme_name = os.path.splitext(readme_name)[0] + ".yaml"
        return readme_name

    @classmethod
    def flatten(cls, lst: int | List[Any]) -> List[int]:
        """Flatten a list of lists into a single list.

        Args:
            lst (list): The list of lists to flatten.

        Returns:
            list: The flattened list.
        """
        if not isinstance(lst, list):
            return [lst]
        else:
            return [item for sublist in lst for item in cls.flatten(sublist)]


@dataclass(kw_only=True)
class Experiment(RawData):
    """A class for an experiment in a battery experimental procedure.

    Args:
        dataframe (pl.LazyFrame | pl.DataFrame): The data for the experiment.
        info (Dict[str, str | int | float]): A dict containing test info.
        column_definitions (Dict[str, str], optional):
            A dict containing column definitions. Defaults to None.

    Filtering attributes:
        step (Callable[..., int]):
            A method to return a step object from the experiment. See :func:`step`.
        cycle (Callable[..., int]):
            A method to return a cycle object from the experiment. See :func:`cycle`.
        charge (Callable[..., int]):
            A method to return a charge step object from the experiment.
            See :func:`charge`.
        discharge (Callable[..., int]):
            A method to return a discharge step object from the experiment.
            See :func:`discharge`.
        chargeordischarge (Callable[..., int]):
            A method to return a charge or discharge step object from the experiment.
            See :func:`chargeordischarge`.
        rest (Callable[..., int]):
            A method to return a rest step object from the experiment.
            See :func:`rest`.
        constant_current (Callable[..., int]):
            A method to return a constant current step object.
            See :func:`constant_current`.
        constant_voltage (Callable[..., int]):
            A method to return a constant voltage step object.
            See :func:`constant_voltage`.
    """

    def __post_init__(
        self,
    ) -> None:
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

    step = step
    cycle = cycle
    charge = charge
    discharge = discharge
    chargeordischarge = chargeordischarge
    rest = rest
    constant_current = constant_current
    constant_voltage = constant_voltage


@dataclass(kw_only=True)
class Cycle(RawData):
    """A class for a cycle in a battery experimental procedure.

    Args:
        dataframe (pl.LazyFrame | pl.DataFrame): The data for the cycle.
        info (Dict[str, str | int | float]): A dict containing test info.
        column_definitions (Dict[str, str], optional):
            A dict containing column definitions. Defaults to None.

    Filtering attributes:
        step (Callable[..., int]):
            A method to return a step object from the procedure. See :func:`step`.
        charge (Callable[..., int]):
            A method to return a charge step object from the procedure.
            See :func:`charge`.
        discharge (Callable[..., int]):
            A method to return a discharge step object from the procedure.
            See :func:`discharge`.
        chargeordischarge (Callable[..., int]):
            A method to return a charge or discharge step object from the procedure.
            See :func:`chargeordischarge`.
        rest (Callable[..., int]):
            A method to return a rest step object from the procedure.
            See :func:`rest`.
        constant_current (Callable[..., int]):
            A method to return a constant current step object.
            See :func:`constant_current`.
        constant_voltage (Callable[..., int]):
            A method to return a constant voltage step object.
            See :func:`constant_voltage`.
    """

    def __post_init__(
        self,
    ) -> None:
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

    step = step
    charge = charge
    discharge = discharge
    chargeordischarge = chargeordischarge
    rest = rest
    constant_current = constant_current
    constant_voltage = constant_voltage


@dataclass(kw_only=True)
class Step(RawData):
    """A class for a step in a battery experimental procedure.

    Args:
        dataframe (pl.LazyFrame | pl.DataFrame): The data for the cycle.
        info (Dict[str, str | int | float]): A dict containing test info.
        column_definitions (Dict[str, str], optional):
            A dict containing column definitions. Defaults to None.

    Filtering attributes:
        constant_current (Callable[..., int]):
            A method to return a constant current step object.
            See :func:`constant_current`.
        constant_voltage (Callable[..., int]):
            A method to return a constant voltage step object.
            See :func:`constant_voltage`.
    """

    def __post_init__(
        self,
    ) -> None:
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

    step = step
    constant_current = constant_current
    constant_voltage = constant_voltage
