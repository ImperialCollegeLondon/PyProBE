"""A module for the Procedure class."""
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import polars as pl
import yaml

from pyprobe.rawdata import RawData


def filter_numerical(
    _data: pl.LazyFrame | pl.DataFrame,
    column: str,
    indices: Tuple[Union[int, range], ...],
) -> pl.LazyFrame:
    """Filter a LazyFrame by a numerical condition.

    Args:
        _data (pl.LazyFrame | pl.DataFrame): A LazyFrame object.
        column (str): The column to filter on.
        indices (Tuple[Union[int, range], ...]): A tuple of index
            values to filter by.
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
            return _data.filter(pl.col(column).rank("dense").is_in(index_list))
        elif all(item < 0 for item in index_list):
            index_list = [item * -1 for item in index_list]
            return _data.filter(
                pl.col(column).rank("dense", descending=True).is_in(index_list)
            )
        else:
            raise ValueError("Indices must be all positive or all negative.")
    else:
        return _data


def step(
    self: Union["Procedure", "Experiment", "Cycle"],
    *step_numbers: Union[int, range],
    condition: Optional[pl.Expr] = None,
) -> RawData:
    """Return a step object from the cycle.

    Args:
        step_number (int | range): Variable-length argument list of
            step numbers or a range object.

    Returns:
        RawData: A step object from the cycle.
    """
    if condition is not None:
        _data = filter_numerical(self._data.filter(condition), "Event", step_numbers)
    else:
        _data = filter_numerical(self._data, "Event", step_numbers)
    return RawData(_data, self.info)


def cycle(self: "FilteredDataType", *cycle_numbers: Union[int]) -> "Cycle":
    """Return a cycle object from the experiment.

    Args:
        cycle_number (int | range): Variable-length argument list of
            cycle numbers or a range object.

    Returns:
        Filter: A filter object for the specified cycles.
    """
    lf_filtered = filter_numerical(self._data, "Cycle", cycle_numbers)

    return Cycle(lf_filtered, self.info)


def charge(
    self: Union["Procedure", "Experiment", "Cycle"], *charge_numbers: Union[int, range]
) -> RawData:
    """Return a charge step object from the cycle.

    Args:
        charge_number (int | range): Variable-length argument list of
            charge numbers or a range object.

    Returns:
        RawData: A charge step object from the cycle.
    """
    condition = pl.col("Current [A]") > 0
    return self.step(*charge_numbers, condition=condition)


def discharge(
    self: Union["Procedure", "Experiment", "Cycle"],
    *discharge_numbers: Union[int, range],
) -> RawData:
    """Return a discharge step object from the cycle.

    Args:
        discharge_number (int | range): Variable-length argument list of
            discharge numbers or a range object.

    Returns:
        RawData: A discharge step object from the cycle.
    """
    condition = pl.col("Current [A]") < 0
    return self.step(*discharge_numbers, condition=condition)


def chargeordischarge(
    self: Union["Procedure", "Experiment", "Cycle"],
    *chargeordischarge_numbers: Union[int, range],
) -> RawData:
    """Return a charge or discharge step object from the cycle.

    Args:
        chargeordischarge_number (int | range): Variable-length argument list of
            charge or discharge numbers or a range object.

    Returns:
        RawData: A charge or discharge step object from the cycle.
    """
    condition = pl.col("Current [A]") != 0
    return self.step(*chargeordischarge_numbers, condition=condition)


def rest(
    self: Union["Procedure", "Experiment", "Cycle"], *rest_numbers: Union[int, range]
) -> RawData:
    """Return a rest step object from the cycle.

    Args:
        rest_number (int | range): Variable-length argument list of rest
            numbers or a range object.

    Returns:
        RawData: A rest step object from the cycle.
    """
    condition = pl.col("Current [A]") == 0
    return self.step(*rest_numbers, condition=condition)


def constant_current(
    self: Union["Procedure", "Experiment", "Cycle"],
    *constant_current_numbers: Union[int, range],
) -> RawData:
    """Return a constant current step object.

    Args:
        constant_current_numbers (int | range): Variable-length argument list of
            constant current numbers or a range object.

    Returns:
        RawData: A constant current step object.
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
    return self.step(*constant_current_numbers, condition=condition)


def constant_voltage(
    self: Union["Procedure", "Experiment", "Cycle"],
    *constant_voltage_numbers: Union[int, range],
) -> RawData:
    """Return a constant voltage step object.

    Args:
        constant_current_numbers (int | range): Variable-length argument list of
            constant voltage numbers or a range object.

    Returns:
        RawData: A constant voltage step object.
    """
    condition = (
        pl.col("Voltage [V]").abs()
        > 0.999 * pl.col("Voltage [V]").abs().round_sig_figs(4).mode()
    ) & (
        pl.col("Voltage [V]").abs()
        < 1.001 * pl.col("Voltage [V]").abs().round_sig_figs(4).mode()
    )
    return self.step(*constant_voltage_numbers, condition=condition)


class Procedure(RawData):
    """A class for a procedure in a battery experiment."""

    def __init__(
        self,
        data_path: str,
        info: Dict[str, str | int | float],
        custom_readme_name: Optional[str] = None,
    ) -> None:
        """Create a procedure class.

        Args:
            data_path (str): The path to the data parquet file.
            info (Dict[str, str | int | float]): A dict containing test info.
            custom_readme_name (str, optional): The name of the custom README file.
                Defaults to None.
        Filtering attributes:
            step (Callable[..., int]):
                A method to return a step object from the procedure. See `step`.
            cycle (Callable[..., int]):
                A method to return a cycle object from the procedure.
                See `cycle`.
            charge (Callable[..., int]):
                A method to return a charge step object from the procedure.
                See `charge`.
            discharge (Callable[..., int]):
                A method to return a discharge step object from the procedure.
                See `discharge`.
            chargeordischarge (Callable[..., int]):
                A method to return a charge or discharge step object from the procedure.
                See `chargeordischarge`.
            rest (Callable[..., int]):
                A method to return a rest step object from the procedure.
                See `rest`.
            constant_current (Callable[..., int]):
                A method to return a constant current step object.
                See `constant_current`.
            constant_voltage (Callable[..., int]):
                A method to return a constant voltage step object.
                See `constant_voltage`.
        """
        _data = pl.scan_parquet(data_path)
        data_folder = os.path.dirname(data_path)
        if custom_readme_name:
            readme_path = os.path.join(data_folder, f"{custom_readme_name}.yaml")
        else:
            readme_path = os.path.join(data_folder, "README.yaml")
        (
            self.titles,
            self.steps_idx,
        ) = self.process_readme(readme_path)
        super().__init__(_data, info)
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

    def experiment(self, *experiment_names: str) -> RawData:
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
        lf_filtered = self._data.filter(conditions)
        return Experiment(lf_filtered, self.info)

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

    @staticmethod
    def process_readme(
        readme_path: str,
    ) -> tuple[Dict[str, str], List[List[int]]]:
        """Function to process the README.yaml file.

        Args:
            readme_path (str): The path to the README.yaml file.

        Returns:
            Tuple[Dict[str, str], List[int], List[int]]:
                - dict: The titles of the experiments inside a procedure.
                    Format {title: experiment type}.
                - list: The cycle numbers inside the procedure.
                - list: The step numbers inside the procedure.
        """
        with open(readme_path, "r") as file:
            readme_dict = yaml.safe_load(file)

        titles = {
            experiment: readme_dict[experiment]["Type"] for experiment in readme_dict
        }

        max_step = 0
        steps: List[List[int]] = []
        for experiment in readme_dict:
            if "Step Numbers" in readme_dict[experiment]:
                step_list = readme_dict[experiment]["Step Numbers"]
            elif "Total Steps" in readme_dict[experiment]:
                step_list = list(range(readme_dict[experiment]["Total Steps"]))
                step_list = [x + max_step + 1 for x in step_list]
            else:
                step_list = list(range(len(readme_dict[experiment]["Steps"])))
                step_list = [x + max_step + 1 for x in step_list]
            max_step = step_list[-1]
            steps.append(step_list)

        return titles, steps

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


class Experiment(RawData):
    """A class for an experiment in a battery procedure.

    Args:
        _data (pl.LazyFrame | pl.DataFrame): The data for the experiment.
        info (Dict[str, str | int | float]): A dict containing test info.

    Filtering attributes:
        step (Callable[..., int]):
            A method to return a step object from the experiment. See `step`.
        cycle (Callable[..., int]):
            A method to return a cycle object from the experiment. See `cycle`.
        charge (Callable[..., int]):
            A method to return a charge step object from the experiment. See `charge`.
        discharge (Callable[..., int]):
            A method to return a discharge step object from the experiment.
            See `discharge`.
        chargeordischarge (Callable[..., int]):
            A method to return a charge or discharge step object from the experiment.
            See `chargeordischarge`.
        rest (Callable[..., int]):
            A method to return a rest step object from the experiment. See `rest`.
        constant_current (Callable[..., int]):
            A method to return a constant current step object. See `constant_current`.
        constant_voltage (Callable[..., int]):
            A method to return a constant voltage step object. See `constant_voltage`.
    """

    def __init__(
        self,
        _data: pl.LazyFrame | pl.DataFrame,
        info: Dict[str, str | int | float],
    ) -> None:
        """Create an experiment class.

        Args:
            data (pl.LazyFrame | pl.DataFrame): The data for the experiment.
            info (Dict[str, str | int | float]): A dict containing test info.
        """
        super().__init__(_data, info)

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


class Cycle(RawData):
    """A class for a cycle in a battery experiment.

    Args:
        _data (pl.LazyFrame | pl.DataFrame): The data for the cycle.
        info (Dict[str, str | int | float]): A dict containing test info.

    Filtering attributes:
        _data (pl.LazyFrame | pl.DataFrame): The data for the cycle.
        info (Dict[str, str | int | float]): A dict containing
        step (Callable[..., int]):
            A method to return a step object from the cycle. See `step`.
        charge (Callable[..., int]):
            A method to return a charge step object from the cycle. See `charge`.
        discharge (Callable[..., int]):
            A method to return a discharge step object from the cycle.
            See `discharge`.
        chargeordischarge (Callable[..., int]):
            A method to return a charge or discharge step object from the cycle.
            See `chargeordischarge`.
        rest (Callable[..., int]):
            A method to return a rest step object from the cycle. See `rest`.
        constant_current (Callable[..., int]):
            A method to return a constant current step object.
            See `constant_current`.
        constant_voltage (Callable[..., int]):
            A method to return a constant voltage step object.
            See `constant_voltage`.
    """

    def __init__(
        self,
        _data: pl.LazyFrame | pl.DataFrame,
        info: Dict[str, str | int | float],
    ) -> None:
        """Create a cycle class.

        Args:
            data (pl.LazyFrame | pl.DataFrame): The data for the cycle.
            info (Dict[str, str | int | float]): A dict containing test info.
        """
        super().__init__(_data, info)

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


FilteredDataType = Union[Procedure, Experiment, Cycle]
