"""A module for the filtering classes."""

import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import polars as pl

from pyprobe import utils
from pyprobe.columns import BDF, ColumnDict
from pyprobe.rawdata import RawData

if TYPE_CHECKING:
    from pyprobe.pyprobe_types import (
        ExperimentOrCycleType,
        FilterToCycleType,
    )


from loguru import logger


def _filter_numerical(
    dataframe: pl.LazyFrame | pl.DataFrame,
    column: str | pl.Expr,
    indices: tuple[int | range, ...],
) -> pl.LazyFrame | pl.DataFrame:
    """Filter a polars LazyFrame or DataFrame by a numerical condition.

    Args:
        dataframe: A LazyFrame or DataFrame to filter.
        column: The column name or expression to filter on.
        indices: A tuple of index values to filter by.

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
        col_expr = pl.col(column) if isinstance(column, str) else column
        if all(item >= 0 for item in index_list):
            index_list = [item + 1 for item in index_list]
            return dataframe.filter(col_expr.rank("dense").is_in(index_list))
        elif all(item < 0 for item in index_list):
            index_list = [item * -1 for item in index_list]
            return dataframe.filter(
                col_expr.rank("dense", descending=True).is_in(index_list),
            )
        else:
            error_msg = "Indices must be all positive or all negative."
            logger.error(error_msg)
            raise ValueError(error_msg)
    else:
        return dataframe


def _step(
    filtered_object: "FilterToCycleType",
    *step_numbers: int | range,
    condition: pl.Expr | None = None,
) -> "Step":
    """Return a step object. Filters to a numerical condition on the Step Index column.

    Args:
        filtered_object: A filter object that this method is called on.
        step_numbers: Variable-length argument list of step indices or a range object.
        condition: A polars expression to filter the step before applying the numerical
            filter. Defaults to None.

    Returns:
        Step: A step object.
    """
    step_index_expr = filtered_object.columns.resolve(BDF.STEP_COUNT)
    if condition is not None:
        lf = _filter_numerical(
            filtered_object.lf.filter(condition),
            step_index_expr,
            step_numbers,
        )
    else:
        lf = _filter_numerical(
            filtered_object.lf,
            step_index_expr,
            step_numbers,
        )
    return Step(
        lf=lf,
        metadata=filtered_object.metadata,
        column_definitions=filtered_object.column_definitions,
        step_descriptions=filtered_object.step_descriptions,
    )


def get_cycle_column(
    filtered_object: "FilterToCycleType",
) -> pl.DataFrame | pl.LazyFrame:
    """Add a Cycle Count column to the data.

    If cycle details have been provided in the README, the cycle column will be
    created by checking for the last step of the cycle. For nested cycles, the
    "outer" cycle will be created first; subsequent filtering with the cycle method
    allows for filtering on the "inner" cycles.

    If no cycle details have been provided, the cycle column will be inferred from
    a decrease in the step count.

    Args:
        filtered_object: The experiment or cycle object.

    Returns:
        pl.DataFrame | pl.LazyFrame: The data with a cycle count column.
    """
    step_expr = filtered_object.columns.resolve(BDF.STEP_INDEX)
    cycle_col_name = BDF.CYCLE_COUNT.name
    if len(filtered_object.cycle_info) > 0:
        cycle_ends = (
            (
                (step_expr.shift() == filtered_object.cycle_info[0][1])
                & (step_expr != filtered_object.cycle_info[0][1])
            )
            .fill_null(strategy="zero")
            .cast(pl.Int16)
        )
        cycle_column = (
            cycle_ends.cum_sum().fill_null(strategy="zero").alias(cycle_col_name)
        )
    else:
        warnings.warn(
            "No cycle information provided. Cycles will be inferred from the step "
            "numbers.",
        )
        cycle_column = (
            (step_expr.cast(pl.Int64) - step_expr.cast(pl.Int64).shift() < 0)
            .fill_null(strategy="zero")
            .cum_sum()
            .alias(cycle_col_name)
        )
    return filtered_object.lf.with_columns(cycle_column)


def _cycle(filtered_object: "ExperimentOrCycleType", *cycle_numbers: int) -> "Cycle":
    """Return a cycle object. Filters on the Cycle Count column.

    Args:
        filtered_object: A filter object that this method is called on.
        cycle_numbers: Variable-length argument list of cycle indices or a range object.

    Returns:
        Cycle: A cycle object.
    """
    df = get_cycle_column(filtered_object)
    if len(filtered_object.cycle_info) > 1:
        next_cycle_info = filtered_object.cycle_info[1:]
    else:
        next_cycle_info = []

    df_column_set = ColumnDict(df.collect_schema().names())
    cycle_expr = df_column_set.resolve(BDF.CYCLE_COUNT)
    lf_filtered = _filter_numerical(df, cycle_expr, cycle_numbers)

    return Cycle(
        lf=lf_filtered,
        metadata=filtered_object.metadata,
        column_definitions=filtered_object.column_definitions,
        step_descriptions=filtered_object.step_descriptions,
        cycle_info=next_cycle_info,
    )


def _charge(
    filtered_object: "FilterToCycleType",
    *charge_numbers: int | range,
) -> "Step":
    """Return a charge step.

    Args:
        filtered_object: A filter object that this method is called on.
        charge_numbers: Variable-length argument list of charge indices or a range
            object.

    Returns:
        Step: A charge step object.
    """
    current_expr = filtered_object.columns.resolve(BDF.CURRENT_AMPERE)
    condition = current_expr > current_expr.abs().max() / 10e4
    return filtered_object.step(*charge_numbers, condition=condition)


def _discharge(
    filtered_object: "FilterToCycleType",
    *discharge_numbers: int | range,
) -> "Step":
    """Return a discharge step.

    Args:
        filtered_object: A filter object that this method is called on.
        discharge_numbers: Variable-length argument list of discharge indices or a range
            object.

    Returns:
        Step: A discharge step object.
    """
    current_expr = filtered_object.columns.resolve(BDF.CURRENT_AMPERE)
    condition = current_expr < -current_expr.abs().max() / 10e4
    return filtered_object.step(*discharge_numbers, condition=condition)


def _chargeordischarge(
    filtered_object: "FilterToCycleType",
    *chargeordischarge_numbers: int | range,
) -> "Step":
    """Return a charge or discharge step.

    Args:
        filtered_object: A filter object that this method is called on.
        chargeordischarge_numbers: Variable-length argument list of charge or discharge
            indices or a range object.

    Returns:
        Step: A charge or discharge step object.
    """
    current_expr = filtered_object.columns.resolve(BDF.CURRENT_AMPERE)
    charge_condition = current_expr > current_expr.abs().max() / 10e4
    discharge_condition = current_expr < -current_expr.abs().max() / 10e4
    condition = charge_condition | discharge_condition
    return filtered_object.step(*chargeordischarge_numbers, condition=condition)


def _rest(filtered_object: "FilterToCycleType", *rest_numbers: int | range) -> "Step":
    """Return a rest step object.

    Args:
        filtered_object: A filter object that this method is called on.
        rest_numbers: Variable-length argument list of rest indices or a range object.

    Returns:
        Step: A rest step object.
    """
    current_expr = filtered_object.columns.resolve(BDF.CURRENT_AMPERE)
    condition = current_expr == 0
    return filtered_object.step(*rest_numbers, condition=condition)


def _constant_current(
    filtered_object: "FilterToCycleType",
    *constant_current_numbers: int | range,
) -> "Step":
    """Return a constant current step object.

    Args:
        filtered_object: A filter object that this method is called on.
        constant_current_numbers: Variable-length argument list of constant current
            indices or a range object.

    Returns:
        Step: A constant current step object.
    """
    current_expr = filtered_object.columns.resolve(BDF.CURRENT_AMPERE)
    condition = (
        (current_expr != 0)
        & (current_expr.abs() > 0.999 * current_expr.abs().round_sig_figs(4).mode())
        & (current_expr.abs() < 1.001 * current_expr.abs().round_sig_figs(4).mode())
    )
    return filtered_object.step(*constant_current_numbers, condition=condition)


def _constant_voltage(
    filtered_object: "FilterToCycleType",
    *constant_voltage_numbers: int | range,
) -> "Step":
    """Return a constant voltage step object.

    Args:
        filtered_object: A filter object that this method is called on.
        *constant_voltage_numbers: Variable-length argument list of constant voltage
            indices or a range object.

    Returns:
        Step: A constant voltage step object.
    """
    voltage_expr = filtered_object.columns.resolve(BDF.VOLTAGE_VOLT)
    condition = (
        voltage_expr.abs() > 0.999 * voltage_expr.abs().round_sig_figs(4).mode()
    ) & (voltage_expr.abs() < 1.001 * voltage_expr.abs().round_sig_figs(4).mode())
    return filtered_object.step(*constant_voltage_numbers, condition=condition)


class Procedure(RawData):
    """A class for a procedure in a battery experiment."""

    def __init__(
        self,
        lf: pl.LazyFrame | pl.DataFrame | str,
        metadata: dict[str, Any | None],
        readme_dict: dict[str, dict[str, list[str | int | tuple[int, int, int]]]],
        column_definitions: dict[str, str] | None = None,
        step_descriptions: dict[str, list[str | int | None]] | None = None,
        cycle_info: list[tuple[int, int, int]] | None = None,
    ) -> None:
        """Initialize a procedure with README-derived experiment metadata.

        Args:
            lf: A LazyFrame, DataFrame, or a path to a parquet file.
            metadata: Dictionary containing metadata about the procedure and
                data source.
            readme_dict: Experiment definitions from README.
            column_definitions: Column descriptions.
            step_descriptions: Step-by-step descriptions.
            cycle_info: Cycle boundary information.
        """
        super().__init__(
            lf=lf,
            metadata=metadata,
            column_definitions=column_definitions,
            step_descriptions=step_descriptions,
        )
        self.readme_dict = readme_dict
        self.cycle_info = cycle_info.copy() if cycle_info is not None else []
        self._populate_step_descriptions()

    def _populate_step_descriptions(self) -> None:
        """Populate step_descriptions from readme_dict."""
        self.step_descriptions = {"Step": [], "Description": []}
        for experiment in self.readme_dict:
            steps = cast(list[int], self.readme_dict[experiment]["Steps"])
            descriptions: list[str | None] = [None] * len(steps)
            if "Step Descriptions" in self.readme_dict[experiment]:
                descriptions = cast(
                    list[str | None],
                    self.readme_dict[experiment]["Step Descriptions"],
                )
            self.step_descriptions["Step"].extend(steps)
            self.step_descriptions["Description"].extend(descriptions)

    @classmethod
    def load(
        cls,
        parquet_path: str | Path,
        readme_path: str | Path | None = None,
        metadata_prefer: Literal["parquet", "json"] = "parquet",
    ) -> "Procedure":
        """Load a Procedure from a processed .parquet file.

        Reads BDF-normalised data and any embedded metadata from *parquet_path*.
        When *readme_path* is ``None``, the method auto-guesses by looking for
        ``README.yaml`` in the same directory as *parquet_path*. If found it is
        used; if not found a log message is emitted and the Procedure is returned
        without experiment definitions.

        Args:
            parquet_path: Path to a ``.parquet`` file (e.g. from
                :func:`~pyprobe.io.process_cycler`).
            readme_path: Explicit path to a README.yaml for experiment definitions.
                When ``None`` (default), the parent directory of *parquet_path* is
                checked automatically.
            metadata_prefer: Whether to prefer the Parquet footer (``"parquet"``,
                default) or a JSON sidecar (``"json"``) when both metadata sources
                exist.

        Returns:
            Procedure with BDF-format columns, metadata, and
            optional experiment definitions from README.yaml.

        Raises:
            FileNotFoundError: If *parquet_path* does not exist.

        Example:
            Load a procedure from a processed parquet file::

                from pyprobe.io import process_cycler
                from pyprobe.filters import Procedure

                path = process_cycler("data.xlsx")
                procedure = Procedure.load(path)
                procedure = Procedure.load(path, readme_path="README.yaml")
        """
        from pyprobe.io import read_metadata
        from pyprobe.readme_processor import process_readme

        parquet_path = Path(parquet_path)
        if not parquet_path.exists():
            raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

        lf = pl.scan_parquet(parquet_path)
        parquet_metadata = read_metadata(parquet_path, prefer=metadata_prefer)

        if readme_path is None:
            candidate = parquet_path.parent / "README.yaml"
            if candidate.exists():
                readme_path = candidate
            else:
                logger.info(
                    "No README.yaml found in '{}'; proceeding without "
                    "experiment definitions.",
                    parquet_path.parent,
                )

        readme_dict: dict[str, dict[str, Any]] = {}
        if readme_path is not None:
            rp = Path(readme_path)
            if rp.exists():
                readme_dict = process_readme(str(rp)).experiment_dict
            else:
                logger.warning("README path provided but not found: {}", readme_path)

        return cls(lf=lf, metadata=parquet_metadata, readme_dict=readme_dict)

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
            experiment_names: Variable-length argument list of experiment names.

        Returns:
            Experiment: An experiment object from the procedure.
        """
        steps_idx = []
        for experiment_name in experiment_names:
            if experiment_name not in self.experiment_names:
                error_msg = f"{experiment_name} not in procedure."
                logger.error(error_msg)
                raise ValueError(error_msg)
            steps_idx.append(self.readme_dict[experiment_name]["Steps"])
        flattened_steps = utils.flatten_list(steps_idx)
        conditions = [
            pl.col(BDF.STEP_INDEX.name).is_in(flattened_steps),
        ]
        lf_filtered = self.lf.filter(conditions)
        cycles_list: list[tuple[int, int, int]] = []
        if len(experiment_names) > 1:
            warnings.warn(
                "Multiple experiments selected. Cycles will be inferred from "
                "the step numbers.",
            )
        elif "Cycles" in self.readme_dict[experiment_names[0]]:
            cycles_list = self.readme_dict[experiment_names[0]]["Cycles"]  # type: ignore[assignment]

        return Experiment(
            lf=lf_filtered,
            metadata=self.metadata,
            column_definitions=self.column_definitions,
            step_descriptions=self.step_descriptions,
            cycle_info=cycles_list,
        )

    def remove_experiment(self, *experiment_names: str) -> None:
        """Remove an experiment from the procedure.

        Args:
            experiment_names: Variable-length argument list of experiment names.
        """
        steps_idx = []
        for experiment_name in experiment_names:
            if experiment_name not in self.experiment_names:
                error_msg = f"{experiment_name} not in procedure."
                logger.error(error_msg)
                raise ValueError(error_msg)
            steps_idx.append(self.readme_dict[experiment_name]["Steps"])
        flattened_steps = utils.flatten_list(steps_idx)
        conditions = [
            pl.col(BDF.STEP_INDEX.name).is_in(flattened_steps).not_(),
        ]
        for experiment_name in experiment_names:
            self.readme_dict.pop(experiment_name)
        self._populate_step_descriptions()
        self.lf = self.lf.filter(conditions)

    @property
    def experiment_names(self) -> list[str]:
        """Return the names of the experiments in the procedure.

        Returns:
            List[str]: The names of the experiments in the procedure.
        """
        return list(self.readme_dict.keys())

    @utils.deprecated(
        reason="Use add_data instead.",
        version="2.3.1",
    )
    def add_external_data(
        self,
        filepath: str,
        importing_columns: list[str] | dict[str, str],
        date_column_name: str = "Date",
    ) -> None:
        """Add data from another source to the procedure.

        Args:
            filepath: The path to the external file.
            importing_columns: The columns to import from the external file.
            date_column_name: The name of the date column in the external data.
        """
        raise NotImplementedError(
            "add_external_data is deprecated. Use add_data instead."
        )


class Experiment(RawData):
    """A class for an experiment in a battery experimental procedure."""

    cycle_info: list[tuple[int, int, int]] = []
    """A list of tuples representing the cycle information from the README yaml file.

    The tuple format is
    :code:`(start step (inclusive), end step (inclusive), cycle count)`.
    """

    def __init__(
        self,
        lf: pl.LazyFrame | pl.DataFrame | str,
        metadata: dict[str, Any | None],
        column_definitions: dict[str, str] | None = None,
        step_descriptions: dict[str, list[str | int | None]] | None = None,
        cycle_info: list[tuple[int, int, int]] | None = None,
    ) -> None:
        """Initialize an experiment view with optional cycle metadata.

        Args:
            lf: A LazyFrame, DataFrame, or a path to a parquet file.
            metadata: Dictionary containing metadata about the experiment and
                data source.
            column_definitions: Column descriptions.
            step_descriptions: Step-by-step descriptions.
            cycle_info: Cycle boundary information.
        """
        super().__init__(
            lf=lf,
            metadata=metadata,
            column_definitions=column_definitions,
            step_descriptions=step_descriptions,
        )
        self.cycle_info = cycle_info.copy() if cycle_info is not None else []

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

    cycle_info: list[tuple[int, int, int]] = []
    """A list of tuples representing the cycle information from the README yaml file.

    The tuple format is
    :code:`(start step (inclusive), end step (inclusive), cycle count)`.
    """

    def __init__(
        self,
        lf: pl.LazyFrame | pl.DataFrame | str,
        metadata: dict[str, Any | None],
        column_definitions: dict[str, str] | None = None,
        step_descriptions: dict[str, list[str | int | None]] | None = None,
        cycle_info: list[tuple[int, int, int]] | None = None,
    ) -> None:
        """Initialize a cycle view with optional nested cycle metadata.

        Args:
            lf: A LazyFrame, DataFrame, or a path to a parquet file.
            metadata: Dictionary containing metadata about the cycle and data source.
            column_definitions: Column descriptions.
            step_descriptions: Step-by-step descriptions.
            cycle_info: Cycle boundary information.
        """
        super().__init__(
            lf=lf,
            metadata=metadata,
            column_definitions=column_definitions,
            step_descriptions=step_descriptions,
        )
        self.cycle_info = cycle_info.copy() if cycle_info is not None else []

    step = _step
    charge = _charge
    discharge = _discharge
    chargeordischarge = _chargeordischarge
    rest = _rest
    constant_current = _constant_current
    constant_voltage = _constant_voltage


class Step(RawData):
    """A class for a step in a battery experimental procedure."""

    def __init__(
        self,
        lf: pl.LazyFrame | pl.DataFrame | str,
        metadata: dict[str, Any | None],
        column_definitions: dict[str, str] | None = None,
        step_descriptions: dict[str, list[str | int | None]] | None = None,
    ) -> None:
        """Initialize a step view.

        Args:
            lf: A LazyFrame, DataFrame, or a path to a parquet file.
            metadata: Dictionary containing metadata about the step and data source.
            column_definitions: Column descriptions.
            step_descriptions: Step-by-step descriptions.
        """
        super().__init__(
            lf=lf,
            metadata=metadata,
            column_definitions=column_definitions,
            step_descriptions=step_descriptions,
        )

    step = _step
    constant_current = _constant_current
    constant_voltage = _constant_voltage
