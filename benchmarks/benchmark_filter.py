"""Benchmark functions for the Filter class."""

import benchmarks.benchmark_functions as benchmark_functions


def time_return_cycle_data() -> None:
    """Benchmark the return of the data attribute of a cycle."""
    cell = benchmark_functions.add_procedure_from_parquet()
    cell.procedure["sample procedure"].experiment("Discharge Pulses").cycle(0).data


def time_return_step_data() -> None:
    """Benchmark the return of the data attribute of a step."""
    cell = benchmark_functions.add_procedure_from_parquet()
    cell.procedure["sample procedure"].experiment("Discharge Pulses").step(1).data


def time_return_step_from_cycle_data() -> None:
    """Benchmark the return of the data attribute of a step from a cycle."""
    cell = benchmark_functions.add_procedure_from_parquet()
    cell.procedure["sample procedure"].experiment("Discharge Pulses").cycle(0).step(
        1
    ).data


def time_return_discharge_data() -> None:
    """Benchmark the return of the data attribute of a discharge."""
    cell = benchmark_functions.add_procedure_from_parquet()
    cell.procedure["sample procedure"].experiment("Discharge Pulses").discharge(0).data


def time_return_cycle_from_initialized_experiment_data() -> None:
    """Benchmark returning a cycle from an initialized Experiment object."""
    cell = benchmark_functions.add_procedure_from_parquet()
    cell.procedure["sample procedure"].experiment("Discharge Pulses").data
    cell.procedure["sample procedure"].experiment("Discharge Pulses").cycle(0).data
