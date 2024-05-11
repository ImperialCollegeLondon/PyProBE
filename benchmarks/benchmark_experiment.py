"""Benchmark functions for the Experiment class."""
import benchmarks.benchmark_functions as benchmark_functions


def time_return_experiment() -> None:
    """Benchmark the creation of an Experiment object."""
    benchmark_functions.return_cycling_experiment()


def time_return_experiment_data() -> None:
    """Benchmark the return of the data attribute of an Experiment object."""
    cell = benchmark_functions.return_procedure()
    cell.experiment("Break-in Cycles").data


def time_return_experiment_from_initialized_procedure() -> None:
    """Benchmark Experiment creation from initialized procedure."""
    cell = benchmark_functions.return_procedure()
    cell.data
    cell.experiment("Break-in Cycles").data
