"""Benchmark functions for the Procedure class."""
import benchmarks.benchmark_functions as benchmark_functions


def time_return_procedure() -> None:
    """Benchmark the creation of a Procedure object."""
    benchmark_functions.return_procedure()


def time_return_procedure_data() -> None:
    """Benchmark the return of the data attribute of a Procedure object."""
    cell = benchmark_functions.add_data_from_parquet()
    cell.procedure["sample procedure"].data


def time_return_procedure_data_twice() -> None:
    """Benchmark the return of the data attribute of a Procedure object twice."""
    cell = benchmark_functions.add_data_from_parquet()
    cell.procedure["sample procedure"].data
    cell.procedure["sample procedure"].data
