"""Benchmark functions for the Cell class."""
import benchmarks.benchmark_functions as benchmark_functions


def time_create_cell() -> None:
    """Benchmark the creation of a Cell object."""
    benchmark_functions.create_cell()


def time_add_data_from_parquet() -> None:
    """Benchmark adding data to a Cell object from a parquet file."""
    benchmark_functions.add_data_from_parquet()
