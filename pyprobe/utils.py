"""A collection of utility functions for PyProBE."""

from typing import Any, Protocol


def flatten_list(lst: int | list[Any]) -> list[int]:
    """Flatten a list of lists into a single list.

    Args:
        lst (list): The list of lists to flatten.

    Returns:
        list: The flattened list.
    """
    if not isinstance(lst, list):
        return [lst]
    else:
        return [item for sublist in lst for item in flatten_list(sublist)]


class PyBaMMSolution(Protocol):
    """Protocol defining required PyBaMM Solution interface."""

    def get_data_dict(
        self,
        variables: list[str] | None = None,
        short_names: dict[str, str] | None = None,
        cycles_and_steps: bool = True,
    ) -> dict[str, Any]:
        """Get solution data as dictionary."""
