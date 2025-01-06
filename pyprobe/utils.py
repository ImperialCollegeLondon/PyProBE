"""A collection of utility functions for PyProBE."""

from typing import Any, Dict, List, Optional, Protocol


def flatten_list(lst: int | List[Any]) -> List[int]:
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
        variables: Optional[List[str]] = None,
        short_names: Optional[Dict[str, str]] = None,
        cycles_and_steps: bool = True,
    ) -> Dict[str, Any]:
        """Get solution data as dictionary."""
