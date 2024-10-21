"""A collection of utility functions for PyProBE."""
from typing import Any, List


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
