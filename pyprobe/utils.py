"""A collection of utility functions for PyProBE."""

import functools
import warnings
from typing import Any, Protocol

from pydantic import ValidationError


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


class PyProBEValidationError(Exception):
    """Custom exception for PyProBE validation errors."""

    pass


def catch_pydantic_validation(func: Any) -> Any:
    """A decorator that wraps pydantic ValidationError to raise a custom exception."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except ValidationError as e:
            error_message = "Validation error, invalid input provided to"
            " {func.__module__}.{func.__name__}\n"
            for error in e.errors():
                error_message += f"\n{error['loc'][0]}: {error['msg']}"
            raise PyProBEValidationError(error_message) from None

    return wrapper


def deprecated(*, reason: str, version: str, plain_reason: str | None = None) -> Any:
    """A decorator to mark a function as deprecated.

    Args:
        reason (str): The reason for deprecation, using RST formatting for docs.
        version (str): The version in which the function was deprecated.
        plain_reason (str, optional):
            The plain text reason for deprecation. Defaults to None.
    """

    def decorator(func: Any) -> Any:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            warnings.warn(
                plain_reason if plain_reason else reason,
                DeprecationWarning,
                stacklevel=2,
            )
            return func(*args, **kwargs)

        # Prepend a Sphinx deprecation note to the original docstring.
        deprecation_note = f".. deprecated:: {version} {reason}\n\n"
        wrapper.__doc__ = deprecation_note
        return wrapper

    return decorator
