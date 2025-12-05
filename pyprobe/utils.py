"""A collection of utility functions for PyProBE."""

import functools
import sys
from typing import Any, Literal, Protocol

from loguru import logger
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
            error_message = (
                f"Validation error, invalid input provided to "
                f"{func.__module__}.{func.__name__}\n"
            )
            for error in e.errors():
                loc_str = ".".join(str(loc) for loc in error["loc"])
                error_message += f"\n{loc_str}: {error['msg']}"
            raise PyProBEValidationError(error_message) from None

    return wrapper


def deprecated(*, reason: str, version: str, plain_reason: str | None = None) -> Any:
    """A decorator to mark a function or property as deprecated.

    Args:
        reason (str): The reason for deprecation, using RST formatting for docs.
        version (str): The version in which the function was deprecated.
        plain_reason (str, optional):
            The plain text reason for deprecation. Defaults to None.
    """

    def decorator(func: Any) -> Any:
        # Handle property objects
        if isinstance(func, property):
            warning_msg = plain_reason if plain_reason else reason

            # Wrap the getter
            def getter(self: Any) -> Any:
                logger.warning(
                    "Deprecation Warning: " + warning_msg,
                )
                if func.fget is None:
                    raise AttributeError("unreadable attribute")
                return func.fget(self)

            # Wrap the setter if it exists
            def make_setter(fset: Any) -> Any:
                def set_func(self: Any, value: Any) -> None:
                    logger.warning(
                        "Deprecation Warning: " + warning_msg,
                    )
                    fset(self, value)

                return set_func

            setter = make_setter(func.fset) if func.fset is not None else None

            # Create new property with wrapped getter/setter
            deprecation_note = f".. deprecated:: {version} {reason}\n\n"
            new_prop = property(getter, setter, func.fdel, func.__doc__)
            new_prop.__doc__ = deprecation_note + (func.__doc__ or "")
            return new_prop

        # Handle regular functions
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            logger.warning(
                "Deprecation Warning: " + (plain_reason if plain_reason else reason),
            )
            return func(*args, **kwargs)

        # Prepend a Sphinx deprecation note to the original docstring.
        deprecation_note = f".. deprecated:: {version} {reason}\n\n"
        wrapper.__doc__ = deprecation_note
        return wrapper

    return decorator


def set_log_level(
    level: Literal[
        "TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"
    ] = "ERROR",
) -> None:
    """Set PyProBE's logging level to send to stderr.

    Args:
        level:
            The logging level to display. Options are:
            - TRACE: Show all messages, including trace messages.
            - DEBUG: Show all debugging information.
            - INFO: Show detailed information.
            - SUCCESS: Show success messages.
            - WARNING: Show warning messages.
            - ERROR: Show error messages (default).
            - CRITICAL: Show critical error messages only.

    Example:

    .. code-block:: python

        import pyprobe
        pyprobe.set_log_level("INFO")  # Show more detailed logs
        pyprobe.set_log_level("DEBUG") # Show all debugging information
        pyprobe.set_log_level("ERROR") # Default - show only errors
    """
    logger.remove()  # Remove all handlers
    fmt = (
        "<green>{time:HH:mm:ss}</green> | <level>{level}</level> | "
        "<cyan>{name}:{function}:{line}</cyan> - <level>{message}</level> | "
        "Context: {extra}"
    )
    logger.add(sys.stderr, level=level.upper(), format=fmt, colorize=True)
