"""Helpers for building and reading test metadata."""

from collections.abc import Mapping
from typing import Any


def build_metadata(**keys: Any) -> dict[str, Any]:
    """Build a metadata dictionary from keyword arguments.

    This helper abstracts metadata construction for tests, allowing future
    changes to metadata representation without updating every test.

    Args:
        **keys: Key-value pairs to include in the metadata.

    Returns:
        dict: A metadata dictionary.
    """
    return dict(keys)


def read_extras(obj: Any) -> Mapping[str, Any]:
    """Read the extras from a metadata-bearing object.

    This helper abstracts metadata access for tests, allowing future
    changes to metadata representation without updating every test.

    Args:
        obj: An object with metadata (e.g., Table, Result, RawData).

    Returns:
        Mapping: The extras mapping.

    Raises:
        TypeError: Where metadata is neither a Mapping nor has an extras attribute.
    """
    metadata = obj.metadata

    # If metadata is a Mapping (dict), return it directly
    if isinstance(metadata, Mapping):
        return metadata

    # If metadata has an extras attribute, return it or empty dict
    try:
        extras = metadata.extras
        return extras if extras is not None else {}
    except AttributeError:
        raise TypeError(
            f"metadata must be a Mapping or have an extras attribute, "
            f"got {type(metadata).__name__}"
        ) from None
