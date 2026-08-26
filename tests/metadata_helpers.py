"""Helpers for building and reading test metadata."""

from typing import Any

import bdf


def build_metadata(**keys: Any) -> bdf.Metadata:
    """Build a metadata record from keyword arguments.

    This helper abstracts metadata construction for tests, allowing future
    changes to metadata representation without updating every test.

    Args:
        **keys: Key-value pairs to include in the record's extras mapping.

    Returns:
        bdf.Metadata: A metadata record.
    """
    return bdf.Metadata(extras=dict(keys) if keys else None)


def read_extras(obj: Any) -> dict[str, Any]:
    """Read the extras from a metadata-bearing object.

    This helper abstracts metadata access for tests, allowing future
    changes to metadata representation without updating every test.

    Args:
        obj: An object with metadata (e.g., Table, Curve).

    Returns:
        dict: The extras mapping.

    Raises:
        TypeError: Where the object's metadata has no extras attribute.
    """
    metadata = obj.metadata

    try:
        extras = metadata.extras
        return extras if extras is not None else {}
    except AttributeError:
        raise TypeError(
            f"metadata must be a bdf.Metadata record, got {type(metadata).__name__}"
        ) from None
