"""Tests for the utils module."""

import inspect
from typing import Any

import pytest
from pydantic import BaseModel, Field

from pyprobe import utils
from pyprobe.utils import PyProBEValidationError, catch_pydantic_validation


def test_flatten():
    """Test flattening lists."""
    lst = [[1, 2, 3], [4, 5], 6]
    flat_list = utils.flatten_list(lst)
    assert flat_list == [1, 2, 3, 4, 5, 6]


class TestModel(BaseModel):
    """Test model for validation."""

    value: int = Field(gt=0)
    name: str = Field(min_length=3)


@catch_pydantic_validation
def sample_func(value: Any, name: Any) -> None:
    """Test docstring."""
    TestModel(value=value, name=name)


def test_catch_pydantic_validation_single_error():
    """Test single validation error handling."""
    with pytest.raises(PyProBEValidationError) as exc_info:
        sample_func(-1, "test")

    assert "value" in str(exc_info.value)
    assert "greater than 0" in str(exc_info.value).lower()


def test_catch_pydantic_validation_multiple_errors():
    """Test multiple validation errors handling."""
    with pytest.raises(PyProBEValidationError) as exc_info:
        sample_func(-1, "a")

    error_msg = str(exc_info.value)
    assert "value" in error_msg
    assert "name" in error_msg
    assert "greater than 0" in error_msg.lower()
    assert "at least 3 characters" in error_msg.lower()


def test_catch_pydantic_validation_preserves_signature():
    """Test decorator preserves function signature."""

    @catch_pydantic_validation
    def sample_func(x: int, y: str = "test") -> str:
        """Test docstring."""
        return y * x

    sig = inspect.signature(sample_func)
    assert "x" in sig.parameters
    assert "y" in sig.parameters
    assert sig.return_annotation is str
    assert sample_func.__name__ == "sample_func"
    assert sample_func.__doc__ == "Test docstring."
