"""Tests for the utils module."""
from pyprobe import utils


def test_flatten():
    """Test flattening lists."""
    lst = [[1, 2, 3], [4, 5], 6]
    flat_list = utils.flatten_list(lst)
    assert flat_list == [1, 2, 3, 4, 5, 6]
