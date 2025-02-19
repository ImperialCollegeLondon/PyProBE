"""Module for testing the differentiation functions."""

import numpy as np

import pyprobe.analysis.base.differentiation_functions as diff_functions


def test_differentiate_LEAN():
    """Test the LEAN differentiation method."""
    x_min = 0
    x_max = 1
    num_points = 10000
    x = np.linspace(x_min, x_max, num_points)

    average_gradient = 2
    n = 200

    total_range = x_max * average_gradient
    levels = np.linspace(0, total_range, n)

    points_per_level = num_points // n
    remainder = num_points % n

    y = np.repeat(levels, points_per_level)
    if remainder > 0:
        y = np.append(y, levels[:remainder])

    x_pts, _, dxdy = diff_functions.calc_gradient_with_lean(x, y, 1, "dxdy")
    assert np.isclose(np.median(dxdy), 1 / average_gradient, rtol=0.1)
    x_pts, _, dydx = diff_functions.calc_gradient_with_lean(x, y, 1, "dydx")
    assert np.isclose(np.median(dydx), average_gradient, rtol=0.1)

    # flip the data
    y = np.flip(y)
    x = np.flip(x)
    x_pts, _, dxdy = diff_functions.calc_gradient_with_lean(x, y, 1, "dxdy")
    assert np.isclose(np.median(dxdy), 1 / average_gradient, rtol=0.1)


def test_x_sections():
    """Test the LEAN get_x_sections method."""
    x = np.array([0, 1, 2, 3, 4])
    x_sections = diff_functions.get_x_sections(x)
    assert x_sections == [slice(0, 5)]

    x = np.array(
        [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20, 20.5, 21, 21.5, 22, 22.5, 23]
    )
    x_sections = diff_functions.get_x_sections(x)
    assert x_sections == [
        slice(0, 7),
        slice(6, 14),
        slice(13, 20),
    ]
