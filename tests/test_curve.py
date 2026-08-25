"""Tests for the Curve continuous data object."""

import numpy as np
import pytest
from scipy.interpolate import PchipInterpolator, PPoly, make_smoothing_spline

from pyprobe.columns import BDF, column_factory_from_string
from pyprobe.result import Curve, Quantified, Table
from tests.metadata_helpers import read_extras


@pytest.fixture
def xy():
    """A monotonic (x, y) sample with y = x**2."""
    x = np.linspace(0.0, 1.0, 21)
    return x, x**2


@pytest.fixture
def curve(xy):
    """A Curve built from a PchipInterpolator over the sample."""
    x, y = xy
    return Curve.from_poly(
        PchipInterpolator(x, y),
        x_quantity=BDF.TEST_TIME_SECOND,
        y_quantity=BDF.VOLTAGE_VOLT,
    )


def test_curve_is_ppoly(curve):
    """A Curve satisfies isinstance(curve, PPoly) for scipy/matplotlib interop."""
    assert isinstance(curve, PPoly)


def test_curve_is_quantified(curve):
    """A Curve satisfies the Quantified contract."""
    assert isinstance(curve, Quantified)
    assert curve.metadata is not None
    assert curve.column_definitions is not None


def test_curve_call_evaluates(curve, xy):
    """Calling a Curve evaluates the underlying function at the data points."""
    x, y = xy
    np.testing.assert_allclose(curve(x), y, atol=1e-6)


def test_curve_columns_expose_axis_roles(curve):
    """Curve.columns exposes .x and .y resolving to the axis quantities."""
    assert curve.columns.x.name == BDF.TEST_TIME_SECOND.name
    assert curve.columns.y.name == BDF.VOLTAGE_VOLT.name
    assert curve.columns.can_resolve(BDF.VOLTAGE_VOLT)


def test_from_poly_accepts_ppoly(xy):
    """from_poly accepts a PPoly subclass and records its method."""
    x, y = xy
    curve = Curve.from_poly(
        PchipInterpolator(x, y),
        x_quantity="Test Time / s",
        y_quantity="Voltage / V",
    )
    assert isinstance(curve, PPoly)
    assert read_extras(curve)["curve_method"] == "PchipInterpolator"


def test_from_poly_accepts_bspline(xy):
    """from_poly normalises a BSpline to PPoly and records the method."""
    x, y = xy
    bspline = make_smoothing_spline(x, y)
    curve = Curve.from_poly(
        bspline,
        x_quantity="Test Time / s",
        y_quantity="Voltage / V",
    )
    assert isinstance(curve, PPoly)
    assert read_extras(curve)["curve_method"] == "smoothing_spline"
    # value round-trips within tolerance of the original BSpline
    np.testing.assert_allclose(curve(x), bspline(x), atol=1e-8)


def test_from_poly_rejects_other_types():
    """from_poly raises TypeError for non-PPoly, non-BSpline inputs."""
    with pytest.raises(TypeError):
        Curve.from_poly(object(), x_quantity="a", y_quantity="b")


def test_derivative_returns_curve_with_derived_quantity(curve):
    """derivative() returns a Curve carrying the d(y)/d(x) quantity."""
    derivative = curve.derivative()
    assert isinstance(derivative, Curve)
    assert derivative.y_quantity.quantity == "d(Voltage)_d(Test Time)"
    assert derivative.y_quantity.unit == "V s^-1"
    assert derivative.y_quantity.name == "d(Voltage)_d(Test Time) / V s^-1"
    # x quantity and metadata are preserved
    assert derivative.x_quantity.name == curve.x_quantity.name
    assert derivative.metadata == curve.metadata


def test_derivative_quantity_name_round_trips_through_column_parser(curve):
    """Derived names avoid embedded slashes so the column parser can read them."""
    derivative = curve.derivative()
    parsed = column_factory_from_string(derivative.y_quantity.name)
    assert parsed.quantity == derivative.y_quantity.quantity
    assert parsed.unit == derivative.y_quantity.unit


def test_derivative_values_match_analytic(curve, xy):
    """The derivative of y = x**2 is approximately 2x on the interior."""
    x, _ = xy
    derivative = curve.derivative()
    interior = x[2:-2]
    np.testing.assert_allclose(derivative(interior), 2 * interior, atol=0.1)


def test_to_table_round_trip(curve, xy):
    """to_table samples the curve onto a grid and returns a Table."""
    x, y = xy
    table = curve.to_table(x)
    assert isinstance(table, Table)
    assert table.columns.names == (
        BDF.TEST_TIME_SECOND.name,
        BDF.VOLTAGE_VOLT.name,
    )
    sampled = table.get(BDF.VOLTAGE_VOLT.name)
    np.testing.assert_allclose(sampled, y, atol=1e-6)


def test_curve_plots_with_matplotlib(curve, xy):
    """A Curve drops straight into matplotlib as a callable PPoly."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x, _ = xy
    fig, ax = plt.subplots()
    ax.plot(x, curve(x))
    plt.close(fig)
