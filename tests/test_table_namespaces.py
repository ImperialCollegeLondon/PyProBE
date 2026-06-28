"""Tests for the flat operation methods on Table and CyclingData."""

import warnings

import numpy as np
import polars as pl
import polars.testing as pl_testing
import pytest
from scipy.interpolate import (
    Akima1DInterpolator,
    CubicSpline,
    PchipInterpolator,
    make_smoothing_spline,
)

from pyprobe.analysis import differentiation, smoothing
from pyprobe.columns import BDF
from pyprobe.rawdata import CyclingData, RawData
from pyprobe.result import Curve, Table


@pytest.fixture
def table():
    """A Table with a monotonic time axis and a smooth voltage signal."""
    time = np.linspace(0.0, 10.0, 51)
    voltage = 3.0 + 0.1 * time
    return Table(
        lf=pl.DataFrame(
            {
                BDF.TEST_TIME_SECOND.name: time,
                BDF.VOLTAGE_VOLT.name: voltage,
            }
        ).lazy(),
        metadata={"cell_id": "test"},
    )


@pytest.fixture
def cycling_data():
    """A CyclingData object with net capacity and net energy columns."""
    capacity = np.array([0.0, 1.0, 2.0, 1.5, 3.0])
    return CyclingData(
        lf=pl.DataFrame(
            {
                BDF.UNIX_TIME_SECOND.name: [0.0, 1.0, 2.0, 3.0, 4.0],
                BDF.CURRENT_AMPERE.name: [1.0] * 5,
                BDF.VOLTAGE_VOLT.name: [3.0] * 5,
                BDF.NET_CAPACITY_AH.name: capacity,
                BDF.NET_ENERGY_WH.name: capacity * 3.7,
            }
        ).lazy(),
        metadata={},
    )


class TestToCurve:
    """to_curve returns a labelled Curve for any scipy fit callable."""

    @pytest.mark.parametrize(
        "fit",
        [PchipInterpolator, CubicSpline, Akima1DInterpolator, make_smoothing_spline],
    )
    def test_to_curve_returns_labelled_curve(self, table, fit):
        """Each scipy fit callable returns a Curve labelled with x and y."""
        curve = table.to_curve(BDF.VOLTAGE_VOLT, x=BDF.TEST_TIME_SECOND, fit=fit)
        assert isinstance(curve, Curve)
        assert curve.columns.x.name == BDF.TEST_TIME_SECOND.name
        assert curve.columns.y.name == BDF.VOLTAGE_VOLT.name

    def test_to_curve_default_fit_is_pchip(self, table):
        """The default fit is PchipInterpolator, recorded in the metadata."""
        curve = table.to_curve(BDF.VOLTAGE_VOLT, x=BDF.TEST_TIME_SECOND)
        assert isinstance(curve, Curve)
        assert curve.metadata["curve_method"] == "PchipInterpolator"

    def test_to_curve_interpolator_passes_through_points(self, table):
        """An interpolating Curve passes through the supplied data points."""
        x, y = table.get(BDF.TEST_TIME_SECOND.name, BDF.VOLTAGE_VOLT.name)
        curve = table.to_curve(
            BDF.VOLTAGE_VOLT, x=BDF.TEST_TIME_SECOND, fit=CubicSpline
        )
        np.testing.assert_allclose(curve(x), y, atol=1e-6)

    def test_to_curve_forwards_kwargs(self, table):
        """Extra kwargs are forwarded to the fit callable."""
        curve = table.to_curve(
            BDF.VOLTAGE_VOLT,
            x=BDF.TEST_TIME_SECOND,
            fit=CubicSpline,
            bc_type="natural",
        )
        assert isinstance(curve, Curve)

    def test_to_curve_non_conforming_fit_raises_typeerror(self, table):
        """A fit returning neither PPoly nor BSpline raises TypeError."""
        with pytest.raises(TypeError):
            table.to_curve(
                BDF.VOLTAGE_VOLT,
                x=BDF.TEST_TIME_SECOND,
                fit=lambda x, y: "not a poly",
            )


class TestSavgol:
    """savgol returns a Table equal to the standalone savgol_smoothing."""

    def test_savgol_returns_table_matching_standalone(self, table):
        """Savgol returns a Table equal to the standalone function."""
        result = table.savgol(BDF.VOLTAGE_VOLT.name, window_length=5, polyorder=2)
        assert isinstance(result, Table)
        expected = smoothing.savgol_smoothing(
            table, BDF.VOLTAGE_VOLT.name, window_length=5, polyorder=2
        )
        pl_testing.assert_frame_equal(result.data, expected.data)


class TestDownsample:
    """downsample returns a Table matching the standalone function."""

    def test_downsample_returns_table_matching_standalone(self, table):
        """Downsample returns a Table equal to the standalone function."""
        result = table.downsample(BDF.TEST_TIME_SECOND.name, sampling_interval=1.0)
        assert isinstance(result, Table)
        expected = smoothing.downsample(
            table, BDF.TEST_TIME_SECOND.name, sampling_interval=1.0
        )
        pl_testing.assert_frame_equal(result.data, expected.data)


class TestGradient:
    """gradient returns a Table matching the standalone function."""

    def test_gradient_returns_table_matching_standalone(self, table):
        """Gradient returns a Table with x, y and gradient columns."""
        result = table.gradient(y=BDF.VOLTAGE_VOLT.name, x=BDF.TEST_TIME_SECOND.name)
        assert isinstance(result, Table)
        expected = differentiation.gradient(
            table, x=BDF.TEST_TIME_SECOND.name, y=BDF.VOLTAGE_VOLT.name
        )
        pl_testing.assert_frame_equal(result.data, expected.data)


class TestCyclingDataQuantities:
    """The scalar quantity methods on CyclingData return documented values."""

    def test_net_capacity_is_extent(self, cycling_data):
        """net_capacity returns max minus min of the net capacity."""
        capacity = cycling_data.get(BDF.NET_CAPACITY_AH.name)
        value = cycling_data.net_capacity()
        assert isinstance(value, float)
        assert value == pytest.approx(capacity.max() - capacity.min())

    def test_net_energy_is_extent(self, cycling_data):
        """net_energy returns max minus min of the net energy."""
        energy = cycling_data.get(BDF.NET_ENERGY_WH.name)
        assert cycling_data.net_energy() == pytest.approx(energy.max() - energy.min())

    def test_capacity_throughput_is_cumulative_abs(self, cycling_data):
        """capacity_throughput returns the cumulative absolute change."""
        capacity = cycling_data.get(BDF.NET_CAPACITY_AH.name)
        assert cycling_data.capacity_throughput() == pytest.approx(
            np.abs(np.diff(capacity)).sum()
        )

    def test_energy_throughput_is_cumulative_abs(self, cycling_data):
        """energy_throughput returns the cumulative absolute change."""
        energy = cycling_data.get(BDF.NET_ENERGY_WH.name)
        assert cycling_data.energy_throughput() == pytest.approx(
            np.abs(np.diff(energy)).sum()
        )


class TestRawDataAlias:
    """RawData is a deprecated alias of CyclingData."""

    def test_isinstance_holds_for_any_cycling_data(self, cycling_data):
        """isinstance(obj, RawData) is True for any CyclingData instance."""
        assert isinstance(cycling_data, RawData)

    def test_construction_warns(self, cycling_data):
        """Constructing RawData directly emits a DeprecationWarning."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            RawData(lf=cycling_data.lf, metadata={})
        assert any(issubclass(w.category, DeprecationWarning) for w in caught)

    def test_capacity_matches_net_capacity_and_warns(self, cycling_data):
        """The deprecated capacity property returns net_capacity and warns."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            value = cycling_data.capacity
        assert value == cycling_data.net_capacity()
        assert any(issubclass(w.category, DeprecationWarning) for w in caught)
