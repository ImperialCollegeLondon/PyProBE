"""Tests for analysis helper functions in pyprobe.analysis.utils."""

import copy

import numpy as np
import polars as pl
import pytest

import pyprobe.analysis.utils as utils
from pyprobe.analysis import differentiation, pulsing, smoothing
from pyprobe.columns import BDF, Column, ColumnResolutionError
from pyprobe.result import Result, Table
from tests.metadata_helpers import build_metadata


@pytest.fixture
def result():
    """A simple Result with Current / A, Voltage / V, and Test Time / s columns."""
    return Result(
        lf=pl.LazyFrame(
            {
                BDF.CURRENT_AMPERE.name: [1.0, 2.0, 3.0],
                BDF.VOLTAGE_VOLT.name: [3.0, 3.5, 4.0],
                BDF.TEST_TIME_SECOND.name: [0.0, 1.0, 2.0],
            }
        ),
        metadata=build_metadata(cell_id="test"),
        column_definitions={"Current": "current def", "Voltage": "voltage def"},
    )


@pytest.fixture
def result_with_recipe_cols():
    """Result with step charge/discharge inputs for net capacity resolution."""
    return Result(
        lf=pl.LazyFrame(
            {
                BDF.STEP_CHARGING_CAPACITY_AH.name: [0.0, 0.5, 1.0],
                BDF.STEP_DISCHARGING_CAPACITY_AH.name: [0.0, 0.5, 1.0],
                BDF.CURRENT_AMPERE.name: [1.0, 1.0, 1.0],
                BDF.TEST_TIME_SECOND.name: [0.0, 1.0, 2.0],
                BDF.STEP_COUNT.name: [0, 0, 0],
            }
        ),
        metadata=build_metadata(),
        column_definitions={},
    )


def test_validate_columns_all_resolvable_success(result):
    """validate_columns returns None when all columns resolve."""
    assert utils.validate_columns(result, BDF.CURRENT_AMPERE, "Voltage / V") is None


def test_validate_columns_unresolvable_raises(result):
    """validate_columns raises ColumnResolutionError on first miss."""
    with pytest.raises(ColumnResolutionError, match="Nonexistent"):
        utils.validate_columns(result, BDF.CURRENT_AMPERE, "Nonexistent / 1")


def test_validate_columns_bdf_recipe_resolution(result_with_recipe_cols):
    """validate_columns succeeds when BDF.NET_CAPACITY_AH resolves via recipe."""
    utils.validate_columns(result_with_recipe_cols, BDF.NET_CAPACITY_AH)


def test_get_columns_single_returns_ndarray(result):
    """get_columns with one column returns a single NDArray."""
    arr = utils.get_columns(result, BDF.VOLTAGE_VOLT)
    assert isinstance(arr, np.ndarray)
    np.testing.assert_array_equal(arr, [3.0, 3.5, 4.0])


def test_get_columns_multiple_returns_tuple_in_order(result):
    """get_columns with multiple columns returns tuple in argument order."""
    t, v = utils.get_columns(result, BDF.TEST_TIME_SECOND, BDF.VOLTAGE_VOLT)
    np.testing.assert_array_equal(t, [0.0, 1.0, 2.0])
    np.testing.assert_array_equal(v, [3.0, 3.5, 4.0])


def test_get_columns_unit_conversion(result):
    """get_columns performs unit conversion via Column reference."""
    arr = utils.get_columns(result, Column("Current", "mA"))
    np.testing.assert_array_almost_equal(arr, [1000.0, 2000.0, 3000.0])


def test_get_columns_unresolvable_raises(result):
    """get_columns raises ColumnResolutionError on miss."""
    with pytest.raises(ColumnResolutionError):
        utils.get_columns(result, "Nonexistent / 1")


def test_resolve_exprs_returns_expr_tuple(result):
    """resolve_exprs returns a tuple of pl.Expr with correct output names."""
    exprs = utils.resolve_exprs(result, BDF.CURRENT_AMPERE, BDF.VOLTAGE_VOLT)
    assert isinstance(exprs, tuple)
    assert len(exprs) == 2
    assert all(isinstance(e, pl.Expr) for e in exprs)
    names = [e.meta.output_name() for e in exprs]
    assert names == [BDF.CURRENT_AMPERE.name, BDF.VOLTAGE_VOLT.name]


def test_resolve_exprs_unresolvable_raises(result):
    """resolve_exprs raises ColumnResolutionError on miss."""
    with pytest.raises(ColumnResolutionError):
        utils.resolve_exprs(result, "Nonexistent / 1")


def test_build_result_lazyframe_input(result):
    """build_result with LazyFrame preserves it and uses provided column_definitions."""
    lf = pl.LazyFrame({"x": [1, 2, 3]})
    new = utils.build_result(result, lf, column_definitions={"x": "x def"})
    assert new.lf is lf
    assert new.column_definitions == {"x": "x def"}
    assert new.metadata.model_dump(exclude={"extras"}) == result.metadata.model_dump(
        exclude={"extras"},
    )


def test_build_result_dataframe_input(result):
    """build_result with DataFrame converts it to LazyFrame."""
    df = pl.DataFrame({"x": [1, 2, 3]})
    new = utils.build_result(result, df)
    assert isinstance(new.lf, pl.LazyFrame)
    assert new.column_definitions == {
        "Current": "current def",
        "Voltage": "voltage def",
    }


def test_build_result_inherits_column_definitions_when_none(result):
    """build_result without column_definitions inherits from source."""
    lf = pl.LazyFrame({"x": [1]})
    new = utils.build_result(result, lf)
    assert new.column_definitions == {
        "Current": "current def",
        "Voltage": "voltage def",
    }


def test_build_result_replaces_column_definitions(result):
    """build_result with column_definitions replaces source definitions entirely."""
    lf = pl.LazyFrame({"x": [1]})
    new = utils.build_result(result, lf, column_definitions={"x": "x def"})
    assert new.column_definitions == {"x": "x def"}
    assert "Current" not in new.column_definitions


def test_build_result_inherit_and_extend(result):
    """build_result can inherit-and-extend by spreading source.column_definitions."""
    lf = pl.LazyFrame({"x": [1]})
    new = utils.build_result(
        result, lf, column_definitions={**result.column_definitions, "New": "new def"}
    )
    assert "Current" in new.column_definitions
    assert new.column_definitions["New"] == "new def"


def test_build_result_deep_copies_metadata(result):
    """build_result's Table construction gives the new object its own record."""
    lf = pl.LazyFrame({"x": [1]})
    new = utils.build_result(result, lf)
    assert new.metadata == result.metadata
    assert new.metadata is not result.metadata


def test_append_columns_from_array(result):
    """append_columns adds an ndarray column to the result."""
    new = utils.append_columns(result, {"Smoothed / V": np.array([3.1, 3.6, 4.1])})
    assert "Smoothed / V" in new.lf.collect_schema().names()
    np.testing.assert_array_almost_equal(new.get("Smoothed / V"), [3.1, 3.6, 4.1])


def test_append_columns_from_expr(result):
    """append_columns adds a polars Expr column to the result."""
    new = utils.append_columns(
        result,
        {"Power / W": pl.col(BDF.CURRENT_AMPERE.name) * pl.col(BDF.VOLTAGE_VOLT.name)},
    )
    assert "Power / W" in new.lf.collect_schema().names()
    expected = np.array([1.0 * 3.0, 2.0 * 3.5, 3.0 * 4.0])
    np.testing.assert_array_almost_equal(new.get("Power / W"), expected)


def test_append_columns_collision_raises_without_overwrite(result):
    """append_columns raises ColumnCollisionError on collision when overwrite=False."""
    with pytest.raises(utils.ColumnCollisionError, match="Voltage / V"):
        utils.append_columns(result, {BDF.VOLTAGE_VOLT.name: pl.lit(0.0)})


def test_append_columns_collision_with_overwrite_succeeds(result):
    """append_columns with overwrite=True replaces existing column without error."""
    new = utils.append_columns(
        result, {BDF.VOLTAGE_VOLT.name: pl.lit(0.0)}, overwrite=True
    )
    np.testing.assert_array_equal(new.get(BDF.VOLTAGE_VOLT.name), [0.0, 0.0, 0.0])


def test_append_columns_array_length_mismatch_propagates(result):
    """append_columns propagates polars ShapeError on array length mismatch."""
    with pytest.raises(Exception):
        utils.append_columns(result, {"Short / 1": np.array([1.0])}).lf.collect()


def test_assemble_array_bdf_reference(result):
    """assemble_array stacks BDF column across multiple results."""
    result2 = copy.deepcopy(result)
    arr = utils.assemble_array([result, result2], BDF.VOLTAGE_VOLT)
    expected = np.vstack([result.get(BDF.VOLTAGE_VOLT), result2.get(BDF.VOLTAGE_VOLT)])
    np.testing.assert_array_equal(arr, expected)


def test_assemble_array_string_reference(result):
    """assemble_array stacks a string-named column across multiple results."""
    result2 = copy.deepcopy(result)
    arr = utils.assemble_array([result, result2], BDF.CURRENT_AMPERE.name)
    expected = np.vstack(
        [result.get(BDF.CURRENT_AMPERE.name), result2.get(BDF.CURRENT_AMPERE.name)]
    )
    np.testing.assert_array_equal(arr, expected)


_FULL_COLS = {
    BDF.CURRENT_AMPERE.name: [1.0, 2.0, 3.0],
    BDF.VOLTAGE_VOLT.name: [3.0, 3.5, 4.0],
    BDF.TEST_TIME_SECOND.name: [0.0, 1.0, 2.0],
    "SOC / %": [0.0, 50.0, 100.0],
    "x / 1": [0.0, 1.0, 2.0],
    "y / 1": [1.0, 2.0, 3.0],
}

_PULSING_COLS = {
    BDF.CURRENT_AMPERE.name: [0.0, 1.0, 0.0],
    BDF.VOLTAGE_VOLT.name: [3.0, 3.5, 4.0],
    BDF.TEST_TIME_SECOND.name: [0.0, 1.0, 2.0],
    BDF.NET_CAPACITY_AH.name: [0.0, 0.5, 1.0],
    "SOC / %": [0.0, 50.0, 100.0],
}


def _strip(cols: dict[str, list[float]], drop: str) -> Result:
    return Result(
        lf=pl.LazyFrame({k: v for k, v in cols.items() if k != drop}),
        metadata=build_metadata(),
    )


@pytest.mark.parametrize(
    "func, base_cols, kwargs, strip_column",
    [
        (
            differentiation.gradient,
            _FULL_COLS,
            {"x": "x / 1", "y": "y / 1"},
            "y / 1",
        ),
        (
            differentiation.differentiate_lean,
            _FULL_COLS,
            {"x": "x / 1", "y": "y / 1"},
            "x / 1",
        ),
        (
            smoothing.spline_smoothing,
            _FULL_COLS,
            {"target_column": "y / 1", "x": "x / 1"},
            "y / 1",
        ),
        (
            smoothing.savgol_smoothing,
            _FULL_COLS,
            {"target_column": "y / 1", "window_length": 3, "polyorder": 1},
            "y / 1",
        ),
        (
            smoothing.downsample,
            _FULL_COLS,
            {"target_column": "x / 1", "sampling_interval": 0.5},
            "x / 1",
        ),
        (
            pulsing.get_ocv_curve,
            _PULSING_COLS,
            {},
            BDF.CURRENT_AMPERE.name,
        ),
        (
            pulsing.get_resistances,
            _PULSING_COLS,
            {},
            BDF.VOLTAGE_VOLT.name,
        ),
    ],
)
def test_all_public_analysis_funcs_raise_on_missing_column(
    func, base_cols, kwargs, strip_column
):
    """Analysis functions raise ColumnResolutionError on a missing required column."""
    stripped = _strip(base_cols, strip_column)
    with pytest.raises(ColumnResolutionError):
        func(stripped, **kwargs)


def test_quantify_degradation_modes_raises_on_missing_column():
    """quantify_degradation_modes raises ColumnResolutionError on missing column."""
    from pyprobe.analysis import degradation_mode_analysis as dma

    full = {
        "x_pe low SOC": [0.1],
        "x_pe high SOC": [0.9],
        "x_ne low SOC": [0.1],
        "x_ne high SOC": [0.9],
        "Cell Capacity [Ah]": [1.0],
        "Cathode Capacity [Ah]": [1.1],
        "Anode Capacity [Ah]": [1.2],
        "Li Inventory [Ah]": [1.0],
    }
    stripped = Result(
        lf=pl.LazyFrame({k: v for k, v in full.items() if k != "Cell Capacity [Ah]"}),
        metadata=build_metadata(),
    )
    with pytest.raises(ColumnResolutionError):
        dma.quantify_degradation_modes([stripped])


class TestValidateQuantity:
    """Tests for the validate_quantity boundary check."""

    def test_validate_quantity_resolvable_passes(self, result):
        """Expected quantities present on a Table raise nothing."""
        utils.validate_quantity(result, BDF.CURRENT_AMPERE, BDF.VOLTAGE_VOLT)

    def test_validate_quantity_on_curve_passes(self):
        """Quantities carried by a Curve resolve via its columns accessor."""
        table = Table(
            lf=pl.LazyFrame(
                {
                    BDF.TEST_TIME_SECOND.name: [0.0, 1.0, 2.0],
                    BDF.VOLTAGE_VOLT.name: [3.0, 3.5, 4.0],
                }
            ),
            metadata=build_metadata(),
        )
        curve = table.to_curve(BDF.VOLTAGE_VOLT, x=BDF.TEST_TIME_SECOND)
        utils.validate_quantity(curve, BDF.TEST_TIME_SECOND, BDF.VOLTAGE_VOLT)

    def test_validate_quantity_missing_raises_clear_message(self, result):
        """An absent quantity raises ColumnResolutionError naming the quantity."""
        with pytest.raises(ColumnResolutionError, match="Net Energy"):
            utils.validate_quantity(result, BDF.NET_ENERGY_WH)
