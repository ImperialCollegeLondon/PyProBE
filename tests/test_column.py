"""Tests for the column module.

This module provides tests for BDF column abstractions, including parsing,
unit conversion, and Polars expression generation with recipe-based fallbacks
via ColumnDict.
"""

from __future__ import annotations

from typing import cast

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from pyprobe.columns import (
    BDF,
    BDF_IRI_PREFIX,
    BDF_PATTERN,
    BDF_RECIPES,
    DEFAULT_COLUMNS,
    BDFColumn,
    Column,
    ColumnDict,
    ColumnResolutionError,
    Recipe,
    _apply_conversion,
    _global_cumulative_from_step_ch_dch,
    _global_net_from_step_ch_dch,
    _resolve_unit,
    _seam_charge,
    _split_quantity_unit,
    column_factory,
    column_factory_from_string,
    is_valid_column_name,
)


def _compute_series(df: pl.DataFrame, compute: pl.Expr) -> list[int | float]:
    """Evaluate a Polars expression against a DataFrame and return values.

    Args:
        df: Input DataFrame containing the expression dependencies.
        compute: Polars expression to evaluate.

    Returns:
        Materialized values from the selected expression.
    """
    return df.select(compute.alias("result"))["result"].to_list()


def _flatten(
    recipes: dict[BDF, list[Recipe]],
) -> list[tuple[BDF, Recipe]]:
    """Flatten recipe lists into `(bdf_key, recipe)` pairs.

    Args:
        recipes: Mapping from target BDF column to candidate recipes.

    Returns:
        Flat list of `(bdf_key, recipe)` tuples.
    """
    return [
        (bdf_key, recipe)
        for bdf_key, recipe_list in recipes.items()
        for recipe in recipe_list
    ]


def _recipe_mapping(*columns: BDF) -> dict[BDF, pl.Expr]:
    """Build a `{BDF: pl.Expr}` mapping from BDF names.

    Args:
        *columns: BDF columns to expose as `pl.col(...)` expressions.

    Returns:
        Mapping from each BDF column to its matching `pl.col(...)` expression.
    """
    return {column: pl.col(column.name) for column in columns}


class TestColumnInit:
    """Tests for Column.__init__ and basic construction."""

    @pytest.mark.parametrize(
        "quantity,unit,expected_name",
        [
            ("Current", "A", "Current / A"),
            ("Voltage", "V", "Voltage / V"),
            ("Step Count", "1", "Step Count / 1"),
            ("Net Capacity", "Ah", "Net Capacity / Ah"),
        ],
    )
    def test_init_creates_column_name(
        self, quantity: str, unit: str, expected_name: str
    ) -> None:
        """Column.__init__ correctly constructs column_name."""
        col = Column(quantity, unit)
        assert col.quantity == quantity
        assert col.unit == unit
        assert col.name == expected_name

    def test_init_default_unit_is_dimensionless(self) -> None:
        """Column with no unit arg defaults to '1'."""
        col = Column("Step")
        assert col.unit == "1"
        assert col.name == "Step / 1"


class TestColumnFactory:
    """Tests for the column_factory function."""

    bdf_cases = [(column.quantity, column.unit, column) for column in BDF]

    @pytest.mark.parametrize("quantity,unit,expected_col", bdf_cases)
    def test_factory_returns_expected_column(
        self, quantity: str, unit: str, expected_col: BDFColumn
    ) -> None:
        """column_factory returns the expected BDFColumn for given quantity/unit."""
        col = column_factory(quantity, unit)
        assert col == expected_col

    @pytest.mark.parametrize("quantity,unit,expected_col", bdf_cases)
    def test_factory_from_string_returns_expected_column(
        self, quantity: str, unit: str, expected_col: BDFColumn
    ) -> None:
        """column_factory_from_string returns the expected BDFColumn."""
        col = column_factory_from_string(expected_col.name)
        assert col == expected_col

    non_bdf_cases = [
        ("Custom Quantity", "Custom Unit"),
        ("Temperature", "degC"),
        ("Current", "mA"),
    ]

    @pytest.mark.parametrize("quantity,unit", non_bdf_cases)
    def test_factory_non_bdf_columns(self, quantity: str, unit: str) -> None:
        """column_factory can create Column instances for non-BDF quantities."""
        col = column_factory(quantity, unit)
        assert isinstance(col, Column)
        assert col.quantity == quantity
        assert col.unit == unit


class TestConversionParameters:
    """Tests for Column.conversion_parameters and unit math."""

    @pytest.mark.parametrize(
        "source_unit,target_unit,expected_factor,expected_offset",
        [
            ("A", "mA", 1000.0, 0.0),
            ("mA", "A", 0.001, 0.0),
            ("Ah", "mAh", 1000.0, 0.0),
            ("V", "mV", 1000.0, 0.0),
            ("Wh", "mWh", 1000.0, 0.0),
            ("A", "A", 1.0, 0.0),
            ("W", "kW", 1 / 1000.0, 0.0),
            ("mV", "V", 0.001, 0.0),
        ],
    )
    def test_conversion_parameters_multiplicative(
        self,
        source_unit: str,
        target_unit: str,
        expected_factor: float,
        expected_offset: float,
    ) -> None:
        """Test multiplicative conversions for different unit pairs."""
        col = column_factory_from_string(f"Quantity / {source_unit}")
        factor, offset = col.conversion_parameters(target_unit)
        assert factor == pytest.approx(expected_factor, rel=1e-9)
        assert offset == pytest.approx(expected_offset, abs=1e-9)

    def test_conversion_celsius_to_kelvin(self) -> None:
        """Affine conversion degC to K: factor=1, offset=273.15."""
        col = column_factory_from_string("Temperature / C")
        factor, offset = col.conversion_parameters("K")
        assert factor == pytest.approx(1.0, rel=1e-9)
        assert offset == pytest.approx(273.15, abs=0.01)

    def test_conversion_incompatible_units_raises(self) -> None:
        """Converting between incompatible units raises ValueError."""
        col = column_factory_from_string("Current / A")
        with pytest.raises(ValueError, match="Cannot convert"):
            col.conversion_parameters("V")

    def test_conversion_dimensionless_raises(self) -> None:
        """Converting a dimensionless column raises ValueError."""
        col = Column("Step")
        with pytest.raises(ValueError, match="dimensionless"):
            col.conversion_parameters("1")


class TestBDFColumnIRI:
    """Tests for BDFColumn.iri computed property."""

    @pytest.mark.parametrize(
        "col_obj,expected_iri_suffix",
        [
            (BDF.CURRENT_AMPERE, "current_ampere"),
            (BDF.VOLTAGE_VOLT, "voltage_volt"),
            (BDF.STEP_COUNT, "step_count"),
            (BDF.STEP_ID, "step_id"),
            (BDF.CYCLE_COUNT, "cycle_count"),
            (BDF.CHARGING_CAPACITY_AH, "charging_capacity_ah"),
            (BDF.TEMPERATURE_T1_CELSIUS, "temperature_t1_celsius"),
        ],
    )
    def test_iri_computed_from_quantity_and_unit(
        self, col_obj: BDFColumn, expected_iri_suffix: str
    ) -> None:
        """IRI is computed from quantity and pint long-form unit."""
        assert col_obj.iri == f"{BDF_IRI_PREFIX}{expected_iri_suffix}"

    @pytest.mark.parametrize("col_obj", list(BDF))
    def test_all_bdf_column_iris_are_valid_urls(self, col_obj: BDFColumn) -> None:
        """All BDF column IRIs are complete and properly formatted."""
        iri = col_obj.iri
        assert iri.startswith(BDF_IRI_PREFIX)
        assert len(iri) > len(BDF_IRI_PREFIX)
        assert iri.endswith(iri.split("#")[-1])


@pytest.fixture
def recipe_sample_df() -> pl.DataFrame:
    """Provide one explicit fixture covering every recipe-touched column.

    Twenty rows simulating charge / discharge / charge / discharge / rest steps
    split across two cycles. Values are mutually consistent across every
    registered recipe (including fallback branches) for every BDF column
    reachable from :data:`BDF_RECIPES` -- capacity and energy, step and
    cycle scope, and the global charging/discharging/net/cumulative columns.
    The schedule accumulator columns never reset across these twenty rows, so
    they repeat the global charging/discharging columns exactly.

    Returns:
        DataFrame containing 20 hand-verified rows covering all recipe inputs
        and targets across time, capacity, and energy columns.
    """
    # fmt: off
    return pl.DataFrame({
        # Hourly cadence (dt = 3600 s) so that a plain trapezoidal integral of
        # `Current / A` (or `Power / W`) against this column reproduces the
        # recorded Ah/Wh increments below exactly (1 A for 1 h == 1 Ah).
        # Duplicated at every step boundary (indices 4, 8, 12, 16) so that
        # dt == 0 there and the seam term is exactly zero
        # A genuine (dt > 0) seam is covered separately by `seam_boundary_df`.
        BDF.UNIX_TIME_SECOND.name:              [1000000.0, 1003600.0, 1007200.0, 1010800.0, 1010800.0, 1014400.0, 1018000.0, 1021600.0, 1021600.0, 1025200.0, 1028800.0, 1032400.0, 1032400.0, 1036000.0, 1039600.0, 1043200.0, 1043200.0, 1046800.0, 1050400.0, 1054000.0],  # noqa: E501
        BDF.TEST_TIME_SECOND.name:              [0.0, 3600.0, 7200.0, 10800.0, 10800.0, 14400.0, 18000.0, 21600.0, 21600.0, 25200.0, 28800.0, 32400.0, 32400.0, 36000.0, 39600.0, 43200.0, 43200.0, 46800.0, 50400.0, 54000.0],  # noqa: E501
        BDF.STEP_ID.name:                       [1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],  # noqa: E501
        BDF.STEP_COUNT.name:                    [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4],  # noqa: E501
        BDF.CYCLE_COUNT.name:                   [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # noqa: E501
        BDF.CURRENT_AMPERE.name:                [1.0, 1.0, 1.0, 1.0, -2.0, -2.0, -2.0, -2.0, 1.0, 1.0, 1.0, 1.0, -2.0, -2.0, -2.0, -2.0, 0.0, 0.0, 0.0, 0.0],  # noqa: E501
        BDF.POWER_WATT.name:                    [2.0, 2.0, 2.0, 2.0, -3.0, -3.0, -3.0, -3.0, 2.0, 2.0, 2.0, 2.0, -3.0, -3.0, -3.0, -3.0, 0.0, 0.0, 0.0, 0.0],  # noqa: E501
        BDF.STEP_CHARGING_CAPACITY_AH.name:     [0.0, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # noqa: E501
        BDF.STEP_DISCHARGING_CAPACITY_AH.name:  [0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 4.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 4.0, 6.0, 0.0, 0.0, 0.0, 0.0],  # noqa: E501
        BDF.STEP_NET_CAPACITY_AH.name:          [0.0, 1.0, 2.0, 3.0, 0.0, -2.0, -4.0, -6.0, 0.0, 1.0, 2.0, 3.0, 0.0, -2.0, -4.0, -6.0, 0.0, 0.0, 0.0, 0.0],  # noqa: E501
        BDF.STEP_CUMULATIVE_CAPACITY_AH.name:   [0.0, 1.0, 2.0, 3.0, 0.0, 2.0, 4.0, 6.0, 0.0, 1.0, 2.0, 3.0, 0.0, 2.0, 4.0, 6.0, 0.0, 0.0, 0.0, 0.0],  # noqa: E501
        BDF.STEP_CHARGING_ENERGY_WH.name:       [0.0, 2.0, 4.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 4.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # noqa: E501
        BDF.STEP_DISCHARGING_ENERGY_WH.name:    [0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 6.0, 9.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 6.0, 9.0, 0.0, 0.0, 0.0, 0.0],  # noqa: E501
        BDF.STEP_NET_ENERGY_WH.name:            [0.0, 2.0, 4.0, 6.0, 0.0, -3.0, -6.0, -9.0, 0.0, 2.0, 4.0, 6.0, 0.0, -3.0, -6.0, -9.0, 0.0, 0.0, 0.0, 0.0],  # noqa: E501
        BDF.STEP_CUMULATIVE_ENERGY_WH.name:     [0.0, 2.0, 4.0, 6.0, 0.0, 3.0, 6.0, 9.0, 0.0, 2.0, 4.0, 6.0, 0.0, 3.0, 6.0, 9.0, 0.0, 0.0, 0.0, 0.0],  # noqa: E501
        BDF.CYCLE_CHARGING_CAPACITY_AH.name:    [0.0, 1.0, 2.0, 3.0, 3.0, 3.0, 3.0, 3.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0],  # noqa: E501
        BDF.CYCLE_DISCHARGING_CAPACITY_AH.name: [0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 4.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 4.0, 6.0, 6.0, 6.0, 6.0, 6.0],  # noqa: E501
        BDF.CYCLE_NET_CAPACITY_AH.name:         [0.0, 1.0, 2.0, 3.0, 3.0, 1.0, -1.0, -3.0, 0.0, 1.0, 2.0, 3.0, 3.0, 1.0, -1.0, -3.0, -3.0, -3.0, -3.0, -3.0],  # noqa: E501
        BDF.CYCLE_CUMULATIVE_CAPACITY_AH.name:  [0.0, 1.0, 2.0, 3.0, 3.0, 5.0, 7.0, 9.0, 0.0, 1.0, 2.0, 3.0, 3.0, 5.0, 7.0, 9.0, 9.0, 9.0, 9.0, 9.0],  # noqa: E501
        BDF.CYCLE_CHARGING_ENERGY_WH.name:      [0.0, 2.0, 4.0, 6.0, 6.0, 6.0, 6.0, 6.0, 0.0, 2.0, 4.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0],  # noqa: E501
        BDF.CYCLE_DISCHARGING_ENERGY_WH.name:   [0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 6.0, 9.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 6.0, 9.0, 9.0, 9.0, 9.0, 9.0],  # noqa: E501
        BDF.CYCLE_NET_ENERGY_WH.name:           [0.0, 2.0, 4.0, 6.0, 6.0, 3.0, 0.0, -3.0, 0.0, 2.0, 4.0, 6.0, 6.0, 3.0, 0.0, -3.0, -3.0, -3.0, -3.0, -3.0],  # noqa: E501
        BDF.CYCLE_CUMULATIVE_ENERGY_WH.name:    [0.0, 2.0, 4.0, 6.0, 6.0, 9.0, 12.0, 15.0, 0.0, 2.0, 4.0, 6.0, 6.0, 9.0, 12.0, 15.0, 15.0, 15.0, 15.0, 15.0],  # noqa: E501
        BDF.CHARGING_CAPACITY_AH.name:          [0.0, 1.0, 2.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 4.0, 5.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0],  # noqa: E501
        BDF.DISCHARGING_CAPACITY_AH.name:       [0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 4.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 8.0, 10.0, 12.0, 12.0, 12.0, 12.0, 12.0],  # noqa: E501
        BDF.NET_CAPACITY_AH.name:               [0.0, 1.0, 2.0, 3.0, 3.0, 1.0, -1.0, -3.0, -3.0, -2.0, -1.0, 0.0, 0.0, -2.0, -4.0, -6.0, -6.0, -6.0, -6.0, -6.0],  # noqa: E501
        BDF.CUMULATIVE_CAPACITY_AH.name:        [0.0, 1.0, 2.0, 3.0, 3.0, 5.0, 7.0, 9.0, 9.0, 10.0, 11.0, 12.0, 12.0, 14.0, 16.0, 18.0, 18.0, 18.0, 18.0, 18.0],  # noqa: E501
        BDF.CHARGING_ENERGY_WH.name:            [0.0, 2.0, 4.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 8.0, 10.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0],  # noqa: E501
        BDF.DISCHARGING_ENERGY_WH.name:         [0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 6.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 12.0, 15.0, 18.0, 18.0, 18.0, 18.0, 18.0],  # noqa: E501
        BDF.NET_ENERGY_WH.name:                 [0.0, 2.0, 4.0, 6.0, 6.0, 3.0, 0.0, -3.0, -3.0, -1.0, 1.0, 3.0, 3.0, 0.0, -3.0, -6.0, -6.0, -6.0, -6.0, -6.0],  # noqa: E501
        BDF.CUMULATIVE_ENERGY_WH.name:          [0.0, 2.0, 4.0, 6.0, 6.0, 9.0, 12.0, 15.0, 15.0, 17.0, 19.0, 21.0, 21.0, 24.0, 27.0, 30.0, 30.0, 30.0, 30.0, 30.0],  # noqa: E501
        # Schedule accumulators never reset across these twenty rows, so they
        # repeat the global charging/discharging columns above exactly.
        BDF.SCHEDULE_CHARGING_CAPACITY_AH.name:    [0.0, 1.0, 2.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 4.0, 5.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0],  # noqa: E501
        BDF.SCHEDULE_DISCHARGING_CAPACITY_AH.name: [0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 4.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 8.0, 10.0, 12.0, 12.0, 12.0, 12.0, 12.0],  # noqa: E501
        BDF.SCHEDULE_CHARGING_ENERGY_WH.name:      [0.0, 2.0, 4.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 8.0, 10.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0],  # noqa: E501
        BDF.SCHEDULE_DISCHARGING_ENERGY_WH.name:   [0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 6.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 12.0, 15.0, 18.0, 18.0, 18.0, 18.0, 18.0],  # noqa: E501
    })
    # fmt: on


@pytest.mark.parametrize("bdf_key,recipe", _flatten(BDF_RECIPES))
def test_recipe_matches_explicit_fixture_output(
    bdf_key: BDF, recipe: Recipe, recipe_sample_df: pl.DataFrame
) -> None:
    """Every registered recipe reproduces the target column from one fixture."""
    required_columns = [column.name for column in recipe.required]
    df = recipe_sample_df.select([*required_columns, bdf_key.name])
    result = _compute_series(df, recipe.compute(_recipe_mapping(*recipe.required)))
    assert result == pytest.approx(df[bdf_key.name].to_list())


@pytest.fixture
def seam_boundary_df() -> pl.DataFrame:
    """One genuine (``dt > 0``, nonzero current) step-boundary seam.

    Two steps -- charge at 2 A then discharge at 1 A -- sampled hourly so the
    seam's ``/ 3600`` conversion cancels exactly, matching the worked example
    in ``capacity_columns.md``: per-step signed totals give ``2 - 1 = 1``, but
    the true signed integral of current across the whole test is ``1.5`` --
    the trapezoidal seam charge at the step boundary (``0.5 * (2 + -1) = 0.5``)
    that a plain diff/clip reconstruction drops.

    Returns:
        DataFrame with one step boundary carrying a real (non-zero-``dt``)
        seam.
    """
    return pl.DataFrame(
        {
            BDF.STEP_CHARGING_CAPACITY_AH.name: [0.0, 2.0, 0.0, 0.0],
            BDF.STEP_DISCHARGING_CAPACITY_AH.name: [0.0, 0.0, 0.0, 1.0],
            BDF.CURRENT_AMPERE.name: [2.0, 2.0, -1.0, -1.0],
            BDF.TEST_TIME_SECOND.name: [0.0, 3600.0, 7200.0, 10800.0],
            BDF.STEP_COUNT.name: [0, 0, 1, 1],
        }
    )


class TestSeamCorrection:
    """Tests for the step-boundary seam correction (fix-capacity-seam-recipes)."""

    def test_seam_charge_zero_at_duplicate_timestamp_boundary(self) -> None:
        """Seam term is zero when a boundary has ``dt == 0``."""
        df = pl.DataFrame(
            {
                "current": [1.0, 1.0, -2.0, -2.0],
                "time": [0.0, 1.0, 1.0, 2.0],
                "key": [0, 0, 1, 1],
            }
        )
        result = df.select(
            _seam_charge(pl.col("current"), pl.col("time"), pl.col("key")).alias("seam")
        )["seam"].to_list()
        assert result == [0.0, 0.0, 0.0, 0.0]

    def test_seam_charge_nonzero_at_real_dt_boundary(self) -> None:
        """Seam term is the trapezoidal current integral when ``dt > 0``."""
        df = pl.DataFrame(
            {
                "current": [2.0, 2.0, -1.0, -1.0],
                "time": [0.0, 3600.0, 7200.0, 10800.0],
                "key": [0, 0, 1, 1],
            }
        )
        result = df.select(
            _seam_charge(pl.col("current"), pl.col("time"), pl.col("key")).alias("seam")
        )["seam"].to_list()
        assert result == pytest.approx([0.0, 0.0, 0.5, 0.0])

    def test_global_net_from_step_ch_dch_includes_seam(
        self, seam_boundary_df: pl.DataFrame
    ) -> None:
        """Seam-corrected net matches the true signed current integral.

        Per-step signed totals alone give ``2 - 1 = 1``; the seam-corrected
        reconstruction must include the ``0.5`` seam charge, giving ``1.5``.
        """
        recipe = _global_net_from_step_ch_dch(
            BDF.STEP_CHARGING_CAPACITY_AH,
            BDF.STEP_DISCHARGING_CAPACITY_AH,
            BDF.CURRENT_AMPERE,
            BDF.TEST_TIME_SECOND,
            BDF.STEP_COUNT,
        )
        result = _compute_series(
            seam_boundary_df,
            recipe.compute(
                _recipe_mapping(
                    BDF.STEP_CHARGING_CAPACITY_AH,
                    BDF.STEP_DISCHARGING_CAPACITY_AH,
                    BDF.CURRENT_AMPERE,
                    BDF.TEST_TIME_SECOND,
                    BDF.STEP_COUNT,
                )
            ),
        )
        assert result == pytest.approx([0.0, 2.0, 2.5, 1.5])

    def test_global_cumulative_from_step_ch_dch_includes_seam(
        self, seam_boundary_df: pl.DataFrame
    ) -> None:
        """Seam-corrected cumulative throughput includes the seam magnitude."""
        recipe = _global_cumulative_from_step_ch_dch(
            BDF.STEP_CHARGING_CAPACITY_AH,
            BDF.STEP_DISCHARGING_CAPACITY_AH,
            BDF.CURRENT_AMPERE,
            BDF.TEST_TIME_SECOND,
            BDF.STEP_COUNT,
        )
        result = _compute_series(
            seam_boundary_df,
            recipe.compute(
                _recipe_mapping(
                    BDF.STEP_CHARGING_CAPACITY_AH,
                    BDF.STEP_DISCHARGING_CAPACITY_AH,
                    BDF.CURRENT_AMPERE,
                    BDF.TEST_TIME_SECOND,
                    BDF.STEP_COUNT,
                )
            ),
        )
        assert result == pytest.approx([0.0, 2.0, 2.5, 3.5])

    def test_direct_charging_discharging_identity_resolves_net_capacity(self) -> None:
        """NET_CAPACITY_AH resolves directly from global charging/discharging."""
        cs = ColumnDict(["Charging Capacity / Ah", "Discharging Capacity / Ah"])
        df = pl.DataFrame(
            {
                "Charging Capacity / Ah": [0.0, 3.0, 3.0],
                "Discharging Capacity / Ah": [0.0, 0.0, 2.0],
            }
        )
        result = df.select(cs.resolve(BDF.NET_CAPACITY_AH))["Net Capacity / Ah"]
        assert result.to_list() == [0.0, 3.0, 1.0]

    def test_direct_charging_discharging_identity_resolves_net_energy(self) -> None:
        """NET_ENERGY_WH resolves directly from global charging/discharging."""
        cs = ColumnDict(["Charging Energy / Wh", "Discharging Energy / Wh"])
        df = pl.DataFrame(
            {
                "Charging Energy / Wh": [0.0, 6.0, 6.0],
                "Discharging Energy / Wh": [0.0, 0.0, 4.0],
            }
        )
        result = df.select(cs.resolve(BDF.NET_ENERGY_WH))["Net Energy / Wh"]
        assert result.to_list() == [0.0, 6.0, 2.0]

    def test_direct_charging_discharging_identity_resolves_cumulative_capacity(
        self,
    ) -> None:
        """CUMULATIVE_CAPACITY_AH resolves directly from global charging/discharging."""
        cs = ColumnDict(["Charging Capacity / Ah", "Discharging Capacity / Ah"])
        df = pl.DataFrame(
            {
                "Charging Capacity / Ah": [0.0, 3.0, 3.0],
                "Discharging Capacity / Ah": [0.0, 0.0, 2.0],
            }
        )
        result = df.select(cs.resolve(BDF.CUMULATIVE_CAPACITY_AH))[
            "Cumulative Capacity / Ah"
        ]
        assert result.to_list() == [0.0, 3.0, 5.0]

    def test_direct_identity_does_not_require_step_or_cycle_columns(self) -> None:
        """Direct global-column recipe needs no step/cycle-scoped dependency."""
        cs = ColumnDict(["Charging Capacity / Ah", "Discharging Capacity / Ah"])
        assert cs.can_resolve("Net Capacity / Ah") is True

    def test_step_scope_net_capacity_resolves_without_current_or_time(self) -> None:
        """STEP_NET_CAPACITY_AH is unaffected by the seam correction."""
        cs = ColumnDict(
            ["Step Charging Capacity / Ah", "Step Discharging Capacity / Ah"]
        )
        assert cs.can_resolve("Step Net Capacity / Ah") is True

    def test_cycle_scope_net_capacity_resolves_without_current_or_time(self) -> None:
        """CYCLE_NET_CAPACITY_AH is unaffected by the seam correction."""
        cs = ColumnDict(
            ["Cycle Charging Capacity / Ah", "Cycle Discharging Capacity / Ah"]
        )
        assert cs.can_resolve("Cycle Net Capacity / Ah") is True

    def test_seam_recipe_falls_through_when_current_and_time_missing(self) -> None:
        """Missing current/time falls through to another recipe, not silent drift."""
        cs = ColumnDict(
            [
                "Step Charging Capacity / Ah",
                "Step Discharging Capacity / Ah",
                "Cumulative Capacity / Ah",
                "Current / A",
            ]
        )
        assert cs.can_resolve("Net Capacity / Ah") is True
        expr = cs.resolve(BDF.NET_CAPACITY_AH)
        df = pl.DataFrame(
            {
                "Step Charging Capacity / Ah": [0.0, 1.0],
                "Step Discharging Capacity / Ah": [0.0, 0.0],
                "Cumulative Capacity / Ah": [0.0, 1.0],
                "Current / A": [1.0, 1.0],
            }
        )
        assert df.select(expr)["Net Capacity / Ah"].to_list() == [0.0, 1.0]

    def test_seam_recipe_raises_when_no_recipe_resolves(self) -> None:
        """ColumnResolutionError raised when no recipe's dependencies resolve."""
        cs = ColumnDict(
            ["Step Charging Capacity / Ah", "Step Discharging Capacity / Ah"]
        )
        assert cs.can_resolve("Net Capacity / Ah") is False
        with pytest.raises(ColumnResolutionError):
            cs.resolve(BDF.NET_CAPACITY_AH)


class TestTrapzIntegralRecipe:
    """Tests for the raw current/power-time trapezoidal integral recipes."""

    def test_net_capacity_resolves_from_current_and_time_only(self) -> None:
        """NET_CAPACITY_AH resolves from just current and elapsed time."""
        cs = ColumnDict(["Current / A", "Test Time / s"])
        assert cs.can_resolve("Net Capacity / Ah") is True
        df = pl.DataFrame(
            {
                "Current / A": [2.0, 2.0, -1.0, -1.0],
                "Test Time / s": [0.0, 3600.0, 7200.0, 10800.0],
            }
        )
        result = df.select(cs.resolve(BDF.NET_CAPACITY_AH))["Net Capacity / Ah"]
        assert result.to_list() == pytest.approx([0.0, 2.0, 2.5, 1.5])

    def test_net_energy_resolves_from_power_and_time_only(self) -> None:
        """NET_ENERGY_WH resolves from just power and elapsed time."""
        cs = ColumnDict(["Power / W", "Test Time / s"])
        assert cs.can_resolve("Net Energy / Wh") is True
        df = pl.DataFrame(
            {
                "Power / W": [2.0, 2.0, -1.0, -1.0],
                "Test Time / s": [0.0, 3600.0, 7200.0, 10800.0],
            }
        )
        result = df.select(cs.resolve(BDF.NET_ENERGY_WH))["Net Energy / Wh"]
        assert result.to_list() == pytest.approx([0.0, 2.0, 2.5, 1.5])

    def test_trapz_integral_is_lowest_priority_for_net_capacity(self) -> None:
        """A recipe using recorded charge data outranks the raw trapz integral."""
        cs = ColumnDict(["Cumulative Capacity / Ah", "Current / A", "Test Time / s"])
        df = pl.DataFrame(
            {
                "Cumulative Capacity / Ah": [0.0, 1.0],
                "Current / A": [1.0, 1.0],
                "Test Time / s": [0.0, 3600.0],
            }
        )
        result = df.select(cs.resolve(BDF.NET_CAPACITY_AH))["Net Capacity / Ah"]
        assert result.to_list() == pytest.approx([0.0, 1.0])


class TestChargeDischargeFromNet:
    """Tests for the from-net charging/discharging component recipes."""

    @pytest.mark.parametrize(
        "target,expected",
        [
            (BDF.CHARGING_CAPACITY_AH, [0.0, 1.0, 2.0, 3.0, 3.0, 3.0, 3.0, 3.0]),
            (BDF.DISCHARGING_CAPACITY_AH, [0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 4.0, 6.0]),
        ],
    )
    def test_global_component_from_net_only_resolves(
        self, target: BDF, expected: list[float]
    ) -> None:
        """Global charging/discharging resolve from net capacity alone."""
        cs = ColumnDict(["Net Capacity / Ah"])
        df = pl.DataFrame(
            {"Net Capacity / Ah": [0.0, 1.0, 2.0, 3.0, 3.0, 1.0, -1.0, -3.0]}
        )
        result = df.select(cs.resolve(target))[target.name]
        assert result.to_list() == pytest.approx(expected)

    @pytest.mark.parametrize(
        "target", [BDF.CHARGING_ENERGY_WH, BDF.DISCHARGING_ENERGY_WH]
    )
    def test_global_energy_component_from_net_only_resolves(self, target: BDF) -> None:
        """Global charging/discharging energy resolve from net energy alone."""
        cs = ColumnDict(["Net Energy / Wh"])
        assert cs.can_resolve(target) is True

    @pytest.mark.parametrize(
        "target,expected",
        [
            (BDF.STEP_CHARGING_CAPACITY_AH, [0.0, 0.0, 0.0, 0.0, 1.0, 2.0]),
            (BDF.STEP_DISCHARGING_CAPACITY_AH, [0.0, 2.0, 4.0, 0.0, 0.0, 0.0]),
        ],
    )
    def test_step_component_from_scoped_net_ignores_reset_jump(
        self, target: BDF, expected: list[float]
    ) -> None:
        """The step-boundary reset of a scoped net creates no spurious charge.

        Step net jumps from -4 back to 0 at the boundary; a diff that is not
        scoped to the step key would book that +4 jump as charging.
        """
        cs = ColumnDict(["Step Net Capacity / Ah", "Step Count / 1"])
        df = pl.DataFrame(
            {
                "Step Net Capacity / Ah": [0.0, -2.0, -4.0, 0.0, 1.0, 2.0],
                "Step Count / 1": [0, 0, 0, 1, 1, 1],
            }
        )
        result = df.select(cs.resolve(target))[target.name]
        assert result.to_list() == pytest.approx(expected)

    def test_resolution_prefers_recipe_with_recorded_inputs(self) -> None:
        """A recipe fed by recorded columns outranks a derivation chain.

        With zero current, the trapz-of-current chain behind the scope_reset
        recipe would yield all zeros; the recorded step net must win.
        """
        cs = ColumnDict(
            [
                "Step Net Capacity / Ah",
                "Step Count / 1",
                "Current / A",
                "Test Time / s",
            ]
        )
        df = pl.DataFrame(
            {
                "Step Net Capacity / Ah": [0.0, 1.0, 2.0],
                "Step Count / 1": [0, 0, 0],
                "Current / A": [0.0, 0.0, 0.0],
                "Test Time / s": [0.0, 3600.0, 7200.0],
            }
        )
        result = df.select(cs.resolve(BDF.STEP_CHARGING_CAPACITY_AH))[
            BDF.STEP_CHARGING_CAPACITY_AH.name
        ]
        assert result.to_list() == pytest.approx([0.0, 1.0, 2.0])


class TestSplitQuantityUnit:
    """Tests for _split_quantity_unit helper."""

    @pytest.mark.parametrize(
        "name,expected_quantity,expected_unit",
        [
            ("Current / A", "Current", "A"),
            ("Unix Time", "Unix Time", None),
            ("Net Capacity  /  Ah", "Net Capacity", "Ah"),
            ("Step Count / 1", "Step Count", "1"),
        ],
    )
    def test_split_quantity_unit(
        self, name: str, expected_quantity: str, expected_unit: str | None
    ) -> None:
        """Split column name into quantity and unit."""
        q, u = _split_quantity_unit(name, BDF_PATTERN)
        assert q == expected_quantity
        assert u == expected_unit


class TestResolveUnit:
    """Tests for _resolve_unit temperature unit resolution."""

    @pytest.mark.parametrize(
        "raw_unit,quantity,expected",
        [
            ("C", "Ambient Temperature", "degC"),
            ("C", "Surface Temperature T1", "degC"),
            ("C", "Temperature", "degC"),
            ("C", "TEMPERATURE", "degC"),
            ("C", "Some Temperature", "degC"),
            ("C", "tEmPeRaTuRe", "degC"),
            ("C", "some_temperature_value", "degC"),
            ("C", "temperatureSensor", "degC"),
            ("C", "Charge", "C"),
            ("C", "Current", "C"),
            ("C", "Capacitance", "C"),
            ("C", "Cycle Count", "C"),
            ("C", "", "C"),
            ("A", "Current", "A"),
            ("A", "Ambient Temperature", "A"),
            ("V", "Voltage", "V"),
            ("V", "Temperature", "V"),
            ("Ah", "Charge Capacity", "Ah"),
            ("K", "Temperature", "K"),
            ("degC", "Ambient Temperature", "degC"),
        ],
    )
    def test_resolve_unit(self, raw_unit: str, quantity: str, expected: str) -> None:
        """_resolve_unit returns degC for 'C' with temperature quantities."""
        assert _resolve_unit(raw_unit, quantity) == expected


class TestApplyConversion:
    """Tests for _apply_conversion unit conversion expression builder."""

    @pytest.mark.parametrize(
        "values,factor,offset,alias,expected",
        [
            ([1.0, 2.0, 3.0], 1.0, 0.0, "result", [1.0, 2.0, 3.0]),
            ([1.0, 2.0, 5.0], 1000.0, 0.0, "result", [1000.0, 2000.0, 5000.0]),
            ([0.0, 25.0, 100.0], 1.0, 273.15, "result", [273.15, 298.15, 373.15]),
            ([0.0, 10.0, 20.0], 2.0, 5.0, "result", [5.0, 25.0, 45.0]),
            ([-1.0, 0.0, 1.0], 1000.0, 0.0, "result", [-1000.0, 0.0, 1000.0]),
            ([1000.0, 2000.0, 500.0], 0.001, 0.0, "result", [1.0, 2.0, 0.5]),
            ([0.0, 0.0, 0.0], 1000.0, 273.15, "result", [273.15, 273.15, 273.15]),
            ([1e6, 1e7, 1e8], 0.001, 0.0, "result", [1e3, 1e4, 1e5]),
            ([2.0, 4.0, 6.0], 3.0, 0.0, "result", [6.0, 12.0, 18.0]),
            ([0.0, 10.0, 20.0], 1.0, 5.0, "result", [5.0, 15.0, 25.0]),
        ],
    )
    def test_apply_conversion(
        self,
        values: list[float],
        factor: float,
        offset: float,
        alias: str,
        expected: list[float],
    ) -> None:
        """_apply_conversion applies factor/offset and aliases the result."""
        df = pl.DataFrame({"x": values})
        result = df.select(_apply_conversion(pl.col("x"), factor, offset, alias))
        assert result.columns == [alias]
        assert result[alias].to_list() == pytest.approx(expected, rel=1e-9)

    def test_apply_conversion_integer_input(self) -> None:
        """Integer input is cast to Float64 before conversion."""
        df = pl.DataFrame({"x": [1, 2, 3]})
        result = df.select(_apply_conversion(pl.col("x"), 1000.0, 0.0, "result"))
        assert result["result"].to_list() == pytest.approx([1000.0, 2000.0, 3000.0])

    def test_apply_conversion_empty_dataframe(self) -> None:
        """Empty DataFrame is handled correctly."""
        df = pl.DataFrame({"x": pl.Series([], dtype=pl.Float64)})
        result = df.select(_apply_conversion(pl.col("x"), 1.0, 0.0, "result"))
        assert result["result"].to_list() == []


class TestRecipeDataclass:
    """Tests for Recipe dataclass."""

    def test_recipe_construction(self) -> None:
        """Recipe can be constructed with required BDFColumn list and compute."""
        recipe = Recipe(
            required=[BDF.CURRENT_AMPERE],
            compute=lambda cols: cols[BDF.CURRENT_AMPERE] * pl.lit(2),
        )
        assert recipe.required == [BDF.CURRENT_AMPERE]
        assert callable(recipe.compute)

    def test_recipe_with_multiple_dependencies(self) -> None:
        """Recipe factories fill required from their column arguments."""
        recipe = _global_net_from_step_ch_dch(
            BDF.STEP_CHARGING_CAPACITY_AH,
            BDF.STEP_DISCHARGING_CAPACITY_AH,
            BDF.CURRENT_AMPERE,
            BDF.TEST_TIME_SECOND,
            BDF.STEP_COUNT,
        )
        assert len(recipe.required) == 5
        assert BDF.STEP_CHARGING_CAPACITY_AH in recipe.required
        assert BDF.STEP_DISCHARGING_CAPACITY_AH in recipe.required

    def test_unused_required_column_raises(self) -> None:
        """Recipe raises ValueError if a required column is never accessed."""
        col_a = cast(BDF, BDFColumn("Level A", "1"))
        col_b = cast(BDF, BDFColumn("Level B", "1"))

        def only_uses_a(cols: dict[BDF, pl.Expr]) -> pl.Expr:
            return cols[col_a] + pl.lit(10)

        with pytest.raises(ValueError, match="unused required"):
            Recipe(required=cast(list[BDF], [col_a, col_b]), compute=only_uses_a)

    def test_undeclared_dependency_raises(self) -> None:
        """Recipe raises ValueError if compute accesses a column not in required."""
        col_a = cast(BDF, BDFColumn("Level A", "1"))
        col_b = cast(BDF, BDFColumn("Level B", "1"))

        def uses_b(cols: dict[BDF, pl.Expr]) -> pl.Expr:
            return cols[col_b] + pl.lit(10)

        with pytest.raises(ValueError, match="not in required"):
            Recipe(required=cast(list[BDF], [col_a]), compute=uses_b)

    def test_valid_recipe_construction_succeeds(self) -> None:
        """Recipe construction succeeds when all required columns are used."""
        col_a = cast(BDF, BDFColumn("Level A", "1"))

        def uses_a(cols: dict[BDF, pl.Expr]) -> pl.Expr:
            return cols[col_a] + pl.lit(10)

        recipe = Recipe(required=cast(list[BDF], [col_a]), compute=uses_a)
        assert len(recipe.required) == 1


class TestColumnSetResolve:
    """Tests for ColumnDict.resolve() method."""

    def test_resolve_with_string(self) -> None:
        """String input returns pl.col() for the parsed column name."""
        cs = ColumnDict(["Current / A"])
        expr = cs.resolve("Current / A")
        df = pl.DataFrame({"Current / A": [1.0, 2.0]})
        result = df.select(expr).to_series().to_list()
        assert result == [1.0, 2.0]

    def test_resolve_with_column_instance(self) -> None:
        """Column descriptor input returns pl.col() expression."""
        cs = ColumnDict(["Current / A"])
        col = column_factory_from_string("Current / A")
        expr = cs.resolve(col)
        df = pl.DataFrame({"Current / A": [3.0]})
        result = df.select(expr).to_series().to_list()
        assert result == [3.0]

    def test_resolve_with_bdf_column_exact_match(self) -> None:
        """BDFColumn exact match returns pl.col() expression."""
        cs = ColumnDict(["Current / A"])
        expr = cs.resolve(BDF.CURRENT_AMPERE)
        df = pl.DataFrame({"Current / A": [5.0]})
        result = df.select(expr).to_series().to_list()
        assert result == [5.0]

    def test_resolve_unit_conversion(self) -> None:
        """Unit conversion with Column descriptor scales values."""
        col = Column("Quantity", "mA")
        cs = ColumnDict(["Quantity / A"])
        expr = cs.resolve(col)
        df = pl.DataFrame({"Quantity / A": [1.0, 2.0]})
        result_df = df.select(expr)
        assert "Quantity / mA" in result_df.columns
        assert result_df["Quantity / mA"].to_list() == pytest.approx(
            [1000.0, 2000.0], rel=1e-9
        )

    def test_resolve_identity_conversion(self) -> None:
        """Same-unit conversion aliases without arithmetic."""
        cs = ColumnDict(["Current / A"])
        expr = cs.resolve("Current / A")
        df = pl.DataFrame({"Current / A": [1.0, 2.0]})
        result_df = df.select(expr)
        assert "Current / A" in result_df.columns
        assert result_df["Current / A"].to_list() == [1.0, 2.0]

    def test_resolve_not_found_raises(self) -> None:
        """ColumnResolutionError raised when column cannot be resolved."""
        cs = ColumnDict(["Voltage / V"])
        with pytest.raises(ColumnResolutionError, match="Cannot resolve"):
            cs.resolve(BDF.CURRENT_AMPERE)

    def test_resolve_empty_available_raises(self) -> None:
        """Empty available_columns list raises ColumnResolutionError for BDFColumn."""
        cs = ColumnDict([])
        with pytest.raises(ColumnResolutionError, match="Cannot resolve"):
            cs.resolve(BDF.CURRENT_AMPERE)

    def test_resolve_recipe_with_unit_conversion(self) -> None:
        """resolve() via recipe then converts the result to the requested unit."""
        df = pl.DataFrame(
            {
                "Step Charging Capacity / Ah": [0.0, 0.0, 0.0],
                "Step Discharging Capacity / Ah": [0.1, 0.2, 0.3],
                "Current / A": [0.0, 0.0, 0.0],
                "Test Time / s": [0.0, 1.0, 2.0],
                "Step Count / 1": [0, 0, 0],
            }
        )
        cs = ColumnDict(df.columns)
        expr = cs.resolve("Net Capacity / mAh")
        base = _global_net_from_step_ch_dch(
            BDF.STEP_CHARGING_CAPACITY_AH,
            BDF.STEP_DISCHARGING_CAPACITY_AH,
            BDF.CURRENT_AMPERE,
            BDF.TEST_TIME_SECOND,
            BDF.STEP_COUNT,
        ).compute(
            {
                BDF.STEP_CHARGING_CAPACITY_AH: pl.col("Step Charging Capacity / Ah"),
                BDF.STEP_DISCHARGING_CAPACITY_AH: pl.col(
                    "Step Discharging Capacity / Ah"
                ),
                BDF.CURRENT_AMPERE: pl.col("Current / A"),
                BDF.TEST_TIME_SECOND: pl.col("Test Time / s"),
                BDF.STEP_COUNT: pl.col("Step Count / 1"),
            }
        )
        assert_frame_equal(
            df.select(expr),
            df.select((base * 1000).alias("Net Capacity / mAh")),
        )

    def test_resolve_non_standard_unit_recipe_deps(self) -> None:
        """resolve() works when recipe inputs are in non-standard units (mAh)."""
        cs = ColumnDict(
            [
                "Step Charging Capacity / mAh",
                "Step Discharging Capacity / mAh",
                "Current / A",
                "Test Time / s",
                "Step Count / 1",
            ]
        )
        expr = cs.resolve("Net Capacity / mAh")
        df = pl.DataFrame(
            {
                "Step Charging Capacity / mAh": [500.0, 1000.0],
                "Step Discharging Capacity / mAh": [0.0, 0.0],
                "Current / A": [0.0, 0.0],
                "Test Time / s": [0.0, 1.0],
                "Step Count / 1": [0, 0],
            }
        )
        result = df.select(expr)
        assert "Net Capacity / mAh" in result.columns
        assert len(result) == 2

    def test_resolve_alias_is_converted_name(self) -> None:
        """resolve() aliases the output to the requested unit name, not the source."""
        cs = ColumnDict(["Current / A"])
        df = pl.DataFrame({"Current / A": [1.0]})
        result = df.select(cs.resolve("Current / mA"))
        assert "Current / mA" in result.columns
        assert "Current / A" not in result.columns

    @pytest.mark.parametrize(
        "values,expected",
        [
            ([0.0, 1.0, -1.0], [0.0, 1000.0, -1000.0]),
            ([1e6, 1e7], [1e9, 1e10]),
            ([-5.0, -2.5], [-5000.0, -2500.0]),
        ],
    )
    def test_resolve_unit_conversion_edge_values(
        self, values: list[float], expected: list[float]
    ) -> None:
        """Unit conversion handles zero, large, and negative values correctly."""
        cs = ColumnDict(["Current / A"])
        df = pl.DataFrame({"Current / A": values})
        result = df.select(cs.resolve(Column("Current", "mA"))).to_series().to_list()
        assert result == pytest.approx(expected, rel=1e-9)

    def test_resolve_empty_dataframe(self) -> None:
        """resolve() on an empty DataFrame returns an empty series."""
        cs = ColumnDict(["Current / A"])
        df = pl.DataFrame({"Current / A": pl.Series([], dtype=pl.Float64)})
        result = df.select(cs.resolve("Current / A")).to_series().to_list()
        assert result == []

    def test_resolve_custom_column_exact_match(self) -> None:
        """resolve() returns exact column when custom column matches."""
        df = pl.DataFrame({"Custom Column / A": [10.0, 20.0, 30.0]})
        column_set = ColumnDict(df.columns)
        resolved_expr = column_set.resolve("Custom Column / A")
        expected_expr = pl.col("Custom Column / A")
        assert_frame_equal(df.select(resolved_expr), df.select(expected_expr))

    def test_resolve_custom_column_with_unit_conversion(self) -> None:
        """resolve() applies unit conversion for custom columns."""
        df = pl.DataFrame({"Custom Column / A": [10.0, 20.0, 30.0]})
        column_set = ColumnDict(df.columns)
        resolved_expr = column_set.resolve("Custom Column / mA")
        expected_expr = (pl.col("Custom Column / A") * 1000).alias("Custom Column / mA")
        assert_frame_equal(df.select(resolved_expr), df.select(expected_expr))

    def test_resolve_bdf_column_with_unit_conversion(self) -> None:
        """resolve() applies unit conversion for BDF columns."""
        df = pl.DataFrame({"Voltage / V": [3.7, 3.6, 3.5]})
        column_set = ColumnDict(df.columns)
        resolved_expr = column_set.resolve("Voltage / mV")
        expected_expr = (pl.col("Voltage / V") * 1000).alias("Voltage / mV")
        assert_frame_equal(df.select(resolved_expr), df.select(expected_expr))

    def test_resolve_bdf_column_via_recipe(self) -> None:
        """resolve() computes BDF column via recipe when not directly available."""
        df = pl.DataFrame(
            {
                "Step Charging Capacity / Ah": [0.0, 0.0, 0.0],
                "Step Discharging Capacity / Ah": [0.1, 0.2, 0.3],
                "Current / A": [0.0, 0.0, 0.0],
                "Test Time / s": [0.0, 1.0, 2.0],
                "Step Count / 1": [0, 0, 0],
            }
        )
        column_set = ColumnDict(df.columns)
        resolved_expr = column_set.resolve(BDF.NET_CAPACITY_AH)
        expected_expr = (
            _global_net_from_step_ch_dch(
                BDF.STEP_CHARGING_CAPACITY_AH,
                BDF.STEP_DISCHARGING_CAPACITY_AH,
                BDF.CURRENT_AMPERE,
                BDF.TEST_TIME_SECOND,
                BDF.STEP_COUNT,
            )
            .compute(
                {
                    BDF.STEP_CHARGING_CAPACITY_AH: pl.col(
                        "Step Charging Capacity / Ah"
                    ),
                    BDF.STEP_DISCHARGING_CAPACITY_AH: pl.col(
                        "Step Discharging Capacity / Ah"
                    ),
                    BDF.CURRENT_AMPERE: pl.col("Current / A"),
                    BDF.TEST_TIME_SECOND: pl.col("Test Time / s"),
                    BDF.STEP_COUNT: pl.col("Step Count / 1"),
                }
            )
            .alias(BDF.NET_CAPACITY_AH.name)
        )
        assert_frame_equal(df.select(resolved_expr), df.select(expected_expr))


class TestColumnRelations:
    """Tests for equality and identity between Column and BDFColumn instances."""

    def test_equality_and_identity(self) -> None:
        """BDFColumn instances with same quantity/unit are equal but not identical."""
        col1 = BDFColumn("Current", "A")
        col2 = BDFColumn("Current", "A")
        assert col1 == col2
        assert col1 is not col2

    def test_equality_in_different_classes(self) -> None:
        """BDFColumn and Column with same quantity/unit are not equal."""
        assert BDFColumn("Voltage", "V") != Column("Voltage", "V")

    def test_in_list_and_set(self) -> None:
        """BDFColumn equality holds in lists and sets."""
        col = BDFColumn("Voltage", "V")
        pool = [col, BDFColumn("Current", "A")]
        ref = BDFColumn("Voltage", "V")
        assert ref in pool
        assert ref in {col, BDFColumn("Current", "A")}

    def test_as_dict_keys(self) -> None:
        """BDFColumn instances hash and compare equal as dict keys."""
        col = BDFColumn("Net Capacity", "Ah")
        other = BDFColumn("Step Count", "1")
        d = {col: "Net Capacity Data", other: "Step Count Data"}
        assert BDFColumn("Net Capacity", "Ah") in d
        assert d[BDFColumn("Net Capacity", "Ah")] == "Net Capacity Data"


class TestBDFEnum:
    """Tests for the BDF Enum and its 27 standard column members."""

    def test_member_count(self) -> None:
        """BDF mirrors the non-deprecated COLUMN_ONTOLOGY quantities."""
        from bdf.spec import COLUMN_ONTOLOGY

        ontology_count = sum(1 for _, q in COLUMN_ONTOLOGY if not q.deprecated)
        assert len(list(BDF)) == ontology_count

    def test_all_members_are_bdf_columns(self) -> None:
        """Every BDF member is a BDFColumn instance."""
        for member in BDF:
            assert isinstance(member, BDFColumn)

    def test_default_columns_are_in_bdf(self) -> None:
        """Every entry in DEFAULT_COLUMNS matches a BDF member name."""
        bdf_names = {col.name for col in BDF}
        for name in DEFAULT_COLUMNS:
            assert name in bdf_names

    @pytest.mark.parametrize(
        "quantity,unit,expected",
        [
            ("Test Time", "s", BDF.TEST_TIME_SECOND),
            ("Current", "A", BDF.CURRENT_AMPERE),
            ("Voltage", "V", BDF.VOLTAGE_VOLT),
        ],
    )
    def test_get(self, quantity: str, unit: str, expected: BDF) -> None:
        """BDF.get() returns the correct member for quantity/unit pairs."""
        assert BDF.get(quantity, unit) == expected

    @pytest.mark.parametrize(
        "quantity,unit",
        [
            ("Test Time", "s"),
            ("Current", "A"),
            ("Voltage", "V"),
            ("Net Capacity", "Ah"),
            ("Step Count", "1"),
            ("Step Record Index", "1"),
        ],
    )
    def test_bdf_column_membership(self, quantity: str, unit: str) -> None:
        """BDFColumn instances for BDF quantities are found in the enum."""
        assert BDFColumn(quantity, unit) in BDF


class TestColumnResolvability:
    """Tests for can_resolve and resolve on Column and BDFColumn."""

    # ── can_resolve — positive cases ──────────────────────────────────────────

    @pytest.mark.parametrize(
        "target, available",
        [
            # exact same-unit match
            (
                Column("Column A", "s"),
                {Column("Column A", "s"), Column("Column B", "A")},
            ),
            # exact match in larger set
            (
                Column("Column B", "mA"),
                {Column("Column A", "s"), Column("Column B", "mA")},
            ),
            # BDFColumn exact equality
            (BDFColumn("Net Capacity", "Ah"), {BDFColumn("Net Capacity", "Ah")}),
            # BDFColumn in mixed set
            (
                BDFColumn("Net Capacity", "Ah"),
                {BDFColumn("Net Capacity", "Ah"), Column("Net Capacity", "mAh")},
            ),
            # Column resolves from BDFColumn in available (compatible unit)
            (
                Column("Net Capacity", "mAh"),
                {Column("Column A", "s"), BDFColumn("Net Capacity", "Ah")},
            ),
            # Column with compound pint unit resolves from BDFColumn
            (
                Column("Net Capacity", "mA.h"),
                {Column("Column A", "s"), BDFColumn("Net Capacity", "Ah")},
            ),
            # case-insensitive quantity matching
            (Column("current", "A"), {Column("CURRENT", "A")}),
            # bidirectional: target A from available mA
            (Column("Current", "A"), {Column("Current", "mA")}),
            # bidirectional: target mA from available A
            (Column("Current", "mA"), {Column("Current", "A")}),
            # BDF member from plain Column (same unit)
            (BDF.CURRENT_AMPERE, {Column("Current", "A")}),
            # BDF member from plain Column (different compatible unit)
            (BDF.CURRENT_AMPERE, {Column("Current", "mA")}),
            # BDF member from BDFColumn in available (equality)
            (BDF.CURRENT_AMPERE, {BDFColumn("Current", "A")}),
            # BDF member from mixed available (BDF + plain Column)
            (BDF.CURRENT_AMPERE, {BDFColumn("Voltage", "V"), Column("Current", "A")}),
            # recipe: standard-unit deps (step-level, seam-corrected)
            (
                BDF.NET_CAPACITY_AH,
                {
                    Column("Step Charging Capacity", "Ah"),
                    Column("Step Discharging Capacity", "Ah"),
                    Column("Current", "A"),
                    Column("Test Time", "s"),
                    Column("Step Count", "1"),
                },
            ),
            # recipe: non-standard-unit deps (step-level mAh, seam-corrected)
            (
                BDF.NET_CAPACITY_AH,
                {
                    Column("Step Charging Capacity", "mAh"),
                    Column("Step Discharging Capacity", "mAh"),
                    Column("Current", "A"),
                    Column("Test Time", "s"),
                    Column("Step Count", "1"),
                },
            ),
        ],
    )
    def test_can_resolve(self, target: Column, available: object) -> None:
        """can_resolve returns True for all resolvable combinations."""
        assert target.can_resolve(available) is True  # type: ignore[arg-type]

    # ── can_resolve — negative cases ──────────────────────────────────────────

    @pytest.mark.parametrize(
        "target, available",
        [
            # quantity absent
            (
                Column("Column A", "s"),
                {Column("Column B", "A"), Column("Voltage", "V")},
            ),
            (
                Column("Column B", "mA"),
                {Column("Column A", "s"), Column("Voltage", "V")},
            ),
            (
                Column("Net Capacity", "mAh"),
                {Column("Column A", "s"), Column("Column B", "A")},
            ),
            # incompatible unit
            (Column("Column A", "s"), {Column("Column A", "A")}),
            (BDFColumn("Current", "V"), {Column("Current", "A")}),
            # wrong quantity alongside BDFColumn
            (Column("Voltage", "A"), {BDFColumn("Current", "A")}),
            # BDF recipe with missing deps
            (BDF.NET_CAPACITY_AH, {Column("Voltage", "V")}),
        ],
    )
    def test_cannot_resolve(self, target: Column, available: object) -> None:
        """can_resolve returns False for unresolvable combinations."""
        assert target.can_resolve(available) is False  # type: ignore[arg-type]

    # ── resolve — BDF recipe with exact value checks ───────────────────────────

    @pytest.mark.parametrize(
        "requested, available, expected_scale, df_data",
        [
            # BDF target, Ah deps → base unit (scale 1)
            (
                BDF.NET_CAPACITY_AH,
                {
                    BDF.STEP_DISCHARGING_CAPACITY_AH,
                    BDF.STEP_CHARGING_CAPACITY_AH,
                    BDF.CURRENT_AMPERE,
                    BDF.TEST_TIME_SECOND,
                    BDF.STEP_COUNT,
                },
                1.0,
                {
                    "Step Charging Capacity / Ah": [0, 0, 0],
                    "Step Discharging Capacity / Ah": [0.1, 0.2, 0.3],
                    "Current / A": [0, 0, 0],
                    "Test Time / s": [0, 1, 2],
                    "Step Count / 1": [0, 0, 0],
                },
            ),
            # Column("mAh") target, Ah deps → unit conversion on result
            (
                Column("Net Capacity", "mAh"),
                {
                    BDF.STEP_DISCHARGING_CAPACITY_AH,
                    BDF.STEP_CHARGING_CAPACITY_AH,
                    BDF.CURRENT_AMPERE,
                    BDF.TEST_TIME_SECOND,
                    BDF.STEP_COUNT,
                },
                1000.0,
                {
                    "Step Charging Capacity / Ah": [0, 0, 0],
                    "Step Discharging Capacity / Ah": [0.1, 0.2, 0.3],
                    "Current / A": [0, 0, 0],
                    "Test Time / s": [0, 1, 2],
                    "Step Count / 1": [0, 0, 0],
                },
            ),
            # BDF target, kAh deps → unit conversion of inputs (scale 1000)
            (
                BDF.NET_CAPACITY_AH,
                {
                    Column("Step Discharging Capacity", "kAh"),
                    Column("Step Charging Capacity", "kAh"),
                    BDF.CURRENT_AMPERE,
                    BDF.TEST_TIME_SECOND,
                    BDF.STEP_COUNT,
                },
                1000.0,
                {
                    "Step Charging Capacity / kAh": [0, 0, 0],
                    "Step Discharging Capacity / kAh": [0.1, 0.2, 0.3],
                    "Current / A": [0, 0, 0],
                    "Test Time / s": [0, 1, 2],
                    "Step Count / 1": [0, 0, 0],
                },
            ),
        ],
    )
    def test_resolve_bdf_recipe(
        self,
        requested: Column,
        available: object,
        expected_scale: float,
        df_data: dict[str, object],
    ) -> None:
        """resolve() via recipe returns correctly computed expression."""
        df = pl.DataFrame(df_data)
        expr = requested.resolve(available)  # type: ignore[arg-type]
        base = pl.DataFrame({"Net Capacity / Ah": [0.0, -0.1, -0.2]})
        expected = base.select(
            (pl.col("Net Capacity / Ah") * expected_scale).alias(requested.name)
        )
        assert_frame_equal(df.select(expr), expected)

    @pytest.mark.parametrize(
        "bdf_column",
        [
            BDFColumn("Net Capacity", "Ah"),  # BDFColumn with no matching recipe key
            BDF.TEMPERATURE_T1_CELSIUS,  # BDF member with no recipe defined
        ],
    )
    def test_cannot_resolve_bdf_recipe(self, bdf_column: BDFColumn) -> None:
        """resolve() raises ColumnResolutionError when no recipe matches."""
        with pytest.raises(ColumnResolutionError):
            bdf_column.resolve({BDF.CHARGING_CAPACITY_AH, BDF.DISCHARGING_CAPACITY_AH})

    def test_resolve_raises_for_missing_quantity(self) -> None:
        """resolve() raises ColumnResolutionError when quantity is absent."""
        with pytest.raises(ColumnResolutionError, match="Cannot resolve"):
            Column("Current", "A").resolve({Column("Voltage", "V")})

    def test_resolve_raises_for_incompatible_unit(self) -> None:
        """resolve() raises ColumnResolutionError for incompatible units."""
        with pytest.raises(ColumnResolutionError):
            Column("Current", "V").resolve({Column("Current", "A")})

    def test_resolve_case_insensitive(self) -> None:
        """resolve() matches quantity case-insensitively."""
        expr = Column("current", "A").resolve({Column("CURRENT", "A")})
        df = pl.DataFrame({"CURRENT / A": [3.0]})
        assert df.select(expr).to_series().to_list() == [3.0]

    def test_resolve_bdf_via_bdf_equality(self) -> None:
        """BDF.resolve() with matching BDFColumn available."""
        expr = BDF.CURRENT_AMPERE.resolve({BDFColumn("Current", "A")})
        df = pl.DataFrame({"Current / A": [5.0]})
        assert df.select(expr).to_series().to_list() == [5.0]

    def test_resolve_accepts_column_set(self) -> None:
        """resolve() accepts a ColumnDict directly."""
        cs = ColumnDict(["Current / A"])
        expr = Column("Current", "mA").resolve(cs)
        df = pl.DataFrame({"Current / A": [1.0]})
        assert df.select(expr).to_series().to_list() == [1000.0]

    def test_can_resolve_accepts_column_set(self) -> None:
        """can_resolve() accepts a ColumnDict directly."""
        cs = ColumnDict(["Current / A"])
        assert Column("Current", "mA").can_resolve(cs) is True
        assert Column("Voltage", "V").can_resolve(cs) is False

    def test_resolve_bdf_non_standard_unit_deps_outputs_base_unit(self) -> None:
        """Recipe with mAh deps still outputs Net Capacity / Ah (base unit)."""
        available = {
            Column("Step Charging Capacity", "mAh"),
            Column("Step Discharging Capacity", "mAh"),
            BDF.CURRENT_AMPERE,
            BDF.TEST_TIME_SECOND,
            BDF.STEP_COUNT,
        }
        expr = BDF.NET_CAPACITY_AH.resolve(available)
        df = pl.DataFrame(
            {
                "Step Charging Capacity / mAh": [1000.0, 2000.0],
                "Step Discharging Capacity / mAh": [0.0, 0.0],
                "Current / A": [0.0, 0.0],
                "Test Time / s": [0.0, 1.0],
                "Step Count / 1": [0, 0],
            }
        )
        result = df.select(expr)
        assert "Net Capacity / Ah" in result.columns
        assert len(result) == 2


class TestColumnDictInit:
    """Tests for ColumnDict initialisation and introspection."""

    def test_columndict_repr_uses_new_class_name(self) -> None:
        """repr() uses ColumnDict to reflect mapping-style semantics."""
        cs = ColumnDict(["Current / A", "Custom / 1"])
        assert (
            repr(cs) == "ColumnDict({'Current / A': BDF.CURRENT_AMPERE, "
            "'Custom / 1': Column(quantity='Custom', unit='1')})"
        )

    @pytest.mark.parametrize(
        "available, expected",
        [
            (["Column A / s"], {Column("Column A", "s")}),
            (["Current / A", "Voltage / V"], {BDF.CURRENT_AMPERE, BDF.VOLTAGE_VOLT}),
            (["Current / mA"], {Column("Current", "mA")}),
            (
                ["Discharging Capacity / Ah", "Charging Capacity / Ah"],
                {BDF.DISCHARGING_CAPACITY_AH, BDF.CHARGING_CAPACITY_AH},
            ),
            (
                ["Discharging Capacity / mAh", "Charging Capacity / mAh"],
                {
                    Column("Discharging Capacity", "mAh"),
                    Column("Charging Capacity", "mAh"),
                },
            ),
            (
                ["Discharging Capacity / Ah", "Charging Capacity / kAh"],
                {BDF.DISCHARGING_CAPACITY_AH, Column("Charging Capacity", "kAh")},
            ),
        ],
    )
    def test_internal_columns(
        self, available: list[str], expected: set[Column | BDFColumn]
    ) -> None:
        """values() contains expected Column/BDF instances after init."""
        assert set(ColumnDict(available).values()) == expected

    def test_mapping_getitem(self) -> None:
        """__getitem__ returns parsed descriptors for exact name keys."""
        cs = ColumnDict(["Current / A", "Custom / 1"])
        assert cs["Current / A"] == BDF.CURRENT_AMPERE
        assert cs["Custom / 1"] == Column("Custom", "1")

    def test_mapping_iteration_and_len(self) -> None:
        """Mapping iteration and len operate on column-name keys."""
        cs = ColumnDict(["Current / A", "Voltage / V"])
        assert list(cs) == ["Current / A", "Voltage / V"]
        assert len(cs) == 2
        assert list(cs.keys()) == ["Current / A", "Voltage / V"]

    def test_names_property(self) -> None:
        """Names returns column name strings in order."""
        cs = ColumnDict(["Current / A", "Voltage / V"])
        assert cs.names == ("Current / A", "Voltage / V")

    def test_quantities_property(self) -> None:
        """Quantities returns quantity strings in order."""
        cs = ColumnDict(["Current / A", "Voltage / V"])
        assert cs.quantities == ("Current", "Voltage")

    def test_contains(self) -> None:
        """__contains__ checks by column name string."""
        cs = ColumnDict(["Current / A", "Voltage / V"])
        assert "Current / A" in cs
        assert "Power / W" not in cs

    def test_contains_non_string_returns_false(self) -> None:
        """__contains__ returns False for non-string objects."""
        cs = ColumnDict(["Current / A"])
        assert 42 not in cs
        assert BDF.CURRENT_AMPERE not in cs

    def test_columns_for_quantity_hit(self) -> None:
        """columns_for_quantity returns matching Column descriptors."""
        cs = ColumnDict(["Current / A", "Voltage / V"])
        assert cs.columns_for_quantity("current") == (BDF.CURRENT_AMPERE,)

    def test_columns_for_quantity_case_insensitive(self) -> None:
        """columns_for_quantity is case-insensitive."""
        cs = ColumnDict(["Current / A"])
        assert cs.columns_for_quantity("Current") == (BDF.CURRENT_AMPERE,)
        assert cs.columns_for_quantity("current") == (BDF.CURRENT_AMPERE,)

    def test_columns_for_quantity_multiple(self) -> None:
        """columns_for_quantity returns all columns sharing a quantity."""
        cs = ColumnDict(["Current / A", "Current / mA"])
        result = cs.columns_for_quantity("current")
        assert len(result) == 2
        assert set(result) == {BDF.CURRENT_AMPERE, Column("Current", "mA")}

    def test_columns_for_quantity_missing(self) -> None:
        """columns_for_quantity returns empty tuple for unknown quantity."""
        cs = ColumnDict(["Current / A"])
        assert cs.columns_for_quantity("Voltage") == ()

    @pytest.mark.parametrize(
        "column, expected",
        [
            ("Current / A", True),  # string — direct hit
            ("Current / mA", True),  # string — unit conversion
            ("Voltage / V", False),  # string — missing
            (Column("Current", "A"), True),  # Column — direct hit
            (Column("Current", "mA"), True),  # Column — unit conversion
            (Column("Voltage", "V"), False),  # Column — missing
            (BDF.CURRENT_AMPERE, True),  # BDF member — via unit conversion
        ],
    )
    def test_can_resolve(self, column: object, expected: bool) -> None:
        """can_resolve returns correct boolean values."""
        cs = ColumnDict(["Current / A"])
        assert cs.can_resolve(column) is expected  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "available, column, expected",
        [
            # recipe resolvable (standard units, step-level, seam-corrected)
            (
                [
                    "Step Charging Capacity / Ah",
                    "Step Discharging Capacity / Ah",
                    "Current / A",
                    "Test Time / s",
                    "Step Count / 1",
                ],
                "Net Capacity / Ah",
                True,
            ),
            # recipe resolvable (non-standard units, step-level, seam-corrected)
            (
                [
                    "Step Charging Capacity / mAh",
                    "Step Discharging Capacity / mAh",
                    "Current / A",
                    "Test Time / s",
                    "Step Count / 1",
                ],
                "Net Capacity / Ah",
                True,
            ),
            # recipe + unit conversion on result (step-level, seam-corrected)
            (
                [
                    "Step Charging Capacity / mAh",
                    "Step Discharging Capacity / mAh",
                    "Current / A",
                    "Test Time / s",
                    "Step Count / 1",
                ],
                "Net Capacity / mAh",
                True,
            ),
            # recipe not resolvable (wrong deps)
            (["Voltage / V"], "Net Capacity / Ah", False),
        ],
    )
    def test_can_resolve_recipe(
        self, available: list[str], column: str, expected: bool
    ) -> None:
        """can_resolve handles recipe-based BDF columns correctly."""
        assert ColumnDict(available).can_resolve(column) is expected


class TestIsValidColumnName:
    """Tests for is_valid_column_name."""

    @pytest.mark.parametrize(
        "name, expected",
        [
            ("Current / A", True),
            ("Voltage / mV", True),
            ("Step ID", True),
            ("Step Type", True),
            ("Cell Replaced", False),
            ("cell replaced", False),
            ("Current / nonsense", False),
            ("Extra Quantity / mA", True),
            ("Foo / (((", False),
            ("Cycle / 3", False),
            ("Specific Capacity / mAh/g", True),
            ("dQ/dV / Ah/V", True),
            ("Efficiency/%", False),
            ("Q discharge/mA.h", False),
            ("Cell Note / ", False),
            ("Cell Note /  ", False),
        ],
    )
    def test_is_valid_column_name(self, name: str, expected: bool) -> None:
        """A name passes where a unit or a defined bare column name backs it."""
        assert is_valid_column_name(name) is expected


class TestIsValidColumnNameAgainstCyclerFixtures:
    """Tests for is_valid_column_name over the columns of the sample data files.

    A raw cycler file holds a column the BDF plugin recognises, whose name
    already carries the ``"Quantity / unit"`` form with one space on each
    side of the slash, and a column the plugin does not recognise, whose raw
    header keeps the punctuation the cycler wrote. This class names every
    such column that the sample data fixtures hold, gathered from a scan of
    every fixture with ``include_unknown=True``, and asserts which group the
    name rule keeps and which it drops.
    """

    # A BDF-recognised column of a sample data file. Its name already carries
    # a unit, with one space on each side of the slash, or names a bare BDF
    # quantity that the ontology defines without a unit.
    _RECOGNISED_COLUMNS = (
        "AC Internal Resistance / ohm",
        "Charging Energy / Wh",
        "Cumulative Energy / Wh",
        "Current / A",
        "Cycle Count / 1",
        "DC Internal Resistance / ohm",
        "Discharging Energy / Wh",
        "Internal Resistance / ohm",
        "Net Capacity / Ah",
        "Power / W",
        "Record Index / 1",
        "Schedule Charging Capacity / Ah",
        "Schedule Charging Energy / Wh",
        "Schedule Discharging Capacity / Ah",
        "Schedule Discharging Energy / Wh",
        "Step Charging Capacity / Ah",
        "Step Count / 1",
        "Step Cumulative Capacity / Ah",
        "Step Cumulative Energy / Wh",
        "Step Discharging Capacity / Ah",
        "Step ID",
        "Step Net Energy / Wh",
        "Step Time / s",
        "Step Type",
        "Temperature T1 / degC",
        "Temperature T2 / degC",
        "Test Time / s",
        "Unix Time / s",
        "Voltage / V",
    )

    # A raw column of a sample data file that no BDF plugin maps. Every one
    # of these either holds a slash with no surrounding space, so it carries
    # no separator the rule recognises, or holds no slash at all, and none
    # names a BDF quantity that the ontology defines without a unit.
    _UNRECOGNISED_COLUMNS = (
        "Ah-Cyc-Charge",
        "Ah-Cyc-Charge-0",
        "Ah-Cyc-Discharge",
        "Ah-Cyc-Discharge-0",
        "Ah-Step",
        "Aux_dT/dt_1 (C/s)",
        "Capacitance charge/ï¿½F",
        "Capacitance discharge/ï¿½F",
        "Capacity (Ah)",
        "Capacity/mA.h",
        "Command",
        "Count",
        "Cyc-Count",
        "Efficiency/%",
        "I Range",
        "Level",
        "Ns changes",
        "Q charge/discharge/mA.h",
        "Q charge/mA.h",
        "Q discharge/mA.h",
        "State",
        "T1[ï¿½C]",
        "TC_Counter1",
        "TC_Counter2",
        "TC_Counter3",
        "Temperature/ï¿½C",
        "Wh-Cyc-Charge",
        "Wh-Cyc-Charge-0",
        "Wh-Cyc-Discharge",
        "Wh-Cyc-Discharge-0",
        "Wh-Step",
        "Wh[Wh]",
        "control changes",
        "control/V",
        "control/V/mA",
        "control/mA",
        "counter inc.",
        "dIdt (A/h)",
        "dQ/dV (Ah/V)",
        "dV/dQ (V/Ah)",
        "dV/dt (V/s)",
        "dVdt (V/h)",
        "dq/mA.h",
        "error",
        "half cycle",
        "mAh/g",
        "mode",
        "ox/red",
        "t-Cyc[s]",
        "t-Set[s]",
        "t-Step[s]",
        "x",
    )

    @pytest.mark.parametrize("name", _RECOGNISED_COLUMNS)
    def test_recognised_column_passes(self, name: str) -> None:
        """A column a BDF plugin recognises satisfies the name rule."""
        assert is_valid_column_name(name) is True

    @pytest.mark.parametrize("name", _UNRECOGNISED_COLUMNS)
    def test_unrecognised_column_fails(self, name: str) -> None:
        """A raw column no BDF plugin recognises fails the name rule."""
        assert is_valid_column_name(name) is False

    def test_fixture_column_lists_are_disjoint(self) -> None:
        """The two column lists name no column in common."""
        assert not set(self._RECOGNISED_COLUMNS) & set(self._UNRECOGNISED_COLUMNS)
