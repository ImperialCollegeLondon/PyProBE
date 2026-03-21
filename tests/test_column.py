"""Tests for the column module.

This module provides tests for BDF column abstractions, including parsing,
unit conversion, and Polars expression generation with recipe-based fallbacks
via ColumnSet.
"""

from __future__ import annotations

import polars as pl
import pytest

from pyprobe.column import (
    ALL_COLUMNS,
    BDF_IRI_PREFIX,
    BDF_PATTERN,
    DEFAULT_COLUMNS,
    BDFColumn,
    Column,
    ColumnSet,
    Recipe,
    _apply_conversion,
    _capacity_from_ch_dch,
    _resolve_unit,
    _split_quantity_unit,
    _step_count_from_step_index,
    charging_capacity_ah,
    current_ampere,
    cycle_count,
    discharging_capacity_ah,
    net_capacity_ah,
    step_count,
    step_index,
    temperature_t1_celsius,
    test_time_second,
    unix_time_second,
    voltage_volt,
)


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
        assert col.column_name == expected_name

    def test_init_default_unit_is_dimensionless(self) -> None:
        """Column with no unit arg defaults to '1'."""
        col = Column("Step")
        assert col.unit == "1"
        assert col.column_name == "Step / 1"


class TestColumnFromString:
    """Tests for Column.from_string factory method."""

    @pytest.mark.parametrize(
        "input_str,expected_quantity,expected_unit",
        [
            ("Current / A", "Current", "A"),
            ("Step Count / 1", "Step Count", "1"),
            ("Net Capacity / Ah", "Net Capacity", "Ah"),
            ("Step", "Step", "1"),
            ("Current  /  A", "Current", "A"),
            ("Net Capacity / Ah", "Net Capacity", "Ah"),
        ],
    )
    def test_from_string_parses_correctly(
        self, input_str: str, expected_quantity: str, expected_unit: str
    ) -> None:
        """Parse 'Quantity / unit' string correctly."""
        col = Column.from_string(input_str)
        assert col.quantity == expected_quantity
        assert col.unit == expected_unit

    def test_from_string_roundtrip(self) -> None:
        """Parsing and str() should roundtrip the original name."""
        original = "Net Capacity / Ah"
        col = Column.from_string(original)
        assert str(col) == original

    def test_from_string_invalid_unit_raises_on_conversion(self) -> None:
        """Invalid unit strings raise ValueError at conversion_parameters time."""
        col = Column.from_string("Current / InvalidUnit")
        with pytest.raises(ValueError, match="could not be parsed"):
            col.conversion_parameters("A")


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
        col = Column.from_string(f"Quantity / {source_unit}")
        factor, offset = col.conversion_parameters(target_unit)
        assert factor == pytest.approx(expected_factor, rel=1e-9)
        assert offset == pytest.approx(expected_offset, abs=1e-9)

    def test_conversion_celsius_to_kelvin(self) -> None:
        """Affine conversion degC to K: factor=1, offset=273.15."""
        col = Column.from_string("Temperature / C")
        factor, offset = col.conversion_parameters("K")
        assert factor == pytest.approx(1.0, rel=1e-9)
        assert offset == pytest.approx(273.15, abs=0.01)

    def test_conversion_incompatible_units_raises(self) -> None:
        """Converting between incompatible units raises ValueError."""
        col = Column.from_string("Current / A")
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
            (current_ampere, "current_ampere"),
            (voltage_volt, "voltage_volt"),
            (step_count, "step_count"),
            (cycle_count, "cycle_count"),
            (charging_capacity_ah, "charging_capacity_ampere_hour"),
            (temperature_t1_celsius, "temperature_t1_degree_celsius"),
        ],
    )
    def test_iri_computed_from_quantity_and_unit(
        self, col_obj: BDFColumn, expected_iri_suffix: str
    ) -> None:
        """IRI is computed from quantity and pint long-form unit."""
        assert col_obj.iri == f"{BDF_IRI_PREFIX}{expected_iri_suffix}"

    @pytest.mark.parametrize("col_obj", ALL_COLUMNS)
    def test_all_bdf_column_iris_are_valid_urls(self, col_obj: BDFColumn) -> None:
        """All BDF column IRIs are complete and properly formatted."""
        iri = col_obj.iri
        assert iri.startswith(BDF_IRI_PREFIX)
        assert len(iri) > len(BDF_IRI_PREFIX)
        assert iri.endswith(iri.split("#")[-1])


class TestRecipeComputation:
    """Tests for recipe computation functions."""

    def test_step_count_from_step_index_recipe(self) -> None:
        """_step_count_from_step_index increments on step changes."""
        cs = ColumnSet(["Step Index / 1"])
        df = pl.DataFrame(
            {
                "Step Index / 1": [
                    1,
                    1,
                    2,
                    2,
                    3,
                    3,
                    1,
                    1,
                    2,
                    2,
                    3,
                    3,
                    4,
                    4,
                    4,
                    4,
                    5,
                    5,
                ]
            }
        )
        result = df.select(cs.col(step_count))
        expected = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 6, 6, 7, 7]
        assert result["Step Count / 1"].to_list() == expected

    def test_col_recipe_net_capacity(self) -> None:
        """Recipe resolves Net Capacity from Charging and Discharging Capacity."""
        cs = ColumnSet(["Charging Capacity / Ah", "Discharging Capacity / Ah"])
        df = pl.DataFrame(
            {
                "Charging Capacity / Ah": [1.0, 0.0, 0.0],
                "Discharging Capacity / Ah": [0.0, 1.0, 2.0],
            }
        )
        result = df.select(cs.col(net_capacity_ah))
        expected = [1.0, 0.0, -1.0]
        assert result["Net Capacity / Ah"].to_list() == pytest.approx(expected)

    def test_col_recipe_time_from_unix_time(self) -> None:
        """Recipe resolves Test Time from Unix epoch time in seconds."""
        cs = ColumnSet(["Unix Time / s"])
        df = pl.DataFrame(
            {
                "Unix Time / s": [1648864360.0, 1648864361.0, 1648864362.0],
            }
        )
        result = df.select(cs.col(test_time_second))
        expected = [0.0, 1.0, 2.0]
        assert result["Test Time / s"].to_list() == pytest.approx(expected)


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
            required=[current_ampere],
            compute=lambda cols: cols[current_ampere] * pl.lit(2),
        )
        assert recipe.required == [current_ampere]
        assert callable(recipe.compute)

    def test_recipe_with_multiple_dependencies(self) -> None:
        """Recipe can require multiple BDFColumn instances."""
        recipe = Recipe(
            required=[charging_capacity_ah, discharging_capacity_ah],
            compute=_capacity_from_ch_dch,
        )
        assert len(recipe.required) == 2
        assert charging_capacity_ah in recipe.required
        assert discharging_capacity_ah in recipe.required


class TestBDFColumnInit:
    """Tests for BDFColumn construction with recipes."""

    def test_init_with_recipes(self) -> None:
        """BDFColumn can be initialized with recipes list."""
        recipe = Recipe(
            required=[step_index],
            compute=_step_count_from_step_index,
        )
        col = BDFColumn("Step Count", "1", recipes=[recipe])
        assert len(col.recipes) == 1

    def test_init_default_recipes_is_empty_list(self) -> None:
        """Default recipes is an empty list."""
        col = BDFColumn("Current", "A")
        assert col.recipes == []

    def test_recipes_are_public_attribute(self) -> None:
        """Recipes is a public attribute, not private."""
        col = BDFColumn("Current", "A")
        assert hasattr(col, "recipes")
        col.recipes = [
            Recipe(required=[step_index], compute=_step_count_from_step_index)
        ]
        assert len(col.recipes) == 1


class TestRecipeAttachment:
    """Tests for post-definition recipe attachment pattern."""

    def test_test_time_second_has_recipe(self) -> None:
        """test_time_second has its Unix Time recipe attached."""
        assert len(test_time_second.recipes) == 1
        assert unix_time_second in test_time_second.recipes[0].required

    def test_net_capacity_ah_has_recipe(self) -> None:
        """net_capacity_ah has its Charging/Discharging recipe attached."""
        assert len(net_capacity_ah.recipes) == 1
        required_quantities = {
            col.quantity for col in net_capacity_ah.recipes[0].required
        }
        assert "Charging Capacity" in required_quantities
        assert "Discharging Capacity" in required_quantities

    def test_step_count_has_recipe(self) -> None:
        """step_count has its Step Index recipe attached."""
        assert len(step_count.recipes) == 1
        assert step_index in step_count.recipes[0].required


class TestRecipeValidation:
    """Tests for recipe validation at construction time."""

    def test_unused_required_column_raises(self) -> None:
        """Recipe raises ValueError if a required column is never accessed."""
        col_a = BDFColumn("Level A", "1")
        col_b = BDFColumn("Level B", "1")

        def only_uses_a(cols: dict[BDFColumn, pl.Expr]) -> pl.Expr:
            return cols[col_a] + pl.lit(10)

        with pytest.raises(ValueError, match="unused required"):
            Recipe(required=[col_a, col_b], compute=only_uses_a)

    def test_undeclared_dependency_raises(self) -> None:
        """Recipe raises ValueError if compute accesses a column not in required."""
        col_a = BDFColumn("Level A", "1")
        col_b = BDFColumn("Level B", "1")

        def uses_b(cols: dict[BDFColumn, pl.Expr]) -> pl.Expr:
            return cols[col_b] + pl.lit(10)

        with pytest.raises(ValueError, match="not in required"):
            Recipe(required=[col_a], compute=uses_b)

    def test_valid_recipe_construction_succeeds(self) -> None:
        """Recipe construction succeeds when all required columns are used."""
        col_a = BDFColumn("Level A", "1")

        def uses_a(cols: dict[BDFColumn, pl.Expr]) -> pl.Expr:
            return cols[col_a] + pl.lit(10)

        recipe = Recipe(required=[col_a], compute=uses_a)
        assert len(recipe.required) == 1


class TestColumnSet:
    """Tests for ColumnSet column resolution and unit conversion."""

    def test_col_with_string(self) -> None:
        """String input returns pl.col() for the parsed column name."""
        cs = ColumnSet(["Current / A"])
        expr = cs.col("Current / A")
        df = pl.DataFrame({"Current / A": [1.0, 2.0]})
        result = df.select(expr).to_series().to_list()
        assert result == [1.0, 2.0]

    def test_col_with_column_instance(self) -> None:
        """Column descriptor input returns pl.col() expression."""
        cs = ColumnSet(["Current / A"])
        col = Column.from_string("Current / A")
        expr = cs.col(col)
        df = pl.DataFrame({"Current / A": [3.0]})
        result = df.select(expr).to_series().to_list()
        assert result == [3.0]

    def test_col_with_bdf_column_exact_match(self) -> None:
        """BDFColumn exact match returns pl.col() expression."""
        cs = ColumnSet(["Current / A"])
        expr = cs.col(current_ampere)
        df = pl.DataFrame({"Current / A": [5.0]})
        result = df.select(expr).to_series().to_list()
        assert result == [5.0]

    @pytest.mark.parametrize(
        "source_unit,target_unit,expected_conversion",
        [
            ("A", "mA", 1000.0),
            ("V", "mV", 1000.0),
        ],
    )
    def test_col_with_unit_conversion(
        self, source_unit: str, target_unit: str, expected_conversion: float
    ) -> None:
        """Unit conversion scales values and aliases the result."""
        col = BDFColumn("Quantity", source_unit)
        cs = ColumnSet([f"Quantity / {source_unit}"])
        expr = cs.col(col, unit=target_unit)
        df = pl.DataFrame({f"Quantity / {source_unit}": [1.0, 2.0]})
        result_df = df.select(expr)
        assert f"Quantity / {target_unit}" in result_df.columns
        assert result_df[f"Quantity / {target_unit}"].to_list() == pytest.approx(
            [expected_conversion, expected_conversion * 2], rel=1e-9
        )

    def test_col_identity_conversion(self) -> None:
        """Same-unit conversion aliases without arithmetic."""
        cs = ColumnSet(["Current / A"])
        expr = cs.col(current_ampere, unit="A")
        df = pl.DataFrame({"Current / A": [1.0, 2.0]})
        result_df = df.select(expr)
        assert "Current / A" in result_df.columns
        assert result_df["Current / A"].to_list() == [1.0, 2.0]

    def test_col_celsius_to_kelvin(self) -> None:
        """Affine conversion (degC to K) adds 273.15 offset."""
        col = BDFColumn("Temperature", "degC")
        cs = ColumnSet(["Temperature / degC"])
        expr = cs.col(col, unit="K")
        df = pl.DataFrame({"Temperature / degC": [0.0, 100.0]})
        result = df.select(expr).to_series().to_list()
        assert result == pytest.approx([273.15, 373.15], abs=0.01)

    def test_col_not_found_raises(self) -> None:
        """ValueError raised when column cannot be resolved."""
        cs = ColumnSet(["Voltage / V"])
        with pytest.raises(ValueError, match="Cannot resolve"):
            cs.col(current_ampere)

    def test_col_bdf_with_conversion(self) -> None:
        """BDFColumn exact match combined with unit conversion."""
        cs = ColumnSet(["Voltage / V"])
        expr = cs.col(voltage_volt, unit="mV")
        df = pl.DataFrame({"Voltage / V": [1.0, 2.0]})
        result_df = df.select(expr)
        assert "Voltage / mV" in result_df.columns
        assert result_df["Voltage / mV"].to_list() == pytest.approx(
            [1000.0, 2000.0], rel=1e-9
        )

    def test_col_empty_available_raises(self) -> None:
        """Empty available_columns list raises ValueError for BDFColumn."""
        cs = ColumnSet([])
        with pytest.raises(ValueError, match="Cannot resolve"):
            cs.col(current_ampere)

    def test_recursive_recipe(self) -> None:
        """Recipe dependency resolved recursively via another recipe."""
        level_a = BDFColumn("Level A", "1")
        level_b = BDFColumn("Level B", "1")

        def b_from_a(cols: dict[BDFColumn, pl.Expr]) -> pl.Expr:
            return cols[level_a] + pl.lit(10)

        level_b.recipes = [Recipe(required=[level_a], compute=b_from_a)]
        level_c = BDFColumn("Level C", "1")

        def c_from_b(cols: dict[BDFColumn, pl.Expr]) -> pl.Expr:
            return cols[level_b] * pl.lit(2)

        level_c.recipes = [Recipe(required=[level_b], compute=c_from_b)]

        cs = ColumnSet(["Level A / 1"])
        expr = cs.col(level_c)
        df = pl.DataFrame({"Level A / 1": [5, 10, 15]})
        result = df.select(expr).to_series().to_list()
        assert result == [30, 40, 50]


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.mark.parametrize(
        "values,target_unit,expected",
        [
            ([0.0, 1.0, -1.0], "mA", [0.0, 1000.0, -1000.0]),
            ([1e6, 1e7], "mA", [1e9, 1e10]),
            ([-5.0, -2.5], "mA", [-5000.0, -2500.0]),
        ],
    )
    def test_unit_conversion_edge_values(
        self, values: list[float], target_unit: str, expected: list[float]
    ) -> None:
        """Unit conversion handles zero, large, and negative values."""
        cs = ColumnSet(["Current / A"])
        col = Column.from_string("Current / A")
        df = pl.DataFrame({"Current / A": values})
        result = df.select(cs.col(col, unit=target_unit)).to_series().to_list()
        assert result == pytest.approx(expected, rel=1e-9)

    def test_column_empty_dataframe(self) -> None:
        """Empty DataFrame is handled correctly."""
        cs = ColumnSet(["Current / A"])
        col = Column.from_string("Current / A")
        df = pl.DataFrame({"Current / A": []})
        result = df.select(cs.col(col)).to_series().to_list()
        assert result == []

    def test_columnset_with_many_rows(self) -> None:
        """Large DataFrames are processed correctly."""
        cs = ColumnSet(["Current / A"])
        col = Column.from_string("Current / A")
        large_data = list(range(10000))
        df = pl.DataFrame({"Current / A": large_data})
        result = df.select(cs.col(col, unit="mA")).to_series().to_list()
        assert len(result) == 10000
        assert result[0] == 0.0
        assert result[-1] == pytest.approx(9999000.0, rel=1e-9)


class TestPublicBDFInstances:
    """Tests for all 27 public BDFColumn instances."""

    def test_all_columns_count(self) -> None:
        """ALL_COLUMNS list contains exactly 27 entries."""
        assert len(ALL_COLUMNS) == 27

    def test_default_columns_is_subset(self) -> None:
        """DEFAULT_COLUMNS are all present in ALL_COLUMNS."""
        all_names = [col.column_name for col in ALL_COLUMNS]
        for default_name in DEFAULT_COLUMNS:
            assert default_name in all_names

    @pytest.mark.parametrize("col_obj", ALL_COLUMNS)
    def test_all_instances_in_all_columns_list(self, col_obj: BDFColumn) -> None:
        """All exported instances appear in ALL_COLUMNS."""
        assert col_obj in ALL_COLUMNS

    @pytest.mark.parametrize("col_obj", ALL_COLUMNS)
    def test_all_instances_have_iri(self, col_obj: BDFColumn) -> None:
        """All BDF-standard instances have IRI URLs starting with BDF_IRI_PREFIX."""
        assert col_obj.iri is not None
        assert col_obj.iri.startswith(BDF_IRI_PREFIX)
