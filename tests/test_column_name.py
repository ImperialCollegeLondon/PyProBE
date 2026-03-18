"""Tests for the ColumnName class."""

import polars as pl
import pytest

from pyprobe.bdf import ALL_COLUMNS
from pyprobe.column_name import FORMAT_REGISTRY, ColumnName, _ureg

BDF = FORMAT_REGISTRY["bdf"]
BRACKET = FORMAT_REGISTRY["square_bracket"]
PARENTHESES = FORMAT_REGISTRY["parentheses"]
NEWARE = FORMAT_REGISTRY["neware"]
BASYTEC = FORMAT_REGISTRY["basytec"]
BIOLOGIC = FORMAT_REGISTRY["biologic"]


def _assert_unit_converts(unit, canonical: str) -> None:
    """Assert unit converts 1:1 to canonical."""
    assert _ureg.Quantity(1, unit).to(canonical).magnitude == pytest.approx(1.0)


PARSE_CASES = [
    # Standard formats
    ("Current [A]", BRACKET, "Current", "A"),
    ("Current / A", BDF, "Current", "A"),
    ("Test Time (s)", PARENTHESES, "Test Time", "s"),
    ("Current(A)", NEWARE, "Current", "A"),
    ("~Time[s]", BASYTEC, "Time", "s"),
    ("I/mA", BIOLOGIC, "I", "mA"),
    ("Step", BRACKET, "Step", None),
    ("Step", BDF, "Step", None),
    ("Step Index", PARENTHESES, "Step Index", None),
    # Unit aliases (via _ureg.define or built-in)
    ("Resistance [Ohms]", BRACKET, "Resistance", "ohm"),
    ("Resistance [Ohm]", BRACKET, "Resistance", "ohm"),
    ("Time [Seconds]", BRACKET, "Time", "s"),
    ("Temperature [°C]", BRACKET, "Temperature", "degC"),
    ("Temperature [C]", BRACKET, "Temperature", "degC"),
    ("Time / sec", BDF, "Time", "s"),
    ("Time [hr]", BRACKET, "Time", "hour"),
    ("Charge [A.h]", BRACKET, "Charge", "Ah"),
    # Special characters in quantity
    ("~SOC [%]", BRACKET, "~SOC", "percent"),
    ("<>Temperature [degC]", BRACKET, "<>Temperature", "degC"),
    ("Charge.Rate / C", BDF, "Charge.Rate", "C"),
    ("~Event", BRACKET, "~Event", None),
    ("Current [A]", BDF, "Current [A]", None),  # bracket in slash = bare
]

PARSE_IDS = [
    "bracket",
    "bdf",
    "parentheses",
    "neware",
    "basytec",
    "biologic",
    "bare_bracket",
    "bare_bdf",
    "bare_parentheses",
    "alias_Ohms",
    "alias_Ohm",
    "alias_Seconds",
    "alias_degC",
    "temperature_C_to_degC",
    "alias_sec",
    "alias_hr",
    "alias_A.h",
    "special_tilde",
    "special_angles",
    "special_dot",
    "bare_tilde",
    "bracket_in_bdf",
]


class TestParsing:
    """ColumnName.__init__, .quantity, .unit, __str__."""

    @pytest.mark.parametrize(
        ("name", "pattern", "expected_quantity", "canonical_unit"),
        PARSE_CASES,
        ids=PARSE_IDS,
    )
    def test_quantity_and_unit(self, name, pattern, expected_quantity, canonical_unit):
        """Parse quantity and unit from column names across all formats."""
        cn = ColumnName(name, pattern)
        assert cn.quantity == expected_quantity
        if canonical_unit is None:
            assert cn.unit is None
        else:
            assert cn.unit is not None
            _assert_unit_converts(cn.unit, canonical_unit)

    def test_str_returns_original(self):
        """__str__ returns the original column name string."""
        assert str(ColumnName("Current [A]", BRACKET)) == "Current [A]"
        assert str(ColumnName("I/mA", BIOLOGIC)) == "I/mA"

    def test_invalid_unit_raises(self):
        """Column with unparseable unit raises ValueError."""
        with pytest.raises(ValueError, match="could not be parsed"):
            ColumnName("Foo [xyz_bad]", BRACKET)

    @pytest.mark.parametrize(
        ("name", "pattern"),
        [("Step /", BDF), ("Step [", BRACKET)],
        ids=["partial_slash", "partial_bracket"],
    )
    def test_partial_separator_raises(self, name, pattern):
        """Names with partial separators (missing unit) raise ValueError."""
        with pytest.raises(ValueError):
            ColumnName(name, pattern)

    def test_ohm_variants_equivalent(self):
        """Different spellings of ohm (Ohm, Ohms) both convert to ohm."""
        cn1 = ColumnName("R [Ohm]", BRACKET)
        cn2 = ColumnName("R [Ohms]", BRACKET)
        _assert_unit_converts(cn1.unit, "ohm")
        _assert_unit_converts(cn2.unit, "ohm")

    def test_temperature_c_is_degc_not_coulombs(self):
        """Temperature [C] resolves to degC, not coulombs.

        Charge [C] resolves to coulombs.
        """
        # Temperature [C] should resolve to degC
        temp_cn = ColumnName("Temperature [C]", BRACKET)
        assert temp_cn.unit is not None
        _assert_unit_converts(temp_cn.unit, "degC")

        # Charge [C] should resolve to coulombs (not affected by the special case)
        charge_cn = ColumnName("Charge [C]", BRACKET)
        assert charge_cn.unit is not None
        _assert_unit_converts(charge_cn.unit, "coulomb")


class TestConversionParameters:
    """ColumnName.conversion_parameters."""

    @pytest.mark.parametrize(
        ("name", "pattern", "target", "factor", "offset"),
        [
            ("Current [A]", BRACKET, "mA", 1000.0, 0.0),
            ("Capacity [Ah]", BRACKET, "mAh", 1000.0, 0.0),
            ("Time [hr]", BRACKET, "s", 3600.0, 0.0),
            ("Charge [A.h]", BRACKET, "mAh", 1000.0, 0.0),
            ("Current [A]", BRACKET, "A", 1.0, 0.0),
            ("Voltage [V]", BRACKET, "V", 1.0, 0.0),
            ("Temperature [degC]", BRACKET, "K", 1.0, 273.15),
            ("Temperature [K]", BRACKET, "degC", 1.0, -273.15),
            ("Temperature [C]", BRACKET, "K", 1.0, 273.15),
        ],
        ids=[
            "A_to_mA",
            "Ah_to_mAh",
            "hr_to_s",
            "A.h_to_mAh",
            "A_to_A",
            "V_to_V",
            "degC_to_K",
            "K_to_degC",
            "C_to_K",
        ],
    )
    def test_conversion(self, name, pattern, target, factor, offset):
        """Compute factor and offset for unit conversions."""
        f, o = ColumnName(name, pattern).conversion_parameters(target)
        assert f == pytest.approx(factor)
        assert o == pytest.approx(offset)

    def test_unitless_raises(self):
        """conversion_parameters on unitless column raises ValueError."""
        with pytest.raises(ValueError, match="has no unit"):
            ColumnName("Step", BRACKET).conversion_parameters("s")

    def test_incompatible_raises(self):
        """conversion_parameters with incompatible units raises ValueError."""
        with pytest.raises(ValueError, match="Cannot convert"):
            ColumnName("Current [A]", BRACKET).conversion_parameters("V")

    def test_negative_temperature_conversion(self):
        """Negative temperatures convert correctly across scales."""
        cn = ColumnName("Temperature [degC]", BRACKET)
        f, o = cn.conversion_parameters("K")
        # -40 degC = 233.15 K
        assert (-40.0 * f + o) == pytest.approx(233.15)


class TestFind:
    """ColumnName.find static method."""

    @pytest.mark.parametrize(
        ("quantity", "columns", "fmt", "expected_qty", "expected_unit"),
        [
            (
                "Current",
                ["Current [A]", "Voltage [V]"],
                "square_bracket",
                "Current",
                "A",
            ),
            (
                "Voltage",
                ["Current [A]", "Voltage [V]"],
                "square_bracket",
                "Voltage",
                "V",
            ),
            ("Current", ["Current / A"], "bdf", "Current", "A"),
            ("Step", ["Step"], "square_bracket", "Step", None),
            (
                "Test Time",
                ["Test Time (s)"],
                "parentheses",
                "Test Time",
                "s",
            ),
            ("Current", ["Current(A)"], "neware", "Current", "A"),
            ("Time", ["~Time[s]"], "basytec", "Time", "s"),
            ("I", ["I/mA"], "biologic", "I", "mA"),
        ],
        ids=[
            "bracket",
            "bracket_v",
            "bdf",
            "bare",
            "paren",
            "neware",
            "basytec",
            "biologic",
        ],
    )
    def test_find_match(self, quantity, columns, fmt, expected_qty, expected_unit):
        """Find a column by quantity name across formats."""
        result = ColumnName.find(quantity, columns, fmt)
        assert result is not None
        _, cn = result
        assert cn.quantity == expected_qty
        if expected_unit is None:
            assert cn.unit is None
        else:
            _assert_unit_converts(cn.unit, expected_unit)

    @pytest.mark.parametrize(
        ("quantity", "columns", "fmt"),
        [
            ("Temperature", ["Current [A]"], "square_bracket"),
            ("Current", [], "square_bracket"),
            ("I", ["Current [A]"], "square_bracket"),
        ],
        ids=["not_present", "empty", "alias_no_match"],
    )
    def test_find_no_match(self, quantity, columns, fmt):
        """Find returns None when quantity not present or format mismatch."""
        assert ColumnName.find(quantity, columns, fmt) is None

    def test_case_insensitive(self):
        """Find is case-insensitive for quantity matching."""
        assert ColumnName.find("current", ["Current [A]"], "square_bracket") is not None
        assert ColumnName.find("CURRENT", ["current [A]"], "square_bracket") is not None

    def test_strips_whitespace(self):
        """Find strips whitespace from search quantity."""
        assert (
            ColumnName.find("  Current  ", ["Current [A]"], "square_bracket")
            is not None
        )

    def test_returns_first_match(self):
        """Find returns the first matching column."""
        result = ColumnName.find(
            "Current", ["Current [A]", "Current [mA]"], "square_bracket"
        )
        assert result is not None
        assert result[0] == "Current [A]"

    def test_skips_unparseable(self):
        """Find skips unparseable columns and continues searching."""
        result = ColumnName.find("Current", ["Step [", "Current [A]"], "square_bracket")
        assert result is not None and result[1].quantity == "Current"


class TestResolve:
    """ColumnName.resolve — all resolution steps."""

    @pytest.mark.parametrize(
        ("target", "pat", "src_cols", "src_fmt", "vals_in", "vals_out", "bdf"),
        [
            # Step 1: exact match
            ("Current / A", BDF, ["Current / A"], "bdf", [1.0, 2.0], [1.0, 2.0], None),
            ("Step", BRACKET, ["Step"], "square_bracket", [1, 2, 3], [1, 2, 3], None),
            # Step 2: cross-format, same unit
            (
                "Current / A",
                BDF,
                ["Current [A]"],
                "square_bracket",
                [1.0, 2.0],
                [1.0, 2.0],
                None,
            ),
            (
                "Current [A]",
                BRACKET,
                ["Current / A"],
                "bdf",
                [1.0, 2.0],
                [1.0, 2.0],
                None,
            ),
            # Step 2: unit conversion
            (
                "Current / mA",
                BDF,
                ["Current [A]"],
                "square_bracket",
                [1.0, 2.0],
                [1000.0, 2000.0],
                None,
            ),
            (
                "Capacity / Ah",
                BDF,
                ["Capacity [mAh]"],
                "square_bracket",
                [1000.0, 2500.0],
                [1.0, 2.5],
                None,
            ),
            (
                "Time / s",
                BDF,
                ["Time (min)"],
                "parentheses",
                [1.0, 2.0],
                [60.0, 120.0],
                None,
            ),
            # Step 2: temperature (affine offset)
            (
                "Temperature / K",
                BDF,
                ["Temperature [degC]"],
                "square_bracket",
                [0.0, 25.0, 100.0],
                [273.15, 298.15, 373.15],
                None,
            ),
            (
                "Temperature [degC]",
                BRACKET,
                ["Temperature / K"],
                "bdf",
                [273.15, 298.15],
                [0.0, 25.0],
                None,
            ),
            (
                "Temperature / K",
                BDF,
                ["Temperature [degC]"],
                "square_bracket",
                [-40.0, -273.15],
                [233.15, 0.0],
                None,
            ),
            # Step 2: temperature format conversion (C notation to K)
            (
                "Temperature / K",
                BDF,
                ["Temperature [C]"],
                "square_bracket",
                [0.0, 100.0],
                [273.15, 373.15],
                None,
            ),
            # Step 3: BDF alias with unit conversion
            (
                "Current / A",
                BDF,
                ["I/mA"],
                "biologic",
                [1000.0, 2000.0],
                [1.0, 2.0],
                ALL_COLUMNS,
            ),
        ],
        ids=[
            "exact",
            "exact_unitless",
            "cross_bdf_bracket",
            "cross_bracket_bdf",
            "A_to_mA",
            "mAh_to_Ah",
            "min_to_s",
            "degC_to_K",
            "K_to_degC",
            "negative_temps",
            "temperature_C_format",
            "bdf_alias_I",
        ],
    )
    def test_resolve(self, target, pat, src_cols, src_fmt, vals_in, vals_out, bdf):
        """Resolve column against available columns with unit conversion."""
        expr = ColumnName(target, pat).resolve(src_cols, src_fmt, bdf_columns=bdf)
        output = pl.DataFrame({src_cols[0]: vals_in}).select(expr)
        assert output.columns == [target]
        assert output[target].to_list() == pytest.approx(vals_out)

    @pytest.mark.parametrize(
        ("target", "pat", "src_cols", "src_fmt"),
        [
            ("Current [A]", BRACKET, ["Current / A"], "bdf"),
            ("Voltage / V", BDF, ["Voltage (V)"], "parentheses"),
            ("Time(s)", NEWARE, ["Time (s)"], "parentheses"),
        ],
        ids=["bracket_alias", "bdf_alias", "neware_alias"],
    )
    def test_output_alias(self, target, pat, src_cols, src_fmt):
        """Resolve aliases result to target column name string."""
        expr = ColumnName(target, pat).resolve(src_cols, src_fmt)
        assert pl.DataFrame({src_cols[0]: [1.0]}).select(expr).columns == [target]

    def test_bdf_recipe(self):
        """Resolve step 4: BDF recipe derives column from dependencies."""
        expr = ColumnName("Event", BDF).resolve(
            ["Step"], "bdf", bdf_columns=ALL_COLUMNS
        )
        assert pl.DataFrame({"Step": [1, 1, 2, 2, 3]}).select(expr)[
            "Event"
        ].to_list() == [0, 0, 1, 1, 2]

    def test_no_match_raises(self):
        """Resolve raises ValueError when no column matches."""
        with pytest.raises(ValueError, match="No column matching"):
            ColumnName("Temperature / degC", BDF).resolve(
                ["Current [A]"], "square_bracket"
            )

    def test_alias_without_bdf_columns_raises(self):
        """Resolve raises when alias needed but bdf_columns not provided."""
        with pytest.raises(ValueError, match="No column matching"):
            ColumnName("Current / A", BDF).resolve(["I/mA"], "biologic")

    def test_incompatible_dimensions_raises(self):
        """Resolve raises ValueError for dimensionally incompatible units."""
        with pytest.raises(ValueError, match="Cannot convert"):
            ColumnName("Voltage / mA", BDF).resolve(["Voltage [V]"], "square_bracket")

    def test_skips_unparseable(self):
        """Resolve skips unparseable columns and finds valid match."""
        expr = ColumnName("Current / A", BDF).resolve(
            ["Step [", "Current [A]"], "square_bracket"
        )
        assert pl.DataFrame({"Current [A]": [1.0]}).select(expr)[
            "Current / A"
        ].to_list() == [1.0]

    def test_complex_quantity_names(self):
        """Resolve works with complex quantity names containing special chars."""
        expr = ColumnName("~Charge.Rate / C", BDF).resolve(
            ["~Charge.Rate[C]"], "square_bracket"
        )
        output = pl.DataFrame({"~Charge.Rate[C]": [1.0, 2.0, 3.0]}).select(expr)
        assert output.columns == ["~Charge.Rate / C"]
        assert output["~Charge.Rate / C"].to_list() == [1.0, 2.0, 3.0]
