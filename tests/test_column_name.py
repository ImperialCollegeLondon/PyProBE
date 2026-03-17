"""Tests for the ColumnName class and _parse_column helper."""

import pytest

from pyprobe.column_name import ColumnName, _ureg


class TestExtractQuantityAndUnit:
    """Tests for ColumnName._extract_quantity_and_unit static method."""

    def test_bracket_format_with_unit(self) -> None:
        """'Current [A]' returns ('Current', 'A')."""
        assert ColumnName._extract_quantity_and_unit(
            "Current [A]", ColumnName.BRACKET_FORMAT
        ) == ("Current", "A")

    def test_slash_format_with_unit(self) -> None:
        """'Current / A' returns ('Current', 'A')."""
        assert ColumnName._extract_quantity_and_unit(
            "Current / A", ColumnName.SLASH_FORMAT
        ) == ("Current", "A")

    def test_bare_name_bracket_format(self) -> None:
        """Bare 'Step' with BRACKET_FORMAT returns ('Step', None)."""
        assert ColumnName._extract_quantity_and_unit(
            "Step", ColumnName.BRACKET_FORMAT
        ) == ("Step", None)

    def test_bare_name_slash_format(self) -> None:
        """Bare 'Step' with SLASH_FORMAT returns ('Step', None)."""
        assert ColumnName._extract_quantity_and_unit(
            "Step", ColumnName.SLASH_FORMAT
        ) == ("Step", None)

    def test_partial_slash_raises(self) -> None:
        """'Step /' with SLASH_FORMAT raises ValueError."""
        with pytest.raises(ValueError):
            ColumnName._extract_quantity_and_unit("Step /", ColumnName.SLASH_FORMAT)

    def test_partial_bracket_raises(self) -> None:
        """'Step [' with BRACKET_FORMAT raises ValueError."""
        with pytest.raises(ValueError):
            ColumnName._extract_quantity_and_unit("Step [", ColumnName.BRACKET_FORMAT)

    def test_strips_whitespace_from_quantity(self) -> None:
        """Surrounding whitespace is stripped from the quantity."""
        quantity, _ = ColumnName._extract_quantity_and_unit(
            "Current [A]", ColumnName.BRACKET_FORMAT
        )
        assert quantity == "Current"


class TestColumnNameParsing:
    """Tests for ColumnName construction and properties."""

    def test_bracket_format_parses_quantity(self) -> None:
        """Parsing 'Current [A]' with BRACKET_FORMAT yields quantity 'Current'."""
        cn = ColumnName("Current [A]", ColumnName.BRACKET_FORMAT)
        assert cn.quantity == "Current"

    def test_bracket_format_parses_unit_as_ampere(self) -> None:
        """Parsing 'Current [A]' with BRACKET_FORMAT yields ampere unit."""
        cn = ColumnName("Current [A]", ColumnName.BRACKET_FORMAT)
        assert cn.unit is not None
        assert cn.unit == _ureg.parse_units("A")

    def test_slash_format_parses_quantity(self) -> None:
        """Parsing 'Current / A' with SLASH_FORMAT yields quantity 'Current'."""
        cn = ColumnName("Current / A", ColumnName.SLASH_FORMAT)
        assert cn.quantity == "Current"

    def test_slash_format_parses_unit_as_ampere(self) -> None:
        """Parsing 'Current / A' with SLASH_FORMAT yields ampere unit."""
        cn = ColumnName("Current / A", ColumnName.SLASH_FORMAT)
        assert cn.unit is not None
        assert cn.unit == _ureg.parse_units("A")

    def test_bare_name_bracket_format_unit_is_none(self) -> None:
        """Bare 'Step' with BRACKET_FORMAT yields unit=None."""
        cn = ColumnName("Step", ColumnName.BRACKET_FORMAT)
        assert cn.unit is None
        assert cn.quantity == "Step"

    def test_bare_name_slash_format_unit_is_none(self) -> None:
        """Bare 'Step' with SLASH_FORMAT yields unit=None."""
        cn = ColumnName("Step", ColumnName.SLASH_FORMAT)
        assert cn.unit is None
        assert cn.quantity == "Step"

    def test_partial_slash_raises(self) -> None:
        """'Step /' with SLASH_FORMAT raises ValueError."""
        with pytest.raises(ValueError):
            ColumnName("Step /", ColumnName.SLASH_FORMAT)

    def test_partial_bracket_raises(self) -> None:
        """'Step [' with BRACKET_FORMAT raises ValueError."""
        with pytest.raises(ValueError):
            ColumnName("Step [", ColumnName.BRACKET_FORMAT)

    def test_str_returns_original_name(self) -> None:
        """__str__ returns the original column name string."""
        name = "Current [A]"
        cn = ColumnName(name, ColumnName.BRACKET_FORMAT)
        assert str(cn) == name

    def test_alias_ohms_resolves_to_ohm(self) -> None:
        """'Resistance [Ohms]' with BRACKET_FORMAT resolves unit via alias map."""
        cn = ColumnName("Resistance [Ohms]", ColumnName.BRACKET_FORMAT)
        assert cn.unit is not None
        assert cn.unit == _ureg.parse_units("ohm")

    def test_alias_seconds_resolves(self) -> None:
        """'Time [Seconds]' with BRACKET_FORMAT resolves unit via alias map."""
        cn = ColumnName("Time [Seconds]", ColumnName.BRACKET_FORMAT)
        assert cn.unit is not None
        assert cn.unit == _ureg.parse_units("s")

    def test_invalid_unit_raises_value_error(self) -> None:
        """A column with an unparseable unit raises ValueError."""
        with pytest.raises(ValueError, match="could not be parsed"):
            ColumnName("Foo [not_a_unit_xyz]", ColumnName.BRACKET_FORMAT)


class TestConversionFactor:
    """Tests for ColumnName.conversion_factor."""

    def test_ampere_to_milliampere(self) -> None:
        """'Current [A]' → 'mA' conversion factor is 1000.0."""
        cn = ColumnName("Current [A]", ColumnName.BRACKET_FORMAT)
        assert cn.conversion_factor("mA") == pytest.approx(1000.0)

    def test_capacity_ah_to_mah(self) -> None:
        """'Capacity [Ah]' → 'mAh' conversion factor is 1000.0."""
        cn = ColumnName("Capacity [Ah]", ColumnName.BRACKET_FORMAT)
        assert cn.conversion_factor("mAh") == pytest.approx(1000.0)

    def test_capacity_compound_unit_to_ah(self) -> None:
        """'Capacity [A.h]' → 'Ah' conversion factor is 1.0."""
        cn = ColumnName("Capacity [A.h]", ColumnName.BRACKET_FORMAT)
        assert cn.conversion_factor("Ah") == pytest.approx(1.0)

    def test_incompatible_units_raises_value_error(self) -> None:
        """Converting ampere to volt raises ValueError."""
        cn = ColumnName("Current [A]", ColumnName.BRACKET_FORMAT)
        with pytest.raises(ValueError, match="Cannot convert"):
            cn.conversion_factor("V")


class TestWithUnit:
    """Tests for ColumnName.with_unit."""

    def test_bracket_format_with_unit(self) -> None:
        """with_unit on bracket-format column produces 'Current [mA]'."""
        cn = ColumnName("Current [A]", ColumnName.BRACKET_FORMAT)
        result = cn.with_unit("mA")
        assert str(result) == "Current [mA]"

    def test_slash_format_with_unit(self) -> None:
        """with_unit on slash-format column produces 'Current / mA'."""
        cn = ColumnName("Current / A", ColumnName.SLASH_FORMAT)
        result = cn.with_unit("mA")
        assert str(result) == "Current / mA"

    def test_with_unit_preserves_quantity(self) -> None:
        """with_unit preserves the quantity name."""
        cn = ColumnName("Current [A]", ColumnName.BRACKET_FORMAT)
        result = cn.with_unit("mA")
        assert result.quantity == "Current"


class TestFindInColumns:
    """Tests for ColumnName.find_in_columns."""

    def test_finds_matching_column_bracket_format(self) -> None:
        """find_in_columns returns 'Current [A]' when searching for 'Current'."""
        cols = ["Time [s]", "Current [A]", "Voltage [V]"]
        result = ColumnName.find_in_columns("Current", cols, ColumnName.BRACKET_FORMAT)
        assert result is not None
        assert str(result) == "Current [A]"

    def test_returns_none_when_quantity_absent(self) -> None:
        """find_in_columns returns None when quantity is not present."""
        cols = ["Time [s]", "Voltage [V]"]
        result = ColumnName.find_in_columns("Current", cols, ColumnName.BRACKET_FORMAT)
        assert result is None

    def test_skips_columns_that_do_not_match_pattern(self) -> None:
        """find_in_columns skips columns that don't match the pattern gracefully."""
        cols = ["Step", "Current [A]"]
        result = ColumnName.find_in_columns("Current", cols, ColumnName.BRACKET_FORMAT)
        assert result is not None
        assert str(result) == "Current [A]"
