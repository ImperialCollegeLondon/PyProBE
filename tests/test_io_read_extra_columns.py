"""Regression tests for the dtype of a named extra column on load.

:meth:`pyprobe.filters.Procedure.load` renames a source column that the
BDF ontology does not define, under the alias *extra_columns* gives it. A
raw reader leaves such a column as text, and the rename must not turn a
text value into a null.
"""

from pathlib import Path

from pyprobe.filters import Procedure


def _write_bdf_csv(path: Path, header: str, rows: list[str]) -> Path:
    """Write a CSV file whose header holds BDF column names.

    Args:
        path: The file to write.
        header: The header line, without a line break.
        rows: The data lines, each without a line break.

    Returns:
        Path: The file that was written.
    """
    path.write_text("\n".join([header, *rows]) + "\n")
    return path


def test_non_numeric_extra_column_survives_with_its_values_intact(
    tmp_path: Path,
) -> None:
    """A text extra column keeps its values, rather than becoming null."""
    source = _write_bdf_csv(
        tmp_path / "extra.csv",
        "Test Time / s,Current / A,Voltage / V,Note",
        ["0.0,1.0,3.7,cell replaced", "1.0,-1.0,3.6,ok"],
    )

    procedure = Procedure.load(  # type: ignore[call-arg]
        source,
        extra_columns={"Note": "Comment"},
    )

    data = procedure.data
    assert "Comment" in data.columns
    assert data["Comment"].to_list() == ["cell replaced", "ok"]
