"""Tests for the metadata helper functions."""

import bdf
import pytest

from tests.metadata_helpers import build_metadata, read_extras


class TestMetadataHelpers:
    """Tests for the metadata helper functions."""

    def test_build_metadata_empty(self) -> None:
        """build_metadata with no arguments returns an empty record."""
        metadata = build_metadata()
        assert metadata == bdf.Metadata()

    def test_build_metadata_with_keys(self) -> None:
        """build_metadata creates a record whose extras hold the keyword arguments."""
        metadata = build_metadata(Name="Test_Cell", test="value")
        assert metadata == bdf.Metadata(extras={"Name": "Test_Cell", "test": "value"})

    def test_read_extras_from_mapping(self) -> None:
        """read_extras returns a Mapping when metadata is a Mapping."""

        class ObjWithDictMetadata:
            metadata = {"key": "value"}

        obj = ObjWithDictMetadata()
        assert read_extras(obj) == {"key": "value"}

    def test_read_extras_from_bdf_metadata(self) -> None:
        """read_extras returns extras from bdf.Metadata-like object."""

        class MockBDFMetadata:
            extras = {"key": "value"}

        class ObjWithBDFMetadata:
            metadata = MockBDFMetadata()

        obj = ObjWithBDFMetadata()
        assert read_extras(obj) == {"key": "value"}

    def test_read_extras_with_none_extras(self) -> None:
        """read_extras returns empty dict when extras is None."""

        class MockBDFMetadata:
            extras = None

        class ObjWithBDFMetadata:
            metadata = MockBDFMetadata()

        obj = ObjWithBDFMetadata()
        assert read_extras(obj) == {}

    def test_read_extras_invalid_raises(self) -> None:
        """read_extras raises TypeError for invalid metadata."""

        class ObjWithInvalidMetadata:
            metadata = "invalid"

        obj = ObjWithInvalidMetadata()
        with pytest.raises(TypeError):
            read_extras(obj)
