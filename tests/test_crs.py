"""Tests for digitalearth._crs.source_epsg — best-effort EPSG resolution (PA-8)."""
from digitalearth._crs import source_epsg


class _FakeCRS:
    """Stand-in for a pyproj CRS exposing only to_epsg()."""

    def __init__(self, code):
        self._code = code

    def to_epsg(self):
        """Return the configured EPSG code (or None)."""
        return self._code


class _FakeFeatures:
    """Stand-in for a FeatureCollection/GeoDataFrame with .epsg and .crs attributes."""

    def __init__(self, epsg=None, crs=None):
        self.epsg = epsg
        self.crs = crs


class TestSourceEpsg:
    """Tests for source_epsg."""

    def test_prefers_epsg_attribute(self):
        """source_epsg returns the .epsg attribute when present.

        Test scenario:
            With .epsg set, .crs is never consulted (even if it would give a different code).
        """
        feats = _FakeFeatures(epsg=32618, crs=_FakeCRS(4326))
        assert source_epsg(feats) == 32618, "should return the .epsg attribute first"

    def test_falls_back_to_crs_to_epsg(self):
        """source_epsg derives the code from .crs when .epsg is None.

        Test scenario:
            With .epsg None but .crs.to_epsg() yielding a code, that code is returned.
        """
        feats = _FakeFeatures(epsg=None, crs=_FakeCRS(4326))
        assert source_epsg(feats) == 4326, "should fall back to crs.to_epsg()"

    def test_returns_default_when_unresolved(self):
        """source_epsg returns the default when neither .epsg nor .crs gives a code.

        Test scenario:
            .epsg None and .crs.to_epsg() None -> the (default None) is returned.
        """
        feats = _FakeFeatures(epsg=None, crs=_FakeCRS(None))
        assert source_epsg(feats) is None, "unresolved CRS should give the default (None)"

    def test_custom_default(self):
        """source_epsg returns a custom default when nothing resolves.

        Test scenario:
            With no .epsg and no .crs at all, the explicit default (4326) is returned.
        """
        feats = _FakeFeatures(epsg=None, crs=None)
        assert source_epsg(feats, 4326) == 4326, "should return the supplied default"
