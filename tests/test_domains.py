"""Tests for T5.1 — named domains and Map.set_domain extent setting."""

import pytest

from digitalearth.scene import Map
from digitalearth.scene.domains import DOMAINS, resolve_domain


class TestResolveDomain:
    """Tests for resolve_domain."""

    def test_named_region(self):
        """A registered name resolves to its EPSG:4326 bbox."""
        assert resolve_domain("Europe") == DOMAINS["europe"]

    def test_explicit_bbox(self):
        """An explicit 4-tuple passes through as floats."""
        assert resolve_domain([0, 40, 20, 60]) == (0.0, 40.0, 20.0, 60.0)

    def test_none_returns_none(self):
        """None resolves to None (no domain)."""
        assert resolve_domain(None) is None

    def test_unknown_name_raises(self):
        """An unknown region name raises KeyError."""
        with pytest.raises(KeyError, match="unknown domain"):
            resolve_domain("atlantis")

    def test_bad_bbox_length_raises(self):
        """A bbox that is not length-4 raises ValueError."""
        with pytest.raises(ValueError, match="west, south, east, north"):
            resolve_domain([0, 1, 2])


class TestSetDomain:
    """Tests for Map.set_domain."""

    def test_set_domain_in_geographic_crs(self):
        """In EPSG:4326 the axes limits equal the domain bbox directly."""
        m = Map(crs=4326)
        m.set_domain("europe")
        xlim, ylim = m.ax.get_xlim(), m.ax.get_ylim()
        assert xlim == (-25.0, 45.0)
        assert ylim == (34.0, 72.0)

    def test_set_domain_reprojects_to_display_crs(self):
        """In EPSG:3857 the domain bbox is reprojected to metres (much larger magnitudes)."""
        m = Map(crs=3857)
        m.set_domain("europe")
        assert m.ax.get_xlim()[1] > 1e6

    def test_set_domain_from_constructor(self):
        """set_domain() with no argument falls back to the domain given at construction."""
        m = Map(crs=4326, domain="africa")
        m.set_domain()
        assert m.ax.get_xlim() == (-20.0, 55.0)

    def test_set_domain_noop_without_domain(self):
        """set_domain() is a no-op when no domain is configured."""
        m = Map(crs=4326)
        before = (m.ax.get_xlim(), m.ax.get_ylim())
        m.set_domain()
        assert (m.ax.get_xlim(), m.ax.get_ylim()) == before
