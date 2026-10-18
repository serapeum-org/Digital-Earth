"""Tests for T5.1 — named domains and Map.set_domain extent setting.

The table itself moved to `digitalearth.base.domains`, so every tier resolves a name to the same box; these
read it through the static tier, which is what `Map.set_domain` uses. `digitalearth.static.core.domains` stays as
a re-export and is covered by `tests/base/test_domains_shim.py`.
"""

import pytest

from digitalearth.base.domains import DOMAINS, resolve_domain, union_domains
from digitalearth.static import Map


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

    def test_alias_resolves_to_canonical(self):
        """A shorthand alias resolves to the same bbox as its canonical name."""
        assert resolve_domain("UK") == resolve_domain("united-kingdom")
        assert resolve_domain("usa") == DOMAINS["united-states"]

    def test_country_domains_present(self):
        """The catalogue now carries country-level regions, not just continents."""
        for name in ("germany", "japan", "brazil", "australia"):
            west, south, east, north = resolve_domain(name)
            assert west < east, f"{name} has non-increasing longitude bounds"
            assert south < north, f"{name} has non-increasing latitude bounds"

    def test_union_via_resolve_domain(self):
        """A sequence of names resolves to the single enclosing bbox."""
        assert resolve_domain(["iberia", "france"]) == (-10.0, 36.0, 9.6, 51.1)

    def test_union_encloses_every_part(self):
        """union_domains covers each member region on all four sides."""
        box = union_domains("germany", "italy", "spain")
        for name in ("germany", "italy", "spain"):
            w, s, e, n = DOMAINS[name]
            assert box[0] <= w, f"union west {box[0]} does not cover {name} west {w}"
            assert box[1] <= s, f"union south {box[1]} does not cover {name} south {s}"
            assert box[2] >= e, f"union east {box[2]} does not cover {name} east {e}"
            assert box[3] >= n, f"union north {box[3]} does not cover {name} north {n}"

    def test_union_of_one_returns_itself(self):
        """A single-member union is just that region."""
        assert union_domains("japan") == DOMAINS["japan"]

    def test_empty_union_raises(self):
        """A union with no members is an error, not an empty box."""
        with pytest.raises(ValueError, match="at least one domain"):
            union_domains()

    def test_bool_in_bbox_gives_the_bbox_message(self):
        """A bool in a 4-number bbox is refused with the bbox message, not a cryptic TypeError.

        Test scenario:
            _is_number excludes bool (so True is not read as 1); the sequence is then not all-numbers and
            used to route into union_domains, where list(True) raised TypeError. It must raise the clear
            (west, south, east, north) message instead.
        """
        with pytest.raises(ValueError, match="west, south, east, north"):
            resolve_domain([True, 40, 20, 60])

    def test_decimal_bbox_is_accepted(self):
        """A Decimal bbox resolves, as the pre-PR float() path accepted (no silent narrowing).

        Test scenario:
            Decimal is numbers.Number but not numbers.Real, so a Real-only gate would reject a bbox the old
            tuple(float(v) ...) path accepted. _is_number must admit it (while still excluding bool/complex).
        """
        from decimal import Decimal

        box = resolve_domain(
            [Decimal("0"), Decimal("40"), Decimal("20"), Decimal("60")]
        )
        assert box == (0.0, 40.0, 20.0, 60.0), f"Decimal bbox should resolve, got {box}"

    def test_numpy_scalar_bbox_is_accepted(self):
        """A bbox of numpy scalars resolves like a plain-float one (Windows np.int64 is not int)."""
        import numpy as np

        box = resolve_domain([np.int64(0), np.int64(40), np.int64(20), np.int64(60)])
        assert box == (0.0, 40.0, 20.0, 60.0), f"numpy bbox should resolve, got {box}"


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
