"""Tests for the engine-neutral CRS readers: source_epsg, declared_crs and is_geographic."""

import numpy as np
import pytest
from pyramids.dataset import Dataset, GeoReference

from digitalearth.base.crs import declared_crs, is_geographic, source_epsg


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
        assert source_epsg(feats) is None, (
            "unresolved CRS should give the default (None)"
        )

    def test_custom_default(self):
        """source_epsg returns a custom default when nothing resolves.

        Test scenario:
            With no .epsg and no .crs at all, the explicit default (4326) is returned.
        """
        feats = _FakeFeatures(epsg=None, crs=None)
        assert source_epsg(feats, 4326) == 4326, "should return the supplied default"


#: Every spelling ``Source.crs`` is documented to accept, paired with the answer it must give.
CRS_SPELLINGS = [
    (4326, True),
    (3857, False),
    ("EPSG:4326", True),
    ("EPSG:3857", False),
    ("epsg:4326", True),
    ("epsg:3857", False),
    ("+proj=longlat +datum=WGS84 +no_defs", True),
    ("+proj=ortho +lat_0=53 +lon_0=4", False),
]


class TestIsGeographic:
    """Tests for is_geographic — the geographic/projected question, read by pyramids."""

    @pytest.mark.parametrize("crs, geographic", CRS_SPELLINGS)
    def test_every_documented_spelling_is_read(self, crs, geographic):
        """Each form the ``Source.crs`` contract blesses answers the same way.

        Test scenario:
            The contract allows an EPSG int, an ``"EPSG:<code>"`` string (either case) and a proj4/WKT
            definition. A reader that understood only the int spelling answered "unknown" for the rest, which
            let a projected source spelled ``"EPSG:3857"`` through globe()'s geographic gate.
        """
        assert is_geographic(crs) is geographic, (
            f"{crs!r} should read as geographic={geographic}, got {is_geographic(crs)!r}"
        )

    def test_wkt_is_read_too(self):
        """A CRS given as WKT — what a warped pyramids dataset reports — is read like any other spelling.

        Test scenario:
            ``declared_crs`` falls back to the projection definition, which pyramids hands over as WKT, so
            that spelling has to reach the same answer as the code it was built from.
        """
        from pyramids.base.crs import crs_from_user_input

        assert is_geographic(crs_from_user_input(4326).to_wkt()) is True, (
            "WGS84 as WKT is geographic"
        )
        assert is_geographic(crs_from_user_input(32618).to_wkt()) is False, (
            "UTM 18N as WKT is projected"
        )

    @pytest.mark.parametrize("crs", [None, "not-a-crs", "", True, False, object()])
    def test_an_unreadable_crs_is_unknown_not_projected(self, crs):
        """Anything pyramids cannot interpret is ``None`` — never a confident "projected".

        Test scenario:
            Callers branch on the three answers; reporting an unreadable CRS as ``False`` would refuse data
            over a CRS nobody ever read. Booleans are ``int`` subclasses in Python, so they have to be
            rejected rather than resolved as the codes 0/1.
        """
        assert is_geographic(crs) is None, (
            f"{crs!r} is not a readable CRS, so the answer must be None"
        )


class TestDeclaredCrs:
    """Tests for declared_crs — an input's own CRS: its code, else its definition."""

    def test_a_coded_raster_reports_its_code(self):
        """A raster with an authority code answers with the code itself.

        Test scenario:
            The common case: pyramids resolves the code, and the code is the most useful spelling.
        """
        dataset = Dataset.from_array(
            np.ones((4, 4), "float32"),
            geo_ref=GeoReference(geo=(0.0, 1.0, 0.0, 0.0, 0.0, -1.0), epsg=4326),
        )
        assert declared_crs(dataset) == 4326, "an EPSG raster reports its code"

    def test_a_code_less_projection_reports_its_definition(self):
        """With no authority code the projection definition is reported, not ``None``.

        Test scenario:
            A warp into an orthographic CRS leaves ``epsg is None``; answering ``None`` there would claim the
            CRS is unknown when the coordinates plainly have an address — and is_geographic could then never
            tell the globe that such a source is projected.
        """
        warped = Dataset.from_array(
            np.ones((20, 20), "float32"),
            geo_ref=GeoReference(geo=(4.0, 0.02, 0.0, 53.0, 0.0, -0.02), epsg=4326),
        ).to_crs("+proj=ortho +lat_0=53 +lon_0=4")
        assert warped.epsg is None, "the fixture must be the code-less case"
        definition = declared_crs(warped)
        assert definition, "a code-less projection must still report its definition"
        assert is_geographic(definition) is False, (
            f"the reported definition must read as projected, got {definition!r}"
        )

    def test_an_input_declaring_no_crs_is_unknown(self):
        """A raw array declares no CRS at all — the one case that means "unknown".

        Test scenario:
            ``None`` is reserved for genuinely unknown, which is what keeps the globe's coordinate-range
            fallback reachable for raw arrays.
        """
        assert declared_crs(np.zeros((2, 2))) is None, (
            "a bare numpy array declares no CRS"
        )
