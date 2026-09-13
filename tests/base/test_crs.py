"""Tests for the engine-neutral CRS layer: the readers, and the off-limb signal ``reproject`` raises.

The readers are ``source_epsg``, ``declared_crs``, ``is_geographic`` and ``authority_code``. ``reproject`` is
tested here on **real** off-limb geometry rather than a patched warp, because the vector half of the contract
is precisely that nothing raises on its own: geopandas warps an unplaceable geometry to ``inf`` and reports
success, so only real data proves the guard reads it.
"""

import geopandas as gpd
import numpy as np
import pytest
from pyramids.dataset import Dataset, GeoReference
from pyramids.feature import FeatureCollection
from shapely.geometry import Point

from digitalearth.base.crs import (
    OffLimbError,
    authority_code,
    declared_crs,
    is_geographic,
    reproject,
    source_epsg,
)

#: An orthographic projection centred on the Gulf of Guinea — everything past a quarter turn from (0, 0) is
#: behind its limb, which is what makes "visible" and "hidden" fixtures a matter of longitude alone.
ATLANTIC = "+proj=ortho +lat_0=0 +lon_0=0"


def _points(*lonlat):
    """A lon/lat ``FeatureCollection`` of the given points.

    Args:
        *lonlat: One ``(lon, lat)`` pair per point.

    Returns:
        pyramids.feature.FeatureCollection: the points, in EPSG:4326.
    """
    geometry = [Point(lon, lat) for lon, lat in lonlat]
    return FeatureCollection(gpd.GeoDataFrame(geometry=geometry, crs=4326))


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


class TestReprojectOffLimb:
    """Tests for reproject — the one signal both data families reach when nothing lands in view."""

    def test_a_vector_layer_behind_the_limb_is_reported_off_limb(self):
        """Real off-limb geometry raises, without anything monkeypatched.

        Test scenario:
            The raster half of the contract is GDAL raising and the wording being translated. The vector
            half has no exception to translate: geopandas warps every unplaceable point to ``(inf, inf)``
            and calls that a success, so a guard reading only exceptions could never fire for a vector
            layer — and the decorator every vector builder carries would have meant nothing.
        """
        with pytest.raises(OffLimbError, match="finite coordinate"):
            reproject(_points((175.0, 5.0), (170.0, 10.0)), ATLANTIC)

    def test_a_visible_vector_layer_comes_back_warped(self):
        """Geometry the projection can place is returned, with a finite extent.

        Test scenario:
            The guard must not cost the ordinary case anything: the same call on the near side returns the
            warped layer rather than raising.
        """
        warped = reproject(_points((0.0, 0.0), (5.0, 5.0)), ATLANTIC)
        assert np.isfinite(np.asarray(warped.total_bounds, dtype="float64")).all(), (
            f"a visible layer must warp to a finite extent, got {warped.total_bounds}"
        )

    def test_a_partly_hidden_vector_layer_is_not_off_limb(self):
        """A layer with even one placeable geometry is returned, not skipped.

        Test scenario:
            "Off-limb" means *nothing* landed, exactly as it does for a raster — a warp that hides half the
            points still has something to draw, and reporting it as empty would drop the visible half.
        """
        warped = reproject(_points((0.0, 0.0), (175.0, 5.0)), ATLANTIC)
        assert np.isfinite(np.asarray(warped.total_bounds, dtype="float64")).any(), (
            f"the visible point must survive, got {warped.total_bounds}"
        )

    def test_an_empty_layer_is_not_blamed_on_the_projection(self):
        """An empty collection warps to an empty one instead of being called off-limb.

        Test scenario:
            An empty layer has a non-finite extent too (``total_bounds`` is all-NaN), so a guard reading the
            output alone would accuse the projection of hiding data that was never there. The input's own
            extent is what tells the two apart.
        """
        warped = reproject(_points(), ATLANTIC)
        assert len(warped) == 0, f"an empty layer must stay empty, got {len(warped)}"

    def test_a_raster_off_the_limb_still_reports_gdal_s_case(self):
        """The raster half is untouched: GDAL's refusal is still translated, in its own wording.

        Test scenario:
            The two readings share one exception type, so the raster message has to stay distinguishable —
            it is what tells a maintainer which half of the guard fired.
        """
        raster = Dataset.from_array(
            np.ones((20, 20), "float32"),
            geo_ref=GeoReference(geo=(4.0, 0.02, 0.0, 53.0, 0.0, -0.02), epsg=4326),
        )
        with pytest.raises(OffLimbError, match="too few sample points"):
            reproject(raster, "+proj=ortho +lat_0=15 +lon_0=-175")


class TestAuthorityCode:
    """Tests for authority_code — the EPSG code read out of a definition, by pyramids."""

    def test_a_definition_is_read_down_to_its_authority_block(self):
        """WKT carrying an authority block answers with that code.

        Test scenario:
            ``declared_crs`` hands a warped dataset's WKT on to ``Source``, so the code-or-None question is
            regularly asked of a full definition rather than an ``"EPSG:<code>"`` string.
        """
        from pyramids.base.crs import crs_from_user_input

        assert authority_code(crs_from_user_input(3857).to_wkt()) == 3857, (
            "Web Mercator as WKT still names EPSG:3857"
        )

    @pytest.mark.parametrize(
        "crs", ["+proj=ortho +lat_0=53 +lon_0=4", "not-a-crs", "", None]
    )
    def test_anything_without_a_code_is_none(self, crs):
        """A code-less projection, and anything unreadable, answer ``None``.

        Args:
            crs: The CRS under test.

        Test scenario:
            ``None`` has to keep meaning "no code to name" — the reader may not guess one for a projection
            that genuinely has none, nor raise on input it cannot parse.
        """
        assert authority_code(crs) is None, f"{crs!r} names no authority code"
