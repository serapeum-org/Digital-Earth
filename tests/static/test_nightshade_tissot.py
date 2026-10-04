"""Night shading and Tissot indicatrices on a static map (ST-27).

cleopatra 0.38 ships both artists (``cleopatra.basemap.solar``): ``add_nightshade`` shades the night side of the
day/night terminator, and ``add_tissot`` draws rings a caller has already projected. Neither resolves a CRS —
that is the consumer's job, and here pyramids does it — so these tests check the two things the static tier
adds: the geometry lands where the display CRS puts it, and each overlay is a described, addressable layer.

Expected positions are computed from first principles in the test (spherical Mercator, the mean Earth
radius), never through the code under test.
"""

import math
from datetime import datetime, timezone

import numpy as np
import pytest
from cleopatra.basemap.solar import subsolar_point
from matplotlib.collections import PolyCollection
from pyramids.base.georeference import GeoReference
from pyramids.dataset import Dataset

from digitalearth.static import Map, projections
from digitalearth.static.maps import decoration

#: The equinox at noon UTC: the sun stands over (about) lon 0, so lon 180 is at midnight.
EQUINOX_NOON = datetime(2026, 3, 20, 12, 0, tzinfo=timezone.utc)

#: The June solstice at noon UTC: the south pole is in polar night.
JUNE_NOON = datetime(2026, 6, 21, 12, 0, tzinfo=timezone.utc)

#: The December solstice at noon UTC: the *north* pole is in polar night.
DECEMBER_NOON = datetime(2026, 12, 21, 12, 0, tzinfo=timezone.utc)

#: Web Mercator's sphere radius, in metres (EPSG:3857).
MERCATOR_R = 6378137.0

#: The mean Earth radius cleopatra's geodesic circles are drawn on, in metres (IUGG).
MEAN_EARTH_R = 6371008.8

#: EPSG:3031's latitude of true scale, in degrees south — its standard parallel.
POLAR_TRUE_SCALE_LAT = 71.0

#: The northern limit of EPSG:3031's declared area of use, in degrees.
POLAR_AREA_OF_USE_LAT = -60.0

#: EPSG:3832's central meridian, in degrees — PDC Mercator is centred on the Pacific.
PDC_CENTRAL_LON = 150.0

#: The longitude EPSG:3832 therefore cuts: half a turn from its central meridian, wrapped.
PDC_SEAM_LON = PDC_CENTRAL_LON - 180.0

#: Western edge, in EPSG:3031 metres, of the raster :func:`_wide_polar_raster` builds. It is twice as far
#: out as the ``+/- 3.3e6`` m the CRS declares for itself, which is the whole point of it: the declared area
#: of use is not the projection's domain, so a grid out here still projects.
_POLAR_RASTER_WEST = -7.0e6

#: Width and height, in EPSG:3031 metres, of :func:`_wide_polar_raster`'s grid.
_POLAR_RASTER_WIDTH = 1.4e7


def _mercator(lon: float, lat: float):
    """Project one lon/lat onto spherical Web Mercator, written out from the formula.

    Args:
        lon: Longitude in degrees.
        lat: Latitude in degrees.

    Returns:
        The ``(x, y)`` in metres.
    """
    x = MERCATOR_R * math.radians(lon)
    y = MERCATOR_R * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))
    return x, y


def _polar_stereographic_south(lat: float) -> float:
    """Distance from the south pole on spherical Antarctic Polar Stereographic, from the formula.

    EPSG:3031 is ellipsoidal, so this runs about 0.4% short of what PROJ answers — far inside the
    tolerance the tests that use it ask for, and independent of the code under test.

    Args:
        lat: Latitude in degrees, negative in the southern hemisphere.

    Returns:
        The radial distance from the pole, in metres.
    """
    scale = (1.0 + math.sin(math.radians(POLAR_TRUE_SCALE_LAT))) / 2.0
    return 2.0 * MEAN_EARTH_R * scale * math.tan(math.radians(45.0 + lat / 2.0))


def _covered(artist, xy) -> bool:
    """Whether any polygon of a collection contains one point, in data coordinates.

    Args:
        artist: The ``PolyCollection``.
        xy: The point.

    Returns:
        ``True`` when a path contains it.
    """
    return any(path.contains_point(xy) for path in artist.get_paths())


def _wide_polar_raster() -> Dataset:
    """Return an EPSG:3031 raster twice as wide as the extent that CRS declares for itself.

    Returns:
        An 8x8 single-band dataset spanning :data:`_POLAR_RASTER_WIDTH` metres on both axes, centred on the
        south pole.
    """
    cell = _POLAR_RASTER_WIDTH / 8.0
    north = _POLAR_RASTER_WEST + _POLAR_RASTER_WIDTH
    return Dataset.from_array(
        arr=np.arange(64, dtype="float32").reshape(8, 8),
        geo_ref=GeoReference(
            geo=(_POLAR_RASTER_WEST, cell, 0.0, north, 0.0, -cell), epsg=3031
        ),
    )


def _world_4326() -> Map:
    """Return a lon/lat map framed on the whole world.

    Returns:
        The map.
    """
    canvas = Map(crs=4326)
    canvas.ax.set_xlim(-180, 180)
    canvas.ax.set_ylim(-90, 90)
    return canvas


class TestNightshade:
    """The night side of the terminator, shaded in the display CRS."""

    def test_it_draws_a_polygon_collection_on_the_axes(self):
        """The public return is cleopatra's artist, and it is on the map's axes."""
        canvas = _world_4326()
        artist = canvas.nightshade(EQUINOX_NOON)
        assert isinstance(artist, PolyCollection), type(artist)
        assert artist in canvas.ax.collections, "the shade must be on the map's axes"

    def test_midnight_is_shaded_and_noon_is_not_on_a_lonlat_map(self):
        """At the equinox at noon UTC the antimeridian is at midnight and Greenwich at noon."""
        artist = _world_4326().nightshade(EQUINOX_NOON)
        assert _covered(artist, (170.0, 0.0)), "lon 170 at noon UTC is night"
        assert not _covered(artist, (0.0, 0.0)), "lon 0 at noon UTC is day"

    def test_the_shade_is_reprojected_into_the_display_crs(self):
        """On Web Mercator the shade sits at the Mercator position of the night side, not at lon/lat."""
        canvas = Map(crs=3857)
        canvas.ax.set_xlim(-2.0e7, 2.0e7)
        canvas.ax.set_ylim(-2.0e7, 2.0e7)
        artist = canvas.nightshade(EQUINOX_NOON)
        assert _covered(artist, _mercator(170.0, 10.0)), "lon 170 is night"
        assert not _covered(artist, _mercator(0.0, 10.0)), "lon 0 is day"

    def test_a_polar_night_survives_a_projection_that_cannot_reach_the_pole(self):
        """Web Mercator sends lat -90 to infinity, and the polar night must still be shaded.

        Test scenario:
            cleopatra drops non-finite vertices. Dropping the two pole corners of the night ring would
            close it straight along the terminator and leave the polar cap unshaded, so the static tier
            must hand cleopatra a ring Mercator can project.
        """
        canvas = Map(crs=3857)
        canvas.ax.set_xlim(-2.0e7, 2.0e7)
        canvas.ax.set_ylim(-2.0e7, 2.0e7)
        artist = canvas.nightshade(JUNE_NOON)
        assert _covered(artist, _mercator(0.0, -80.0)), (
            "the south pole is in polar night in June"
        )
        assert not _covered(artist, _mercator(0.0, 80.0)), (
            "the north pole is in polar day in June"
        )

    def test_a_polar_night_fits_the_polar_projection_it_is_drawn_in(self):
        """A bare EPSG:3031 axes is fitted to Antarctica, not to a view a thousand times its extent.

        Test scenario:
            A night region that holds a pole runs along that pole's map edge, and EPSG:3031 has no useful
            image of the *north* pole: PROJ puts lat 90 at ``y = 4.0e23`` m, and even lat 89.9 — the
            latitude the ring's vertices are pulled back to — at ``y = 1.4e10`` m. Autoscaling a bare axes
            to that vertex framed the December solstice at ``+/- 1.5e10`` m, about 4600 times the
            ``+/- 3.3e6`` m EPSG:3031 declares for itself, leaving Antarctica a sub-pixel dot. The fitted
            view must stay inside that declared extent, whose radius is written out here from the
            spherical polar stereographic formula.
        """
        canvas = Map(crs=3031)
        canvas.nightshade(DECEMBER_NOON)
        limit = 2.0 * _polar_stereographic_south(POLAR_AREA_OF_USE_LAT)
        span = max(np.ptp(canvas.ax.get_xlim()), np.ptp(canvas.ax.get_ylim()))
        assert span == pytest.approx(limit, rel=0.02), span

    def test_the_south_polar_night_still_covers_the_pole_on_a_polar_projection(self):
        """Fitting the view to the projection must not crop away the polar night it was drawn for.

        Test scenario:
            The sibling test above pins the view; this one pins that the shade is still in it. At the June
            solstice everything south of about 66.6 S is in polar night, so a point at 80 S — placed here
            from the spherical polar stereographic formula, on the lon 0 meridian, which EPSG:3031 puts on
            the positive ``y`` axis — must fall inside the shade.
        """
        canvas = Map(crs=3031)
        artist = canvas.nightshade(JUNE_NOON)
        assert _covered(artist, (0.0, _polar_stereographic_south(-80.0))), (
            "80 S is in polar night at the June solstice"
        )

    def test_an_empty_axes_is_framed_on_the_shade(self):
        """A map with nothing on it yet is fitted to the overlay, as a reference layer on a bare map is."""
        canvas = Map(crs=4326)
        canvas.nightshade(EQUINOX_NOON)
        assert canvas.ax.get_xlim() != (0.0, 1.0), canvas.ax.get_xlim()

    def test_a_framed_view_is_kept(self):
        """A shade over a regional view does not zoom the map out to the globe."""
        canvas = Map(crs=4326)
        canvas.ax.set_xlim(0, 20)
        canvas.ax.set_ylim(40, 60)
        canvas.nightshade(EQUINOX_NOON)
        assert canvas.ax.get_xlim() == (0.0, 20.0), canvas.ax.get_xlim()

    def test_style_reaches_the_artist(self):
        """Style keywords are cleopatra's, forwarded to the collection."""
        artist = _world_4326().nightshade(EQUINOX_NOON, alpha=0.6, zorder=7)
        assert artist.get_alpha() == pytest.approx(0.6), artist.get_alpha()
        assert artist.get_zorder() == 7, artist.get_zorder()

    def test_the_layer_is_described_with_its_moment(self):
        """The figure records the kind, the instant as ISO 8601 text and the refraction."""
        canvas = _world_4326()
        canvas.nightshade(EQUINOX_NOON, refraction=-6.0)
        layer = canvas.figure_spec.layers.get(canvas.layer_ids[-1])
        assert layer.kind == "nightshade", layer.kind
        assert layer.symbology.props["when"] == "2026-03-20T12:00:00+00:00", (
            layer.symbology.props
        )
        assert layer.symbology.props["refraction"] == -6.0, layer.symbology.props

    def test_a_naive_moment_is_read_as_utc(self):
        """A datetime with no zone is UTC, which is what cleopatra assumes too."""
        canvas = _world_4326()
        canvas.nightshade(datetime(2026, 3, 20, 12, 0))
        layer = canvas.figure_spec.layers.get(canvas.layer_ids[-1])
        assert layer.symbology.props["when"] == "2026-03-20T12:00:00+00:00", (
            layer.symbology.props
        )

    def test_an_iso_string_is_a_moment_too(self):
        """A figure read back carries the instant as text, so the builder takes text."""
        artist = _world_4326().nightshade("2026-03-20T12:00:00+00:00")
        assert _covered(artist, (170.0, 0.0)), "lon 170 at noon UTC is night"

    def test_a_refraction_out_of_range_is_refused_and_not_described(self):
        """A positive refraction is not a terminator; the refusal leaves no layer behind."""
        canvas = _world_4326()
        with pytest.raises(ValueError, match="refraction"):
            canvas.nightshade(EQUINOX_NOON, refraction=5.0)
        assert canvas.layer_ids == [], canvas.layer_ids

    def test_a_moment_that_is_neither_a_datetime_nor_text_is_refused(self):
        """An instant has two spellings; a bare number is neither, and says so by type.

        Test scenario:
            ``when=2026`` reads as a year to a caller and as a POSIX timestamp to nobody in particular.
            Guessing either would shade the wrong night, so the builder refuses it with a ``TypeError``
            naming the type it got, and describes no layer.
        """
        canvas = _world_4326()
        with pytest.raises(TypeError, match="datetime or ISO 8601"):
            canvas.nightshade(2026)
        assert canvas.layer_ids == [], (
            f"a refused moment must leave no layer behind, got {canvas.layer_ids}"
        )

    def test_hiding_the_layer_hides_the_shade(self):
        """The shade is addressable by its id, like every other layer."""
        canvas = _world_4326()
        artist = canvas.nightshade(EQUINOX_NOON, name="night")
        canvas.set_visible("night", False)
        assert artist.get_visible() is False, "hiding the layer must hide its artist"

    def test_on_a_globe_the_near_side_night_is_shaded(self):
        """On an orthographic globe centred on the antimeridian at noon UTC, the disc centre is night."""
        canvas = Map(crs=projections.orthographic(lon=180, lat=0), globe=True)
        artist = canvas.nightshade(EQUINOX_NOON)
        assert artist is not None, "the night side faces the viewer"
        assert _covered(artist, (0.0, 0.0)), "the centre of the disc is at midnight"

    def test_the_globe_fill_follows_cleopatras_own_subsolar_point(self):
        """The globe's local solar-altitude copy must keep agreeing with the upstream it duplicates.

        Test scenario:
            ``_globe_nightshade`` writes the solar-altitude formula out locally, because cleopatra — which
            owns solar geometry — exposes the night side only as a polygon a globe cannot take
            (cleopatra#379). A local copy can drift from the upstream precision model it was derived from,
            so this pins the two together through the public API: a globe centred exactly on cleopatra's
            own ``subsolar_point`` sees the day hemisphere and nothing else, and one centred on its
            antipode sees the night hemisphere and nothing else. Both answers are decided entirely by
            where cleopatra puts the sun, so they change the moment the copy stops agreeing with it.
        """
        sun_lon, sun_lat = subsolar_point(JUNE_NOON)
        noon = Map(
            crs=projections.orthographic(lon=sun_lon, lat=sun_lat),
            globe=True,
        )
        assert noon.nightshade(JUNE_NOON) is None, (
            "a globe centred on the subsolar point shows the day hemisphere only"
        )
        midnight = Map(
            crs=projections.orthographic(lon=sun_lon - 180.0, lat=-sun_lat),
            globe=True,
        )
        artist = midnight.nightshade(JUNE_NOON)
        assert artist is not None, "the antisolar hemisphere is entirely at night"
        assert _covered(artist, (0.0, 0.0)), (
            "the antisolar point is the deepest night there is"
        )

    def test_a_globes_night_shade_honours_the_sample_count(self):
        """``n`` must reach the globe path, which recorded it and then drew a fixed grid regardless.

        Test scenario:
            A globe does not shade a night polygon; it fills the region of a display grid where the sun is
            below the terminator. That grid was a fixed 400 samples per side, so ``n`` was recorded in the
            figure and then ignored: a figure drawn flat and re-drawn on a globe silently changed fidelity,
            and the globe's resolution was not tunable at all. Resolution is observable in how finely the
            terminator contour is traced, so a quarter of the samples must trace it with well under half
            as many vertices.
        """
        fine, coarse = (
            Map(crs=projections.orthographic(lon=180, lat=0), globe=True).nightshade(
                EQUINOX_NOON, n=samples
            )
            for samples in (720, 180)
        )
        fine_vertices = sum(len(path.vertices) for path in fine.get_paths())
        coarse_vertices = sum(len(path.vertices) for path in coarse.get_paths())
        assert coarse_vertices < fine_vertices / 2, (coarse_vertices, fine_vertices)

    def test_a_globe_whose_visible_side_holds_no_night_draws_nothing(self):
        """A globe centred on the day side, asked for deep night, has none to shade — so no layer.

        Test scenario:
            A globe does not shade a night *polygon*; it fills the region of a display grid where the sun
            stands below ``refraction``. With the globe centred near the subsolar point and ``refraction``
            at -80, that region is the 10-degree cap around the antisolar point — entirely behind the
            globe. The flat map proves the same call does shade a world view, so the ``None`` is the
            hemisphere's doing and not a mis-specified request, and an undrawn layer is not described.
        """
        deep_night = -80.0
        canvas = Map(crs=projections.orthographic(lon=0, lat=0), globe=True)
        assert canvas.nightshade(EQUINOX_NOON, refraction=deep_night) is None, (
            "nothing on the day-side hemisphere is 80 degrees from the antisolar point"
        )
        assert canvas.layer_ids == [], (
            f"a layer that drew nothing must not be described, got {canvas.layer_ids}"
        )
        assert (
            _world_4326().nightshade(EQUINOX_NOON, refraction=deep_night) is not None
        ), "the same request over the whole world does reach the antisolar cap"


class TestTissot:
    """Geodesic circles of one ground radius, drawn through the display projection."""

    def test_it_draws_one_ring_per_centre(self):
        """Three centres, three rings."""
        canvas = _world_4326()
        artist = canvas.tissot([-90.0, 0.0, 90.0], [0.0, 0.0, 0.0])
        assert isinstance(artist, PolyCollection), type(artist)
        assert len(artist.get_paths()) == 3, len(artist.get_paths())

    def test_the_default_is_a_world_grid(self):
        """With no centres given, a grid covers the world so the distortion can be read everywhere."""
        artist = _world_4326().tissot()
        assert len(artist.get_paths()) > 10, len(artist.get_paths())

    def test_a_ring_has_the_angular_radius_of_its_ground_radius(self):
        """On lon/lat axes a 500 km ring at the equator spans 500 km / R in latitude each way."""
        artist = _world_4326().tissot([0.0], [0.0], radius_m=500_000.0)
        lat = artist.get_paths()[0].vertices[:, 1]
        expected = math.degrees(500_000.0 / MEAN_EARTH_R)
        assert lat.max() == pytest.approx(expected, rel=1e-3), lat.max()

    def test_mercator_inflates_a_ring_at_sixty_degrees_twice_over(self):
        """Mercator's scale factor is sec(lat), which is 2 at 60 degrees — the point of an indicatrix."""
        canvas = Map(crs=3857)
        canvas.ax.set_xlim(-2.0e7, 2.0e7)
        canvas.ax.set_ylim(-2.0e7, 2.0e7)
        artist = canvas.tissot([0.0, 0.0], [0.0, 60.0], radius_m=100_000.0)
        equator, sixty = (np.ptp(path.vertices[:, 0]) for path in artist.get_paths())
        assert sixty / equator == pytest.approx(2.0, rel=0.02), sixty / equator

    def test_a_ring_across_the_antimeridian_stays_one_ring(self):
        """A circle centred on lon 180 must not smear across the whole map.

        Test scenario:
            Wrapped longitudes put half the ring at +179 and half at -179; drawn as is, its outline would
            cross the whole world. Its width in lon/lat must stay that of a 500 km circle.
        """
        artist = _world_4326().tissot([180.0], [0.0], radius_m=500_000.0)
        width = np.ptp(artist.get_paths()[0].vertices[:, 0])
        expected = 2 * math.degrees(500_000.0 / MEAN_EARTH_R)
        assert width == pytest.approx(expected, rel=1e-2), width

    def test_a_ring_across_the_antimeridian_stays_one_ring_on_a_projected_crs(self):
        """The same ring on Web Mercator keeps its 500 km width instead of smearing across the world.

        Test scenario:
            The sibling test above runs on ``crs=4326``, where the reprojection is short-circuited, so it
            never exercises the projected path. Unwrapping the ring's longitudes past 180 survives only
            while the ring is drawn in degrees: handed to PROJ, lon 184.5 comes back as lon -175.5, so the
            vertices land alternately at the two edges of the Mercator world and the indicatrix is drawn as
            a zigzag band spanning it. The width must be the Mercator width of a 500 km circle at the
            equator — written out here from the spherical Mercator formula — not the width of the world.
        """
        canvas = Map(crs=3857)
        canvas.ax.set_xlim(-2.0e7, 2.0e7)
        canvas.ax.set_ylim(-2.0e7, 2.0e7)
        artist = canvas.tissot([180.0], [0.0], radius_m=500_000.0)
        assert len(artist.get_paths()) == 1, len(artist.get_paths())
        offset = math.degrees(500_000.0 / MEAN_EARTH_R)
        expected = _mercator(180.0 + offset, 0.0)[0] - _mercator(180.0 - offset, 0.0)[0]
        width = np.ptp(artist.get_paths()[0].vertices[:, 0])
        assert width == pytest.approx(expected, rel=1e-2), width

    def test_a_ring_on_the_seam_of_a_pacific_centred_crs_stays_one_ring(self):
        """A CRS whose central meridian is not 0 cuts elsewhere, and its own seam must be found.

        Test scenario:
            The seam used to be assumed to be at lon 180, so it was measured as the x-distance between
            lon -180 and lon 180. On a Pacific-centred Mercator those two longitudes are the *same* point,
            the measurement came out 0 — "no seam, leave the ring alone" — and a ring on the projection's
            real cut at ``lon_0 - 180`` smeared across the whole map. EPSG:3832 is the published form of
            it (PDC Mercator, ``lon_0 = 150``), so its cut is at lon -30, and a 500 km ring there must keep
            the Mercator width of a 500 km circle rather than the width of the world. The expected width is
            written out from the spherical Mercator formula about the same cut; EPSG:3832's own x is offset
            by its central meridian, which a *difference* of two x values cancels.
        """
        canvas = Map(crs=3832)
        canvas.ax.set_xlim(-2.0e7, 2.0e7)
        canvas.ax.set_ylim(-2.0e7, 2.0e7)
        artist = canvas.tissot([PDC_SEAM_LON], [0.0], radius_m=500_000.0)
        assert len(artist.get_paths()) == 1, len(artist.get_paths())
        offset = math.degrees(500_000.0 / MEAN_EARTH_R)
        expected = (
            _mercator(PDC_SEAM_LON + offset, 0.0)[0]
            - _mercator(PDC_SEAM_LON - offset, 0.0)[0]
        )
        width = np.ptp(artist.get_paths()[0].vertices[:, 0])
        assert width == pytest.approx(expected, rel=1e-2), width

    def test_a_far_side_ring_on_a_globe_is_left_out(self):
        """A circle behind the globe has no near-side outline; only the visible one is drawn."""
        canvas = Map(crs=projections.orthographic(lon=0, lat=0), globe=True)
        artist = canvas.tissot([0.0, 180.0], [0.0, 0.0])
        assert len(artist.get_paths()) == 1, len(artist.get_paths())

    def test_a_globe_that_can_place_no_ring_at_all_draws_nothing(self):
        """Every centre behind the globe leaves no ring, which is an undrawn layer rather than an empty one.

        Test scenario:
            The sibling test keeps one visible centre, so a collection is still drawn. With the only centre
            on the far side the drawer has nothing to hand cleopatra, and an empty ``PolyCollection`` would
            describe an indicatrix layer that marks nothing. It returns ``None`` and describes no layer, the
            way the vector builders answer geometry the display CRS cannot place.
        """
        canvas = Map(crs=projections.orthographic(lon=0, lat=0), globe=True)
        assert canvas.tissot([180.0], [0.0]) is None, (
            "a centre on the antimeridian is behind a globe centred on lon 0"
        )
        assert canvas.layer_ids == [], (
            f"a layer that drew nothing must not be described, got {canvas.layer_ids}"
        )

    def test_style_reaches_the_artist(self):
        """``edgecolor`` and friends are cleopatra's, forwarded to the collection."""
        artist = _world_4326().tissot([0.0], [0.0], edgecolor="crimson")
        assert tuple(artist.get_edgecolor()[0]) == pytest.approx(
            (220 / 255, 20 / 255, 60 / 255, 1.0)
        )

    def test_the_layer_is_described_with_its_centres(self):
        """The figure records the kind, the centres and the radius as plain values."""
        canvas = _world_4326()
        canvas.tissot([10.0, 20.0], [0.0, 30.0], radius_m=250_000.0)
        layer = canvas.figure_spec.layers.get(canvas.layer_ids[-1])
        props = layer.symbology.props
        assert layer.kind == "tissot", layer.kind
        assert (list(props["lons"]), list(props["lats"])) == (
            [10.0, 20.0],
            [0.0, 30.0],
        ), props
        assert props["radius_m"] == 250_000.0, props

    def test_centres_of_two_lengths_are_refused(self):
        """``lons`` and ``lats`` pair up; two lengths is a caller error, not a silent truncation."""
        canvas = _world_4326()
        with pytest.raises(ValueError, match="same shape"):
            canvas.tissot([0.0, 10.0], [0.0])

    def test_an_empty_centre_pair_is_refused_by_name(self):
        """An empty pair of centres is a caller mistake, not a drawing with no rings in it.

        Test scenario:
            ``None`` is the documented answer for "every ring was off-limb", which is a real figure the
            projection could not place. ``tissot([], [])`` drew nothing, described nothing and answered
            ``None`` too, so a caller whose centre list came out empty read it as the off-limb case and
            looked at the projection instead of at their own data. ``facet`` refuses an empty stack by
            name for exactly this reason, so an empty centre pair is refused the same way, before any
            ring is computed.
        """
        canvas = _world_4326()
        with pytest.raises(ValueError, match="no centres to draw"):
            canvas.tissot([], [])
        assert canvas.layer_ids == [], (
            f"a refused pair must leave no layer behind, got {canvas.layer_ids}"
        )

    @pytest.mark.parametrize(
        "lons, lats",
        [([0.0, 10.0], None), (None, [0.0, 10.0])],
        ids=["lons-only", "lats-only"],
    )
    def test_one_coordinate_without_the_other_is_refused(self, lons, lats):
        """Centres come as a pair, or not at all — half a pair is not the world grid.

        Args:
            lons: The longitudes given, or ``None``.
            lats: The latitudes given, or ``None``.

        Test scenario:
            Omitting both asks for the world grid, which is a different drawing from the centres given.
            Reading one argument and quietly defaulting the other would put rings on the equator (or on
            the prime meridian) that the caller never named, so the mismatch is refused before any ring is
            computed.
        """
        canvas = _world_4326()
        with pytest.raises(ValueError, match="together"):
            canvas.tissot(lons, lats)
        assert canvas.layer_ids == [], (
            f"a refused pair must leave no layer behind, got {canvas.layer_ids}"
        )


class TestTheExtentAFittedViewIsBoundedBy:
    """``_crs_display_extent`` measures what a freshly autoscaled view is pulled back inside.

    The sibling tests above pin the two outcomes a caller sees — a polar night fitted to EPSG:3031, and a
    shade that stays in the fitted view. These pin the measurement itself, whose answers are "a box in
    display coordinates" and "unknown": a CRS that declares nothing, one PROJ cannot read, and one whose
    declared corners do not project are all "unknown", and an unknown bound must leave the view alone
    rather than collapse it.
    """

    @pytest.mark.parametrize(
        "crs", ["not-a-crs", None, 0], ids=["gibberish", "none", "zero"]
    )
    def test_a_crs_proj_cannot_read_measures_as_unknown(self, crs):
        """Anything the PROJ database refuses is unknown, not an error on the way to a night shade.

        Args:
            crs: Something that is not a CRS at all.

        Test scenario:
            The extent is measured as a *bound* on a view matplotlib has already fitted to real geometry,
            so the measurement is advisory: a display CRS the lookup chokes on is a reason to leave the
            view as it is, not to fail a draw that has already succeeded. ``crs_from_user_input`` answers
            these three with a ``CRSError``, a ``TypeError`` and a ``ValueError`` respectively.
        """
        measured = decoration._crs_display_extent(crs)
        assert measured is None, (
            f"a CRS PROJ cannot read must measure as unknown, got {measured}"
        )

    def test_a_crs_declaring_no_area_of_use_measures_as_unknown(self):
        """A bare PROJ string carries no area of use, so there is no box to bound a view by.

        Test scenario:
            ``+proj=robin`` is a projection definition rather than a registered CRS: PROJ reads it, and the
            ``CRS`` it builds declares no area of use at all. That is a different path from an unreadable
            CRS — the lookup succeeds and answers ``None`` — and it has to end in the same "unknown", since
            a projection with no declared domain is exactly the case where guessing a bound would crop a
            view that is already correct.
        """
        measured = decoration._crs_display_extent("+proj=robin")
        assert measured is None, (
            f"a CRS with no area of use must measure as unknown, got {measured}"
        )

    def test_a_declared_area_that_will_not_project_measures_as_unknown(
        self, monkeypatch
    ):
        """When the declared corners cannot be projected the extent is unknown, not half a box.

        Args:
            monkeypatch: pytest's patcher, which restores pyramids' reprojection afterwards.

        Test scenario:
            The lon/lat area of use is projected into display coordinates, because what bounds a view is a
            box in the units the axes is drawn in. PROJ raises on some definitions rather than clamping,
            and a partial answer would bound the view by a box built from whichever samples survived — so
            the whole measurement is abandoned instead. EPSG:3031 is used because it *does* declare an area
            of use, which is what gets the code as far as the projection step.
        """

        def refuse(*_args, **_kwargs):
            """Stand in for a projection PROJ cannot carry out.

            Raises:
                RuntimeError: always; that is the point of the stand-in.
            """
            raise RuntimeError("proj_create: cannot build operation")

        monkeypatch.setattr(decoration, "reproject_coordinates", refuse)
        measured = decoration._crs_display_extent(3031)
        assert measured is None, (
            f"a declared area that will not project must measure as unknown, got {measured}"
        )

    def test_a_declared_area_that_projects_to_nothing_finite_measures_as_unknown(
        self, monkeypatch
    ):
        """Samples that all come back non-finite are no box either, and their min is not a bound.

        Args:
            monkeypatch: pytest's patcher, which restores pyramids' reprojection afterwards.

        Test scenario:
            A clipped projection has no image of part of its own declared area, and PROJ marks those
            samples ``inf``/``nan`` rather than raising. Taking ``min``/``max`` over them would hand the fit
            a non-finite box, which collapses every view it touches, so an all-non-finite answer is
            "unknown" too.
        """

        def nothing_finite(xs, ys, **_kwargs):
            """Answer every sample as non-finite, the way a clipped projection does.

            Args:
                xs: The sample longitudes.
                ys: The sample latitudes.

            Returns:
                Two lists of ``nan``, as long as the samples given.
            """
            return [math.nan] * len(list(xs)), [math.nan] * len(list(ys))

        monkeypatch.setattr(decoration, "reproject_coordinates", nothing_finite)
        measured = decoration._crs_display_extent(3031)
        assert measured is None, (
            f"an area that projects to nothing finite must measure as unknown, got {measured}"
        )

    def test_an_unknown_extent_leaves_a_fitted_view_whole(self):
        """A shade on a CRS with no declared extent keeps the view matplotlib fitted to it.

        Test scenario:
            ``+proj=robin`` measures as unknown (pinned above), and an unknown bound must not move a view
            that has just been fitted to real geometry. The whole night polygon therefore has to stay
            inside the fitted view: the EPSG:3031 fit in the sibling class above is the case where the view
            *is* pulled in, and this is the case where it must not be.
        """
        canvas = Map(crs="+proj=robin")
        artist = canvas.nightshade(EQUINOX_NOON)
        vertices = np.vstack([path.vertices for path in artist.get_paths()])
        finite = vertices[np.isfinite(vertices).all(axis=1)]
        xmin, xmax = canvas.ax.get_xlim()
        ymin, ymax = canvas.ax.get_ylim()
        assert finite.size, "the shade must have placeable vertices to begin with"
        assert finite[:, 0].min() >= xmin, (
            f"the fitted view starts at x {xmin} and must hold the westmost shaded vertex, got "
            f"{finite[:, 0].min()}"
        )
        assert finite[:, 0].max() <= xmax, (
            f"the fitted view ends at x {xmax} and must hold the eastmost shaded vertex, got "
            f"{finite[:, 0].max()}"
        )
        assert finite[:, 1].min() >= ymin, (
            f"the fitted view starts at y {ymin} and must hold the southmost shaded vertex, got "
            f"{finite[:, 1].min()}"
        )
        assert finite[:, 1].max() <= ymax, (
            f"the fitted view ends at y {ymax} and must hold the northmost shaded vertex, got "
            f"{finite[:, 1].max()}"
        )

    def test_a_view_that_does_not_meet_the_extent_is_left_alone(self):
        """Intersecting a view with an extent it misses entirely would empty it, so it is not done.

        Test scenario:
            The fit keeps the overlap of the autoscaled view and the projection's own extent. A view with
            no overlap at all — here a decade of metres past the ``+/- 3.3e6`` m EPSG:3031 declares for
            itself, on both axes — intersects to a reversed interval, and setting that would flip the axes
            and hide everything on it. Both axes must therefore be left exactly where they were found.
        """
        far_away = (1.0e9, 2.0e9)
        canvas = Map(crs=3031)
        canvas.ax.set_xlim(*far_away)
        canvas.ax.set_ylim(*far_away)
        decoration._fit_within_crs_extent(canvas)
        assert canvas.ax.get_xlim() == far_away, (
            f"a view past the extent must keep its x limits, got {canvas.ax.get_xlim()}"
        )
        assert canvas.ax.get_ylim() == far_away, (
            f"a view past the extent must keep its y limits, got {canvas.ax.get_ylim()}"
        )

    def test_a_layer_drawn_after_the_fit_still_frames_itself(self):
        """Narrowing the view to the projection's extent must not freeze what is drawn next.

        Test scenario:
            The fit intersects the autoscaled view with the CRS's declared extent, and matplotlib turns
            autoscaling **off** on an axis whose limits are set outright. A night shade drawn before the
            data therefore pinned the map to the ``+/- 3.3e6`` m EPSG:3031 declares for itself, and a
            raster reaching ``+/- 7e6`` m — a real, projectable grid, since the declared area of use is
            not the projection's domain — was cropped to under half its width with no warning. The raster
            must still frame itself: its own bbox, read from the dataset rather than from the axes, has to
            fit inside the view that holds after it is drawn.
        """
        canvas = Map(crs=3031)
        canvas.nightshade(DECEMBER_NOON)
        wide = _wide_polar_raster()
        west, east = _POLAR_RASTER_WEST, _POLAR_RASTER_WEST + _POLAR_RASTER_WIDTH
        canvas.field(wide)
        xmin, xmax = sorted(canvas.ax.get_xlim())
        assert xmin <= west, (
            f"the raster starts at x {west} and the view it framed must reach it, got {xmin}"
        )
        assert xmax >= east, (
            f"the raster ends at x {east} and the view it framed must reach it, got {xmax}"
        )


class TestTheProjectedRingSeam:
    """The two helpers ``draw_tissot`` projects its rings through, asked for nothing.

    ``Map.tissot`` refuses an empty centre pair by name, so these answers are the seam's own contract
    rather than something a caller reaches: they are what keeps ``draw_tissot`` from measuring a seam at no
    latitude at all, where the lon/lat edge pairs it builds would have no rows to project.
    """

    def test_no_latitudes_is_no_seam_widths(self):
        """Nothing to measure is an empty width array, of the dtype a measured one has.

        Test scenario:
            The widths are consumed by ``zip`` against the rings, so the answer for no rings has to be an
            empty float array rather than ``None`` or a scalar — one shape for both cases.
        """
        periods = decoration._antimeridian_periods(Map(crs=3857), [])
        assert periods.shape == (0,), (
            f"no latitudes must measure no widths, got shape {periods.shape}"
        )
        assert periods.dtype == np.dtype(float), (
            f"an empty width array must still be float, got {periods.dtype}"
        )

    def test_a_shifted_central_meridian_moves_the_seam_it_is_measured_at(self):
        """The world width is measured at the projection's own cut, not at lon 180.

        Test scenario:
            The measurement pinned here is the one the ring test above consumes, read directly so a
            regression names the cause rather than a ring width. EPSG:3832 and Web Mercator are the same
            projection on the same ellipsoid with different central meridians, so the world is exactly as
            wide on both — while the measurement that assumed the cut was at lon 180 answered ``0`` for the
            Pacific-centred one, which means "no seam, leave the ring alone". Web Mercator is the control:
            its own measurement must not move.
        """
        shifted = decoration._antimeridian_periods(Map(crs=3832), [0.0, 60.0])
        greenwich = decoration._antimeridian_periods(Map(crs=3857), [0.0, 60.0])
        assert shifted == pytest.approx(greenwich, rel=1e-9), (
            f"the same projection must measure the same world width, got {shifted} against {greenwich}"
        )
        assert float(shifted[0]) > 4.0e7, (
            f"a Mercator world is over 4.0e7 m wide; the Pacific-centred one measured {shifted[0]}"
        )

    def test_no_circles_is_no_projected_rings(self):
        """No rings in, no rings out — and no reprojection attempted on the way.

        Test scenario:
            The centres are projected in one call beside the rings, so an empty circle list has to short
            out before that call rather than reproject an empty coordinate pair.
        """
        rings = decoration._projected_rings(Map(crs=3857), [], [], [])
        assert rings == [], f"no circles must project to no rings, got {rings}"


class TestTheSampleCountIsRefusedTheSameWayOnEveryFrame:
    """``n`` counts the samples a ring is drawn from, so a count that cannot form one is refused everywhere."""

    @pytest.mark.parametrize("globe", [False, True], ids=["flat", "globe"])
    @pytest.mark.parametrize(
        "samples", [-5, 0, 3], ids=["negative", "zero", "below-four"]
    )
    def test_a_sample_count_too_small_for_a_ring_is_refused(self, globe, samples):
        """``nightshade(n=…)`` below four is refused on a flat map and on a globe alike.

        Args:
            globe: Whether the map is drawn as a globe.
            samples: A count too small to form a ring.

        Test scenario:
            cleopatra refuses the count while sampling the flat terminator, but a globe never calls that
            sampler — it fills by sun altitude over its own grid — so the same keyword used to draw happily
            there and record itself (``props['n'] == -5``). The builder refuses it, so one keyword has one
            answer on both frames.
        """
        canvas = Map(crs=3857, globe=globe)
        with pytest.raises(ValueError, match=r"nightshade\(\) needs n="):
            canvas.nightshade(JUNE_NOON, n=samples)
        assert canvas.layer_ids == [], (
            f"a refused shade must add no layer, got {canvas.layer_ids}"
        )

    @pytest.mark.parametrize("globe", [False, True], ids=["flat", "globe"])
    def test_a_tissot_sample_count_too_small_for_a_ring_is_refused(self, globe):
        """``tissot(n=…)`` below four is refused before any ring is placed.

        Args:
            globe: Whether the map is drawn as a globe.

        Test scenario:
            The same keyword on the other overlay that samples a ring.
        """
        canvas = Map(crs=3857, globe=globe)
        with pytest.raises(ValueError, match=r"tissot\(\) needs n="):
            canvas.tissot([0.0], [0.0], radius_m=5.0e5, n=3)
        assert canvas.layer_ids == [], (
            f"a refused indicatrix must add no layer, got {canvas.layer_ids}"
        )

    @pytest.mark.parametrize("globe", [False, True], ids=["flat", "globe"])
    def test_the_smallest_count_that_forms_a_ring_still_draws(self, globe):
        """Four samples is the smallest ring, and it is accepted on both frames.

        Args:
            globe: Whether the map is drawn as a globe.

        Test scenario:
            The refusal must sit immediately below the documented floor, not above it.
        """
        canvas = Map(crs=3857, globe=globe)
        canvas.nightshade(JUNE_NOON, n=4, name="ns")
        assert canvas.layer_ids == ["ns"], f"n=4 should draw, got {canvas.layer_ids}"


class TestTheTerminatorAltitudeIsRefusedTheSameWayOnEveryFrame:
    """``refraction`` is the solar altitude that *defines* the terminator, so it has one valid range."""

    @pytest.mark.parametrize("globe", [False, True], ids=["flat", "globe"])
    @pytest.mark.parametrize(
        "refraction",
        [10.0, 0.001, -90.0, -120.0],
        ids=["above-horizon", "just-above", "at-nadir", "below-nadir"],
    )
    def test_an_altitude_that_is_not_a_terminator_is_refused(self, globe, refraction):
        """``nightshade(refraction=…)`` outside ``(-90, 0]`` is refused on a flat map and on a globe alike.

        Args:
            globe: Whether the map is drawn as a globe.
            refraction: A solar altitude that defines no terminator.

        Test scenario:
            cleopatra refuses the range inside ``add_nightshade``, which only the flat path calls — the
            globe path fed ``refraction`` straight into ``sin(radians(refraction))``, so ``+10`` painted a
            large part of the **day** side as night and a value past the nadir answered either a wrong
            shade or ``None``, which the public API documents as "a globe whose visible side holds no
            night". An out-of-range argument was therefore indistinguishable from a legitimate empty
            result. This is the sibling of ``n``: one keyword, one answer, whatever frame it is drawn on.
        """
        canvas = Map(crs=3857, globe=globe)
        with pytest.raises(ValueError, match=r"nightshade\(\) needs refraction="):
            canvas.nightshade(JUNE_NOON, refraction=refraction)
        assert canvas.layer_ids == [], (
            f"a refused shade must add no layer, got {canvas.layer_ids}"
        )

    @pytest.mark.parametrize("globe", [False, True], ids=["flat", "globe"])
    @pytest.mark.parametrize(
        "refraction",
        [0.0, -0.83, -18.0],
        ids=["geometric", "sunset", "astronomical"],
    )
    def test_an_altitude_inside_the_range_still_draws(self, globe, refraction):
        """The documented terminator lines all draw, on a flat map and on a globe alike.

        Args:
            globe: Whether the map is drawn as a globe.
            refraction: A solar altitude inside ``(-90, 0]``.

        Test scenario:
            ``0`` is the geometric terminator and is **in** range while ``-90`` is out of it, so a refusal
            that got the closed end wrong would reject the geometric line the default is measured from.
            The sample count is pinned the same way by its sibling class above.
        """
        canvas = Map(crs=3857, globe=globe)
        canvas.nightshade(JUNE_NOON, refraction=refraction, name="ns")
        assert canvas.layer_ids == ["ns"], (
            f"refraction={refraction} should draw, got {canvas.layer_ids}"
        )

    def test_the_open_end_of_the_range_is_accepted_right_up_to_the_nadir(self):
        """An altitude a thousandth of a degree above the nadir is in range; the nadir itself is not.

        Test scenario:
            The open end is pinned on the validator rather than through a drawing, because an altitude
            that deep legitimately leaves a globe's visible side with no night at all — which is the
            documented ``None``, not a refusal, so a drawn test could not tell the two apart. ``-90`` is
            the nadir: the sun cannot be below it, so no terminator is defined there, and the refusal has
            to fall between the two.
        """
        assert decoration._terminator_altitude(-89.999, "nightshade()") == -89.999, (
            "an altitude just above the nadir defines a terminator and must be accepted"
        )
        with pytest.raises(ValueError, match=r"nightshade\(\) needs refraction="):
            decoration._terminator_altitude(-90.0, "nightshade()")
