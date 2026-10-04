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

from digitalearth.static import Map, projections

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
