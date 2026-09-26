"""Order 26's 2-D half on the interactive tier: a public way to frame the map.

This tier had **no framing method at all**, under any spelling. A Bokeh figure is pannable, so the view was
left to whatever the viewer did with it — which is a fair default and a poor contract: a map of one country
opened on whatever HoloViews autoranged to, and a caller who wanted to open it on a region had no call to
make. ``PENDING`` said so in those words.

``set_bounds`` is that call, spelled and behaving as the Core declares it and as the web and 3-D tiers
already answer: a region or ``None``, ``padding``, and the map back so it chains. The claims below are read
from what the **engine** was handed — the ``xlim``/``ylim`` options on the composed object — and from the
panel's view, because a frame the figure cannot report is the half that was missing on the static tier too.

Runs in the ``interactive`` pixi env.
"""

import geopandas as gpd
import pytest
from shapely.geometry import Point

from digitalearth.base.spec import Bounds
from digitalearth.interactive import InteractiveMap

hv = pytest.importorskip("holoviews")
gv = pytest.importorskip("geoviews")
FeatureCollection = pytest.importorskip("pyramids.feature").FeatureCollection

#: Two point layers in EPSG:4326, drawn on a map in EPSG:4326 so every expected number is the literal input
#: rather than a reprojection nobody can check by eye. Neither rectangle contains the other.
NORTH_WEST = ((0.0, 0.0), (10.0, 10.0))
SOUTH_EAST = ((5.0, -20.0), (30.0, -5.0))


def _points(corners):
    """Return a two-point feature collection at `corners`.

    Args:
        corners: Two ``(lon, lat)`` pairs, which become the layer's extent exactly.

    Returns:
        FeatureCollection: the layer's data, in EPSG:4326.
    """
    return FeatureCollection(
        gpd.GeoDataFrame(
            geometry=[Point(*corner) for corner in corners], crs="EPSG:4326"
        )
    )


@pytest.fixture
def flat() -> InteractiveMap:
    """Return a map drawn in degrees, so the frame is read in the units it was given.

    Returns:
        An :class:`~digitalearth.interactive.map.InteractiveMap` in EPSG:4326.
    """
    return InteractiveMap(crs=4326)


def _framed(scene):
    """Return the limits the engine was handed for the composed figure.

    Args:
        scene: The map to render.

    Returns:
        ``(xlim, ylim)`` as plain float pairs, read off the options HoloViews holds for the rendered object
        — which is what actually frames the figure — or ``(None, None)`` when nothing framed it.
    """
    options = hv.Store.lookup_options("bokeh", scene.render(), "plot").kwargs
    limits = []
    for name in ("xlim", "ylim"):
        asked = options.get(name)
        limits.append(None if asked is None else tuple(float(v) for v in asked))
    return tuple(limits)


class TestTheTierCanBeFramedAtAll:
    """The method exists, frames the figure, and hands the map back."""

    def test_a_named_rectangle_reaches_the_engine(self, flat):
        """The limits HoloViews is given, which is where framing happens on this tier.

        Args:
            flat: A map in degrees.
        """
        flat.points(_points(NORTH_WEST))
        flat.set_bounds(Bounds(0.0, 0.0, 10.0, 20.0, crs=4326))
        assert _framed(flat) == ((0.0, 10.0), (0.0, 20.0)), _framed(flat)

    def test_a_bare_sequence_is_read_as_west_south_east_north(self, flat):
        """The ordering every other tier's rectangle uses, and the one ``Bounds`` writes.

        Args:
            flat: A map in degrees.

        Test scenario:
            The static tier's sequence form is matplotlib's ``[xmin, xmax, ymin, ymax]``, which it keeps
            because that was already its contract. This tier has no such history, so it takes the bbox
            order — the one ``Bounds.as_bbox``, the web tier's ``set_bounds`` and pyramids all use.
        """
        flat.points(_points(NORTH_WEST))
        flat.set_bounds((0.0, 0.0, 10.0, 20.0))
        assert _framed(flat) == ((0.0, 10.0), (0.0, 20.0)), _framed(flat)

    def test_it_chains(self, flat):
        """The Core declares ``returns="self"``, and the other tiers answer that way.

        Args:
            flat: A map in degrees.
        """
        assert flat.set_bounds((0.0, 0.0, 1.0, 1.0)) is flat

    def test_a_sequence_of_the_wrong_length_is_refused(self, flat):
        """Three numbers cannot be a rectangle, and guessing which one is missing would be worse.

        Args:
            flat: A map in degrees.
        """
        with pytest.raises(ValueError, match="west, south, east, north"):
            flat.set_bounds((0.0, 0.0, 1.0))

    def test_an_unframed_map_hands_the_engine_no_limits(self, flat):
        """The pannable default has to survive: a map nobody framed is still framed by the viewer.

        Args:
            flat: A map in degrees.
        """
        flat.points(_points(NORTH_WEST))
        assert _framed(flat) == (None, None), _framed(flat)


class TestNoneFitsTheData:
    """``set_bounds()`` with no rectangle frames the map on what it draws."""

    def test_two_layers_are_framed_on_the_rectangle_that_holds_both(self, flat):
        """The union, taking one corner from each layer.

        Args:
            flat: A map in degrees.
        """
        flat.points(_points(NORTH_WEST))
        flat.points(_points(SOUTH_EAST))
        flat.set_bounds(None)
        assert _framed(flat) == ((0.0, 30.0), (-20.0, 10.0)), _framed(flat)

    def test_a_reference_layer_does_not_drag_the_frame_out_to_the_world(self, flat):
        """A coastline is drawn around the subject rather than being it.

        Args:
            flat: A map in degrees.

        Test scenario:
            A graticule spans the whole world, and GeoViews' ``Feature`` reports its range in **degrees**
            whatever the display CRS is (measured), so unioning one in would be wrong in units as well as
            in extent. The bands the package sorts layers into already say which layers count, and a
            graticule is ``reference``.
        """
        flat.points(_points(NORTH_WEST))
        flat.graticule(spacing=30)
        flat.set_bounds(None)
        assert _framed(flat) == ((0.0, 10.0), (0.0, 10.0)), _framed(flat)

    def test_a_layer_built_hidden_is_not_framed_on(self, flat):
        """A layer the map is not drawing is not part of what it shows.

        Args:
            flat: A map in degrees.
        """
        flat.points(_points(NORTH_WEST))
        flat.points(_points(SOUTH_EAST), visible=False)
        flat.set_bounds(None)
        assert _framed(flat) == ((0.0, 10.0), (0.0, 10.0)), _framed(flat)

    def test_nothing_to_frame_on_is_refused_rather_than_ignored(self, flat):
        """Doing nothing would look exactly like the call was dropped.

        Args:
            flat: A map in degrees.
        """
        flat.graticule(spacing=30)
        with pytest.raises(ValueError, match="nothing to frame on"):
            flat.set_bounds(None)


class TestPadding:
    """The keyword the Core declares beside the name, in this tier's own units."""

    def test_a_fitted_frame_is_grown_by_the_fraction(self, flat):
        """A tenth of a ten-degree span is one degree on each side.

        Args:
            flat: A map in degrees.
        """
        flat.points(_points(NORTH_WEST))
        flat.set_bounds(None, padding=0.1)
        assert _framed(flat) == ((-1.0, 11.0), (-1.0, 11.0)), _framed(flat)

    def test_an_explicit_rectangle_is_grown_the_same_way(self, flat):
        """One keyword, one meaning, whichever spelling the frame arrived in.

        Args:
            flat: A map in degrees.
        """
        flat.set_bounds((0.0, 0.0, 10.0, 20.0), padding=0.5)
        assert _framed(flat) == ((-5.0, 15.0), (-10.0, 30.0)), _framed(flat)

    def test_a_padding_that_would_invert_the_frame_is_refused(self, flat):
        """``Bounds.padded`` owns the refusal; the message has to reach the caller.

        Args:
            flat: A map in degrees.
        """
        with pytest.raises(ValueError, match="would invert"):
            flat.set_bounds((0.0, 0.0, 10.0, 10.0), padding=-0.9)


class TestTheViewReportsTheFrame:
    """A figure framed on a region has to be able to say which region."""

    def test_a_named_rectangle_is_reported_by_the_panel(self, flat):
        """The read-back the static and web tiers both give.

        Args:
            flat: A map in degrees.
        """
        flat.set_bounds(Bounds(0.0, 0.0, 10.0, 20.0, crs=4326))
        reported = flat.figure_spec.panels[0].view.bounds
        assert (reported.as_bbox(), reported.crs) == ([0.0, 0.0, 10.0, 20.0], 4326)

    def test_a_fitted_frame_is_reported_too(self, flat):
        """The region the map framed itself on, not the one it was told.

        Args:
            flat: A map in degrees.
        """
        flat.points(_points(NORTH_WEST))
        flat.points(_points(SOUTH_EAST))
        flat.set_bounds(None)
        assert flat.viewport.bounds.as_bbox() == [0.0, -20.0, 30.0, 10.0]

    def test_an_unframed_map_still_reports_no_region(self, flat):
        """``None`` has to keep meaning "nobody has framed this".

        Args:
            flat: A map in degrees.
        """
        flat.points(_points(NORTH_WEST))
        assert flat.viewport.bounds is None, flat.viewport

    def test_a_rectangle_in_another_crs_is_converted_before_it_is_reported(self):
        """A view holds its region in its own CRS, so the conversion happens on the way in.

        Test scenario:
            One degree east of the prime meridian is 111319 m in Web Mercator. A frame reported in degrees
            by a map drawn in metres would send every reader to the Gulf of Guinea.
        """
        scene = InteractiveMap(crs=3857)
        scene.set_bounds(Bounds(0.0, 0.0, 1.0, 1.0, crs=4326))
        reported = scene.viewport.bounds
        assert (reported.crs, round(reported.xmax)) == (3857, 111319), (
            reported.as_bbox()
        )
