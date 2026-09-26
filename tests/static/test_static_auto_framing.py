"""Order 26's 2-D half on the static tier: ``set_bounds`` fits the data, and pads.

Order 27a took the *name*. ``set_bounds`` framed the axes on a rectangle a caller worked out themselves, and
that was all it did: there was no ``padding``, no ``None`` that fits the figure to what it draws, and — the
part nothing else could work around — the view the figure reported carried no region at all, so a map framed
by its own data described itself as unframed. Wave 6's basemap guard had to record the axes extent on the
layer for exactly that reason, rather than reading it off the panel.

This module holds the capability: what ``set_bounds(None)`` frames on, what it refuses to frame on, what
``padding`` does to a frame that came from either spelling, and what the panel's view says afterwards. The
region is read from the **axes limits** and from ``viewport.bounds`` together, because a frame the figure
cannot report is the half that was missing.
"""

import geopandas as gpd
import pytest
from pyramids.feature import FeatureCollection
from shapely.geometry import Point

from digitalearth.base.spec import Bounds
from digitalearth.static import Map

#: Two point layers in EPSG:4326, built so that neither one's rectangle contains the other and the union takes
#: a different corner from each. Degrees, and a map drawn in degrees, so every expected number below is the
#: literal input rather than a reprojection nobody can check by eye.
NORTH_WEST = ((0.0, 0.0), (10.0, 10.0))
SOUTH_EAST = ((5.0, -20.0), (30.0, -5.0))

#: The union of the two, in matplotlib's ``(xmin, xmax), (ymin, ymax)`` order.
BOTH = ((0.0, 30.0), (-20.0, 10.0))


def _points(corners):
    """Return a two-point feature collection at `corners`.

    Args:
        corners: Two ``(lon, lat)`` pairs, which become the layer's extent exactly — a scatter's data limits
            are its points, with no marker padding (measured).

    Returns:
        FeatureCollection: the layer's data, in EPSG:4326.
    """
    return FeatureCollection(
        gpd.GeoDataFrame(
            geometry=[Point(*corner) for corner in corners], crs="EPSG:4326"
        )
    )


@pytest.fixture
def flat():
    """Yield a map drawn in degrees, closed on the way out.

    Yields:
        A :class:`~digitalearth.static.map.Map` in EPSG:4326.
    """
    scene = Map(crs=4326)
    yield scene
    scene.close()


def _limits(scene):
    """Return the axes limits as plain floats.

    Args:
        scene: The map to read.

    Returns:
        ``((xmin, xmax), (ymin, ymax))`` — what matplotlib was left holding, which is the only evidence that
        a framing call framed anything.
    """
    return (
        tuple(float(value) for value in scene.ax.get_xlim()),
        tuple(float(value) for value in scene.ax.get_ylim()),
    )


class TestNoneFitsTheData:
    """``set_bounds()`` with no rectangle frames the figure on what it draws."""

    def test_one_layer_is_framed_on_its_own_extent(self, flat):
        """The simplest case, and the one a caller reaches for first.

        Args:
            flat: A map in degrees.
        """
        flat.points(_points(NORTH_WEST))
        flat.set_bounds()
        assert _limits(flat) == ((0.0, 10.0), (0.0, 10.0)), _limits(flat)

    def test_two_layers_are_framed_on_the_rectangle_that_holds_both(self, flat):
        """The union, taking one corner from each layer.

        Args:
            flat: A map in degrees.
        """
        flat.points(_points(NORTH_WEST))
        flat.points(_points(SOUTH_EAST))
        flat.set_bounds(None)
        assert _limits(flat) == BOTH, _limits(flat)

    def test_a_reference_layer_does_not_drag_the_frame_out_to_the_world(self, flat):
        """A basemap, a graticule and a coastline are not what the figure is about.

        Args:
            flat: A map in degrees.

        Test scenario:
            ``coastlines()`` draws the whole globe. Framing on every layer would answer every
            ``set_bounds(None)`` on a decorated map with the world, which is the one answer that is never
            what was asked. The bands the package already sorts layers into say which layers count: the
            figure's subject is the ``data`` band, and a coastline is ``reference``.
        """
        flat.points(_points(NORTH_WEST))
        flat.coastlines()
        flat.set_bounds(None)
        assert _limits(flat) == ((0.0, 10.0), (0.0, 10.0)), _limits(flat)

    def test_a_layer_built_hidden_is_not_framed_on(self, flat):
        """A layer the figure is not drawing is not part of what it shows.

        Args:
            flat: A map in degrees.
        """
        flat.points(_points(NORTH_WEST))
        flat.points(_points(SOUTH_EAST), visible=False)
        flat.set_bounds(None)
        assert _limits(flat) == ((0.0, 10.0), (0.0, 10.0)), _limits(flat)

    def test_nothing_to_frame_on_is_refused_rather_than_ignored(self, flat):
        """Doing nothing would look exactly like the call was dropped.

        Args:
            flat: A map in degrees.

        Test scenario:
            The message has to say what to do instead, because the caller cannot see from the outside that
            the only layer they added was a decoration.
        """
        flat.coastlines()
        with pytest.raises(ValueError, match="nothing to frame on"):
            flat.set_bounds(None)


class TestPadding:
    """Breathing room, as a fraction of the span the frame covers."""

    def test_a_fitted_frame_is_grown_by_the_fraction(self, flat):
        """A tenth of a ten-degree span is one degree on each side.

        Args:
            flat: A map in degrees.
        """
        flat.points(_points(NORTH_WEST))
        flat.set_bounds(None, padding=0.1)
        assert _limits(flat) == ((-1.0, 11.0), (-1.0, 11.0)), _limits(flat)

    def test_an_explicit_rectangle_is_grown_the_same_way(self, flat):
        """One keyword, one meaning, whichever spelling the frame arrived in.

        Args:
            flat: A map in degrees.
        """
        flat.set_bounds(Bounds(0.0, 0.0, 10.0, 20.0, crs=4326), padding=0.5)
        assert _limits(flat) == ((-5.0, 15.0), (-10.0, 30.0)), _limits(flat)

    def test_a_bare_sequence_is_grown_the_same_way(self, flat):
        """The matplotlib-ordered form takes it too.

        Args:
            flat: A map in degrees.
        """
        flat.set_bounds([0.0, 10.0, 0.0, 20.0], padding=0.5)
        assert _limits(flat) == ((-5.0, 15.0), (-10.0, 30.0)), _limits(flat)

    def test_no_padding_is_the_default_so_a_named_frame_is_exactly_itself(self, flat):
        """The committed baselines frame themselves through this method; none of them may move.

        Args:
            flat: A map in degrees.
        """
        flat.set_bounds([0.0, 100.0, 0.0, 50.0])
        assert _limits(flat) == ((0.0, 100.0), (0.0, 50.0)), _limits(flat)

    def test_a_padding_that_would_invert_the_frame_is_refused(self, flat):
        """``Bounds.padded`` owns the refusal; the message has to reach the caller.

        Args:
            flat: A map in degrees.
        """
        with pytest.raises(ValueError, match="would invert"):
            flat.set_bounds([0.0, 10.0, 0.0, 10.0], padding=-0.9)


class TestTheViewReportsTheFrame:
    """A figure framed on a region has to be able to say which region."""

    def test_a_named_rectangle_is_reported_by_the_panel(self, flat):
        """What Wave 6's basemap guard had to read off the axes instead.

        Args:
            flat: A map in degrees.
        """
        flat.set_bounds(Bounds(0.0, 0.0, 10.0, 20.0, crs=4326))
        reported = flat.figure_spec.panels[0].view.bounds
        assert (reported.as_bbox(), reported.crs) == ([0.0, 0.0, 10.0, 20.0], 4326)

    def test_a_fitted_frame_is_reported_too(self, flat):
        """The half that was measured absent: a map framed by its data said it was unframed.

        Args:
            flat: A map in degrees.
        """
        flat.points(_points(NORTH_WEST))
        flat.points(_points(SOUTH_EAST))
        flat.set_bounds(None)
        assert flat.viewport.bounds.as_bbox() == [0.0, -20.0, 30.0, 10.0]

    def test_an_unframed_map_still_reports_no_region(self, flat):
        """``None`` has to keep meaning "nobody has framed this", or it says nothing.

        Args:
            flat: A map in degrees.
        """
        flat.points(_points(NORTH_WEST))
        assert flat.viewport.bounds is None, flat.viewport

    def test_a_frame_from_a_named_domain_is_reported(self):
        """``set_domain`` frames through ``set_bounds``, so the view has to follow it.

        Test scenario:
            A view holds a region *or* a named domain, never both — so the domain the map was built with
            gives way to the rectangle it was actually framed on. That is what
            :meth:`~digitalearth.base.spec.viewport.Viewport.framed` already does, and reporting a stale
            domain beside a live frame would be the alternative.
        """
        scene = Map(crs=4326, domain="europe")
        try:
            scene.set_domain()
            view = scene.viewport
            assert (view.bounds.as_bbox(), view.domain) == (
                [-25.0, 34.0, 45.0, 72.0],
                None,
            )
        finally:
            scene.close()

    def test_a_declared_domain_survives_until_something_frames_the_axes(self):
        """A map that was told a region but never framed on it still reports the name."""
        scene = Map(crs=4326, domain="europe")
        try:
            assert scene.viewport.domain == "europe", scene.viewport
        finally:
            scene.close()

    def test_set_global_reports_the_whole_projection(self):
        """The widest frame this tier can set is still a frame.

        Test scenario:
            ``set_global`` goes through ``set_bounds`` with the projection's own limits, so the region it
            reports is the projection domain rather than ``None``.
        """
        scene = Map(crs=4326)
        try:
            scene.set_global()
            assert scene.viewport.bounds is not None, scene.viewport
        finally:
            scene.close()


class TestAFlippedAxisSurvives:
    """The sequence form can invert an axis, and that was its contract before this."""

    def test_the_axis_still_runs_backwards(self, flat):
        """``[10, 0, ...]`` is how matplotlib spells ``invert_xaxis`` through the limits.

        Args:
            flat: A map in degrees.
        """
        flat.set_bounds([10.0, 0.0, 0.0, 10.0])
        assert _limits(flat) == ((10.0, 0.0), (0.0, 10.0)), _limits(flat)

    def test_padding_grows_a_backwards_axis_outwards(self, flat):
        """Growing it must widen the view, not narrow it.

        Args:
            flat: A map in degrees.
        """
        flat.set_bounds([10.0, 0.0, 0.0, 10.0], padding=0.1)
        assert _limits(flat) == ((11.0, -1.0), (-1.0, 11.0)), _limits(flat)

    def test_the_view_reports_the_region_rather_than_the_direction(self, flat):
        """A ``Bounds`` cannot hold a rectangle the wrong way round, and should not.

        Args:
            flat: A map in degrees.

        Test scenario:
            The view answers *which region is shown*, and that is the same region either way round. The
            inversion is a property of the axes, not of the region, so it is read from the axes — which the
            test above does.
        """
        flat.set_bounds([10.0, 0.0, 0.0, 10.0])
        assert flat.viewport.bounds.as_bbox() == [0.0, 0.0, 10.0, 10.0]

    def test_a_sequence_of_the_wrong_length_is_still_refused(self, flat):
        """The one refusal this method already had.

        Args:
            flat: A map in degrees.
        """
        with pytest.raises(ValueError, match="exactly 4 values"):
            flat.set_bounds([0.0, 1.0, 0.0, 1.0, 2.0])


class TestTheOldSpellingReachesAllOfIt:
    """``set_extent`` is a live alias, so it forwards the new keyword as well as the old rectangle."""

    def test_the_alias_fits_the_data_too(self, flat):
        """A caller who has not renamed their calls still gets the capability.

        Args:
            flat: A map in degrees.
        """
        flat.points(_points(NORTH_WEST))
        with pytest.warns(DeprecationWarning):
            flat.set_extent(None, padding=0.1)
        assert _limits(flat) == ((-1.0, 11.0), (-1.0, 11.0)), _limits(flat)
