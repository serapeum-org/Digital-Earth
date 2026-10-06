"""Tests for digitalearth.static.maps.inset — the locator map (ST-19).

Two things carry the risk and are covered closely. The **extent box is drawn in the locator's own CRS**, and
a box reprojected by its corners alone lands wrong — the EPSG:3031 case below is the proof: all four corners
of a polar-stereographic square sit at the *same* latitude, so a corner-only box has zero height. And a
borrowed axes is **spared** by ``Scene.close`` (#371, resolved), so closing the locator leaves the caller's
figure open — measured rather than assumed.
"""

import re

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.patches import PathPatch, Polygon
from pyramids.base.crs import reproject_coordinates

from digitalearth.static import Map, projections
from digitalearth.static.maps.inset import _CORNER_PAD, _CORNERS

#: The largest side a named corner can hold: the inset keeps :data:`_CORNER_PAD` on both sides of it.
LARGEST_CORNER = 1.0 - 2.0 * _CORNER_PAD

#: A box in EPSG:3031 (Antarctic Polar Stereographic) whose four corners all reproject to one latitude.
POLAR_BOX = (-2.0e6, -2.0e6, 2.0e6, 2.0e6)

#: The lon/lat rectangle the locator's extent box is checked against in the same-CRS cases.
PLAIN_BOX = (2.0, 3.0, 8.0, 9.0)


def _marked_ring(locator, main) -> np.ndarray:
    """Mark ``main``'s extent on ``locator`` and return the ring of the box that was drawn.

    ``mark_extent`` hands back the map (round 2, L7), so the patch is read off
    :meth:`~digitalearth.static.scene.Scene.artist` by the id the call was given — which also means a
    box that was skipped raises here rather than reading as an empty ring.

    Args:
        locator: The map the box is drawn on.
        main: The map whose extent is marked.

    Returns:
        The drawn ``Polygon``'s vertices, as an ``(n, 2)`` float array.
    """
    locator.mark_extent(main, name="box")
    return np.asarray(locator.artist("box").get_xy(), dtype=float)


@pytest.fixture
def closed_figures():
    """Close every figure the test leaves behind.

    Yields:
        None: teardown only. A locator shares its parent's figure, so one leak here would be two axes.
    """
    yield
    plt.close("all")


@pytest.fixture
def framed(closed_figures):
    """Return a factory for a ``Map`` already framed on a region — the map a locator locates.

    Args:
        closed_figures: Teardown fixture, so every map this builds has its figure closed.

    Returns:
        A callable taking ``(crs, bounds)`` and returning the framed
        :class:`~digitalearth.static.map.Map`.
    """

    def make(crs, bounds=PLAIN_BOX):
        main = Map(crs=crs)
        main.set_bounds(list(bounds))
        return main

    return make


@pytest.fixture
def globe(closed_figures):
    """Return a factory for a framed ``globe=True`` map — the map whose locator should be a globe.

    Args:
        closed_figures: Teardown fixture, so every map this builds has its figure closed.

    Returns:
        A callable taking no arguments and returning a framed orthographic
        :class:`~digitalearth.static.map.Map` drawn on a globe frame.
    """

    def make():
        main = Map(crs=projections.orthographic(10.0, 20.0), globe=True)
        main.set_bounds([-1.0e6, -1.0e6, 1.0e6, 1.0e6])
        return main

    return make


class TestMarkExtentChainsLikeEveryOtherBuilder:
    """``mark_extent`` hands back the map, not its patch (round 2, L7).

    It was the tier's **second** artist-returning carve-out, and the censuses say there is one: both
    ``raster.py``'s canonical chaining note and this suite's sibling
    ``tests/static/test_decoration_chaining.py`` name ``stock_img`` as "the one deliberate exception".
    The rationale the return carried — *"the box is one patch a caller restyles afterwards"* — is also
    verbatim the reasoning the PR **rejected** for ``text``/``annotate``, and it bought nothing: the
    patch was already registered as a layer and already reachable by id. So the carve-out is gone rather
    than written into two more places, and the censuses are true as they stand.
    """

    def test_it_hands_back_the_map_it_drew_on(self, framed):
        """The value is the locator itself, like every other builder on the tier.

        Args:
            framed: Factory for the framed main map.

        Test scenario:
            ``Map.inset`` already answers this way and ``tests/test_mixin_contract.py`` holds the tier
            to it; the sibling method on the same mixin answered with a ``Polygon``.
        """
        locator = Map(crs=4326)
        assert locator.mark_extent(framed(4326)) is locator, (
            "mark_extent should hand back the map it drew on"
        )

    def test_it_does_not_break_a_chain_mid_expression(self, framed):
        """Which is what the return convention is *for*.

        Test scenario:
            ``loc.coastlines().mark_extent(detail).set_title("locator")`` raised
            ``AttributeError: 'Polygon' object has no attribute 'set_title'`` — the same shape of break
            round 1's L6 fixed across the decoration methods.
        """
        locator = Map(crs=4326)
        shown = locator.mark_extent(framed(4326)).set_title("locator").ax.get_title()
        assert shown == "locator", (
            f"the chain should carry on through mark_extent; got {shown!r}"
        )

    def test_the_patch_is_reachable_by_the_name_that_was_asked_for(self, framed):
        """The artist is not lost by chaining — it is read back the public way (ST-20's own route).

        Args:
            framed: Factory for the framed main map.

        Test scenario:
            This is the half that makes the carve-out unnecessary: ``_mark`` registers the patch through
            ``add_layer`` before anything is returned, so ``Scene.artist`` has always had it.
        """
        locator = Map(crs=4326)
        locator.mark_extent(framed(4326), name="where")
        assert isinstance(locator.artist("where"), Polygon), (
            f"artist('where') should be the box; got {type(locator.artist('where'))}"
        )

    def test_the_generated_name_reaches_it_too(self, framed):
        """A caller who named nothing still gets at the patch, by the id the call generated.

        Args:
            framed: Factory for the framed main map.

        Test scenario:
            ``name=None`` generates ``custom-1``, which is what the method's docstring promises and the
            only route left once the patch is not the return value.
        """
        locator = Map(crs=4326)
        locator.mark_extent(framed(4326))
        assert isinstance(locator.artist("custom-1"), Polygon), (
            f"the generated id should reach the box; got {locator.layer_ids}"
        )

    def test_a_box_that_could_not_be_placed_has_no_artist_to_hand_back(self, framed):
        """And that is the signal the old ``None`` return carried, in the tier's own words.

        Args:
            framed: Factory for the framed main map.

        Test scenario:
            An extent with no finite image in the locator's CRS draws nothing, so no layer is registered
            and ``Scene.artist`` refuses the id — a refusal a test cannot pass vacuously, unlike reading
            a ``None`` back off the call.
        """
        main = framed(projections.orthographic(0, 0), (2.0e7, 2.0e7, 3.0e7, 3.0e7))
        locator = Map(crs=4326)
        locator.mark_extent(main, name="where")
        with pytest.raises(KeyError, match="no layer 'where'"):
            locator.artist("where")

    def test_chaining_on_from_a_box_that_was_not_drawn_still_works(self, framed):
        """The skip is not a failure, so the expression it sits in carries on.

        Args:
            framed: Factory for the framed main map.

        Test scenario:
            The old ``None`` return turned a skipped box into an ``AttributeError`` one call later,
            which is a worse report than the warning the skip already logs.
        """
        main = framed(projections.orthographic(0, 0), (2.0e7, 2.0e7, 3.0e7, 3.0e7))
        locator = Map(crs=4326)
        shown = locator.mark_extent(main).set_title("no box").ax.get_title()
        assert shown == "no box", (
            f"a skipped box must not break the chain; got {shown!r}"
        )


class TestMarkExtent:
    """Map.mark_extent — one map's extent, drawn on another."""

    def test_an_unframed_map_has_no_extent_to_mark(self, closed_figures):
        """A map still holding matplotlib's unit square is refused by name.

        Args:
            closed_figures: Teardown fixture closing the figures.

        Test scenario:
            An undrawn, unframed axes holds (0, 1) x (0, 1). Marking that would draw a one-by-one box off
            the coast of Africa and call it the map's extent, which is a wrong picture rather than an
            empty one — so it is refused where the mistake is, naming what to do first.
        """
        locator = Map(crs=4326)
        unframed = Map(crs=4326)
        with pytest.raises(ValueError, match="has not been framed"):
            locator.mark_extent(unframed)

    def test_a_map_framed_on_the_unit_square_is_marked_rather_than_refused(
        self, framed
    ):
        """A map whose real extent *is* ``(0, 0, 1, 1)`` has an extent, and it is drawn.

        Args:
            framed: Factory for the framed main map.

        Test scenario:
            One degree square at the prime meridian is a legitimate frame, and it collides exactly with
            the limits an untouched axes holds. The refusal therefore cannot be a coordinate comparison:
            it reads whether the axes was ever framed at all (round 1, M5).
        """
        main = framed(4326, (0.0, 0.0, 1.0, 1.0))
        ring = _marked_ring(Map(crs=4326), main)
        envelope = (
            float(ring[:, 0].min()),
            float(ring[:, 1].min()),
            float(ring[:, 0].max()),
            float(ring[:, 1].max()),
        )
        assert envelope == (0.0, 0.0, 1.0, 1.0), (
            f"a map framed on the unit square should be marked on it; got {envelope}"
        )

    def test_a_map_autoscaled_to_the_unit_square_by_its_data_is_marked(
        self, closed_figures
    ):
        """A raster covering ``(0, 0, 1, 1)`` frames the axes on it without ever setting a limit.

        Args:
            closed_figures: Teardown fixture closing the figures.

        Test scenario:
            The second half of M5's collision, and the one autoscaling alone does not catch: an image
            drawn with ``extent=(0, 1, 0, 1)`` leaves the limits at exactly ``(0, 1)`` *and* leaves
            autoscaling on (sticky edges), so only the presence of an artist separates it from an axes
            nothing has touched.
        """
        main = Map(crs=4326)
        main.ax.imshow(np.arange(4.0).reshape(2, 2), extent=(0.0, 1.0, 0.0, 1.0))
        locator = Map(crs=4326)
        locator.mark_extent(main, name="box")
        assert locator.layer_ids == ["box"], (
            f"a map framed by its own data should have an extent to mark; got {locator.layer_ids}"
        )

    def test_something_that_is_not_a_scene_is_refused(self, closed_figures):
        """An object with no axes cannot have an extent, and says so.

        Args:
            closed_figures: Teardown fixture closing the figures.

        Test scenario:
            The argument is read for ``.ax`` and ``.crs``; a bare tuple of numbers is the plausible
            mistake, and it would otherwise fail several frames in with an AttributeError.
        """
        locator = Map(crs=4326)
        with pytest.raises(TypeError, match="mark_extent"):
            locator.mark_extent((0.0, 0.0, 1.0, 1.0))

    def test_the_box_is_drawn_in_the_locators_own_crs(self, framed):
        """A 3857 extent marked on a 4326 locator lands in degrees, not metres.

        Args:
            framed: Factory for the framed main map.

        Test scenario:
            The expected corner is computed here, straight from pyramids, so the assertion does not read
            the implementation's own answer back to itself.
        """
        expected_x, expected_y = reproject_coordinates(
            [1113194.9], [1118889.97], from_crs=3857, to_crs=4326
        )
        main = framed(3857, (0.0, 0.0, 1113194.9, 1118889.97))
        ring = _marked_ring(Map(crs=4326), main)
        corner = (round(float(ring[:, 0].max()), 4), round(float(ring[:, 1].max()), 4))
        assert corner == (
            round(float(expected_x[0]), 4),
            round(float(expected_y[0]), 4),
        ), f"the box's far corner should be the reprojected one; got {corner}"

    def test_the_edges_are_reprojected_and_not_the_corners(self, framed):
        """A polar box keeps its height, which a corner-only reprojection loses entirely.

        Args:
            framed: Factory for the framed main map.

        Test scenario:
            All four corners of the EPSG:3031 square reproject to one latitude, so the box taken from the
            corners is a flat line. The drawn ring must be taller than that — the lesson recorded in #375.
        """
        _, corner_lats = reproject_coordinates(
            [POLAR_BOX[0], POLAR_BOX[2], POLAR_BOX[2], POLAR_BOX[0]],
            [POLAR_BOX[1], POLAR_BOX[1], POLAR_BOX[3], POLAR_BOX[3]],
            from_crs=3031,
            to_crs=4326,
        )
        corner_span = max(corner_lats) - min(corner_lats)
        main = framed(3031, POLAR_BOX)
        ring = _marked_ring(Map(crs=4326), main)
        drawn_span = float(ring[:, 1].max() - ring[:, 1].min())
        assert drawn_span > corner_span + 1.0, (
            f"the edge-sampled box should be taller than the {corner_span}-degree corner box; "
            f"got {drawn_span}"
        )

    def test_a_same_crs_box_is_the_framed_rectangle(self, framed):
        """With no reprojection to make, the box is exactly the main map's limits.

        Args:
            framed: Factory for the framed main map.

        Test scenario:
            The default locator CRS is the main map's own, which is the case where the mark is a true
            rectangle. Its envelope must be the four numbers the main map was framed on.
        """
        main = framed(4326)
        ring = _marked_ring(Map(crs=4326), main)
        envelope = (
            float(ring[:, 0].min()),
            float(ring[:, 1].min()),
            float(ring[:, 0].max()),
            float(ring[:, 1].max()),
        )
        assert envelope == PLAIN_BOX, (
            f"an unreprojected box should be the framed rectangle; got {envelope}"
        )

    def test_a_box_with_no_image_in_the_locators_crs_marks_nothing(self, framed):
        """An extent the locator's CRS cannot place draws no box.

        Args:
            framed: Factory for the framed main map.

        Test scenario:
            An orthographic map framed far off its own limb reprojects to non-finite coordinates (pyproj
            answers inf rather than raising). There is no outline to draw, and inventing one from the
            finite leftovers would be a wrong picture.
        """
        from digitalearth.static import projections

        main = framed(projections.orthographic(0, 0), (2.0e7, 2.0e7, 3.0e7, 3.0e7))
        locator = Map(crs=4326)
        locator.mark_extent(main, name="box")
        assert locator.layer_ids == [], (
            f"an unplaceable extent should mark nothing; got {locator.layer_ids}"
        )

    def test_the_skip_is_logged(self, framed, caplog):
        """The box that could not be placed is named in a warning rather than dropped silently.

        Args:
            framed: Factory for the framed main map.
            caplog: Captures the warning.

        Test scenario:
            The only other symptom of the skip above is a locator with no box on it, which is
            indistinguishable from the call never having been made.
        """
        from digitalearth.static import projections

        main = framed(projections.orthographic(0, 0), (2.0e7, 2.0e7, 3.0e7, 3.0e7))
        with caplog.at_level("WARNING"):
            Map(crs=4326).mark_extent(main)
        assert "mark_extent" in caplog.text, (
            f"the skip should name the method; log was {caplog.text!r}"
        )

    def test_the_box_is_an_addressable_layer(self, framed):
        """The mark is registered, so it can be hidden or removed like any other layer.

        Args:
            framed: Factory for the framed main map.

        Test scenario:
            A patch put straight on the axes is invisible to the figure. Registering it through the public
            ``add_layer`` is what makes ``set_visible``/``remove_layer`` reach it.
        """
        locator = Map(crs=4326)
        locator.mark_extent(framed(4326), name="where")
        assert locator.get_layer("where").kind == "custom:matplotlib", (
            f"the mark should be a custom layer; got {locator.get_layer('where').kind}"
        )

    def test_the_style_can_be_overridden(self, framed):
        """A keyword lands on the patch rather than being dropped.

        Args:
            framed: Factory for the framed main map.

        Test scenario:
            The defaults are a hollow red outline; a caller re-colouring it for a dark basemap must get
            their own colour, so the override has to reach the artist.
        """
        from matplotlib.colors import to_hex

        locator = Map(crs=4326)
        locator.mark_extent(framed(4326), name="box", edgecolor="#00ff00")
        patch = locator.artist("box")
        assert to_hex(patch.get_edgecolor()) == "#00ff00", (
            f"the caller's edgecolor should be used; got {patch.get_edgecolor()}"
        )

    def test_the_box_is_not_filled_by_default(self, framed):
        """The default mark is an outline, so it does not hide the geography under it.

        Args:
            framed: Factory for the framed main map.

        Test scenario:
            A filled box over a locator's land and coastlines would cover exactly the part of the world
            the reader is being pointed at.
        """
        locator = Map(crs=4326)
        locator.mark_extent(framed(4326), name="box")
        patch = locator.artist("box")
        assert patch.get_facecolor()[3] == 0.0, (
            f"the box should be unfilled; got facecolor {patch.get_facecolor()}"
        )


class TestInset:
    """Map.inset — the locator map as one call."""

    def test_the_inset_is_a_child_of_the_main_axes(self, framed):
        """``inset()`` draws into an axes inside the main one.

        Args:
            framed: Factory for the framed main map.

        Test scenario:
            ``ax.inset_axes`` registers a child axes rather than a figure-level one, which is what keeps
            the locator moving with its map.
        """
        main = framed(4326)
        assert main.inset().locator.ax in main.ax.child_axes, (
            "the locator's axes should be a child of the main axes"
        )

    def test_the_locator_shares_the_parent_figure(self, framed):
        """One figure holds both the map and its locator.

        Args:
            framed: Factory for the framed main map.

        Test scenario:
            The locator is a borrowed-axes Scene, so a single ``savefig`` has to produce both.
        """
        main = framed(4326)
        assert main.inset().locator.fig is main.fig, (
            "the locator should draw into the parent's figure"
        )

    def test_the_locator_defaults_to_the_main_maps_crs(self, framed):
        """With no ``crs`` asked for, the locator is in the map's own display CRS.

        Args:
            framed: Factory for the framed main map.

        Test scenario:
            The default is the one case where the extent box needs no reprojection at all, so it is a
            true rectangle; a different locator CRS is opt-in.
        """
        main = framed(3857, (0.0, 0.0, 1113194.9, 1118889.97))
        assert main.inset().locator.crs == 3857, (
            "the locator should inherit the display CRS"
        )

    def test_an_asked_for_crs_is_used(self, framed):
        """``crs=`` puts the locator in a projection of its own.

        Args:
            framed: Factory for the framed main map.

        Test scenario:
            A Web Mercator city map located on a lon/lat world map is the normal pairing, and it is the
            path that reprojects the extent box.
        """
        main = framed(3857, (0.0, 0.0, 1113194.9, 1118889.97))
        assert main.inset(crs=4326).locator.crs == 4326, (
            "the asked-for locator CRS should win"
        )

    def test_the_default_locator_shows_the_whole_world(self, framed):
        """Without an ``extent`` the locator is framed on the projection's full domain.

        Args:
            framed: Factory for the framed main map.

        Test scenario:
            A locator map answers "where on the wider area is this", so the wider area defaults to all of
            it. In EPSG:4326 that is the whole lon/lat rectangle.
        """
        locator = framed(4326).inset().locator
        limits = [float(v) for v in (*locator.ax.get_xlim(), *locator.ax.get_ylim())]
        assert limits == [-180.0, 180.0, -90.0, 90.0], (
            f"the default locator should be global; got {limits}"
        )

    def test_an_extent_frames_the_locator_on_it(self, framed):
        """``extent=`` narrows the locator to a region of its own.

        Args:
            framed: Factory for the framed main map.

        Test scenario:
            A continent is more use than a globe for a city map, so the wider area is a caller's choice.
        """
        locator = framed(4326).inset(extent=[-20.0, 30.0, 40.0, 70.0]).locator
        limits = [float(v) for v in (*locator.ax.get_xlim(), *locator.ax.get_ylim())]
        assert limits == [-20.0, 40.0, 30.0, 70.0], (
            f"the locator should hold the asked-for extent; got {limits}"
        )

    def test_the_locator_draws_reference_geography_and_the_box(self, framed):
        """The default locator carries land, coastlines and the extent mark.

        Args:
            framed: Factory for the framed main map.

        Test scenario:
            A locator with no geography on it locates nothing, and one with no box does not say where.
            All three layers are described, so the figure names them.
        """
        locator = framed(4326).inset().locator
        kinds = [locator.get_layer(name).kind for name in locator.layer_ids]
        assert kinds == ["land", "coastlines", "custom:matplotlib"], (
            f"expected geography then the mark; got {kinds}"
        )

    def test_the_reference_geography_is_the_callers_choice(self, framed):
        """``reference=`` picks which Natural-Earth layers go in.

        Args:
            framed: Factory for the framed main map.

        Test scenario:
            An ocean-and-borders locator reads better over a dark basemap, so the list is not fixed.
        """
        locator = framed(4326).inset(reference=("ocean",)).locator
        assert locator.layer_ids[0] == "ocean-1", (
            f"the asked-for layer should be the first drawn; got {locator.layer_ids}"
        )

    def test_no_reference_geography_leaves_only_the_box(self, framed):
        """``reference=()`` draws the mark alone, for a caller supplying their own geography.

        Args:
            framed: Factory for the framed main map.

        Test scenario:
            The locator is returned so more can be drawn on it; asking for none of the defaults must be
            possible rather than something to undo afterwards.
        """
        locator = framed(4326).inset(reference=()).locator
        assert len(locator.layer_ids) == 1, (
            f"only the extent box should be on it; got {locator.layer_ids}"
        )

    def test_an_unknown_reference_layer_is_refused(self, framed):
        """A name that is not a reference layer is refused rather than looked up on the Map.

        Args:
            framed: Factory for the framed main map.

        Test scenario:
            The names are resolved to methods, so an unchecked one would reach any attribute at all —
            ``reference=("close",)`` would close the figure. The list is a whitelist for that reason.
        """
        main = framed(4326)
        with pytest.raises(ValueError, match=r"inset\(reference="):
            main.inset(reference=("close",))

    @pytest.mark.parametrize("corner", ["upper right", "upper left", "lower left"])
    def test_each_corner_is_placed_where_it_is_named(self, corner, framed):
        """The named corners put the inset in the matching corner of the main axes.

        Args:
            corner: The position asked for.
            framed: Factory for the framed main map.

        Test scenario:
            The placement is axes-fraction arithmetic, so it is checked by *side* rather than by
            re-deriving the same numbers: an "upper" inset's bottom edge is above the middle, a "right"
            one's left edge is right of it.
        """
        main = framed(4326)
        child = main.inset(position=corner).locator.ax.get_position()
        parent = main.ax.get_position()
        above = (child.y0 - parent.y0) / parent.height > 0.5
        right = (child.x0 - parent.x0) / parent.width > 0.5
        assert (above, right) == ("upper" in corner, "right" in corner), (
            f"{corner} landed at above={above}, right={right}"
        )

    def test_an_explicit_bounds_sequence_is_passed_through(self, framed):
        """A four-number ``position`` is the axes-fraction rectangle, used as given.

        Args:
            framed: Factory for the framed main map.

        Test scenario:
            The escape hatch for a layout the four corners do not cover. The inset's position is read
            back as a fraction of the parent axes, which is what was asked for.
        """
        main = framed(4326)
        child = main.inset(position=(0.1, 0.2, 0.3, 0.4)).locator.ax.get_position()
        parent = main.ax.get_position()
        placed = (
            round((child.x0 - parent.x0) / parent.width, 3),
            round((child.y0 - parent.y0) / parent.height, 3),
        )
        assert placed == (0.1, 0.2), (
            f"the rectangle should be used as given; got {placed}"
        )

    def test_an_unknown_position_is_refused(self, framed):
        """A word that is not a corner is refused, listing the ones that are.

        Args:
            framed: Factory for the framed main map.

        Test scenario:
            ``position="top right"`` is the plausible near-miss, and placing it somewhere by default
            would be a silently wrong layout.
        """
        main = framed(4326)
        with pytest.raises(ValueError, match=r"inset\(position="):
            main.inset(position="top right")

    @pytest.mark.parametrize("bad", [0.0, 1.5, -0.2])
    def test_a_size_that_is_not_a_fraction_of_the_axes_is_refused(self, bad, framed):
        """``size`` is a fraction in (0, 1]; anything else is refused.

        Args:
            bad: A size outside the range.
            framed: Factory for the framed main map.

        Test scenario:
            A zero-size inset draws nothing and a size above one hangs outside the map, neither of which
            matplotlib refuses on its own.
        """
        main = framed(4326)
        with pytest.raises(ValueError, match=r"inset\(size="):
            main.inset(size=bad)

    @pytest.mark.parametrize("too_big", [0.95, 1.0])
    def test_a_size_that_cannot_keep_its_pad_in_a_corner_is_refused(
        self, too_big, framed
    ):
        """A corner keeps a pad on both sides, so the side it can hold is at most ``0.94``.

        Args:
            too_big: A size inside ``(0, 1]`` that still does not fit in a padded corner.
            framed: Factory for the framed main map.

        Test scenario:
            ``size=1.0`` passed the documented ``(0, 1]`` range and placed the inset at ``x0 = -0.03``,
            outside the map, because the corner arithmetic subtracts the 0.03 pad twice (round 1, L1).
            The range the corners can honour is the one they are now held to.
        """
        main = framed(4326)
        with pytest.raises(ValueError, match=r"inset\(size="):
            main.inset(size=too_big)

    @pytest.mark.parametrize("corner", sorted(_CORNERS))
    def test_the_largest_corner_inset_stays_inside_the_map(self, corner, framed):
        """At the top of its range a corner inset is still wholly inside the axes it sits in.

        Args:
            corner: The position asked for.
            framed: Factory for the framed main map.

        Test scenario:
            Measured as a fraction of the parent axes rather than re-derived: the inset's near edge is
            at or inside 0 and its far edge at or inside 1, for every corner, at the largest size the
            corner path accepts. The two edges are asserted apart because they fail apart — raising the
            limit past 0.94 pushes a near-anchored corner's *far* edge out and a far-anchored corner's
            *near* edge out, and a single composite assertion could not say which.
        """
        main = framed(4326)
        child = main.inset(
            position=corner, size=LARGEST_CORNER
        ).locator.ax.get_position()
        parent = main.ax.get_position()
        near = (
            round((child.x0 - parent.x0) / parent.width, 6),
            round((child.y0 - parent.y0) / parent.height, 6),
        )
        far = (
            round((child.x1 - parent.x0) / parent.width, 6),
            round((child.y1 - parent.y0) / parent.height, 6),
        )
        assert min(near) >= 0.0, (
            f"{corner} at size={LARGEST_CORNER} ran out past the parent's near edge: "
            f"near={near} (far={far})"
        )
        assert max(far) <= 1.0, (
            f"{corner} at size={LARGEST_CORNER} ran out past the parent's far edge: "
            f"far={far} (near={near})"
        )

    def test_the_locators_ticks_are_off(self, framed):
        """A locator map carries no tick labels — it is a thumbnail, not a plot.

        Args:
            framed: Factory for the framed main map.

        Test scenario:
            At a quarter of the map's size the tick labels overlap into illegibility, so they are off by
            construction.
        """
        locator = framed(4326).inset().locator
        ticks = (len(locator.ax.get_xticks()), len(locator.ax.get_yticks()))
        assert ticks == (0, 0), f"the locator should carry no ticks; got {ticks}"

    def test_an_unframed_map_is_refused_before_an_axes_is_made(self, closed_figures):
        """``inset()`` on an unframed map is refused, like ``mark_extent`` is.

        Args:
            closed_figures: Teardown fixture closing the figures.

        Test scenario:
            A locator exists to show where the map is looking; a map not yet looking anywhere has nothing
            to locate, and marking its unit square would be a wrong picture.
        """
        main = Map(crs=4326)
        with pytest.raises(ValueError, match="has not been framed"):
            main.inset()

    def test_a_refused_inset_leaves_no_axes_behind(self, closed_figures):
        """The refusal above happens before the inset axes is created.

        Args:
            closed_figures: Teardown fixture closing the figures.

        Test scenario:
            A half-built locator — an empty inset axes on the map — would survive the exception and be
            saved into the figure, so the extent is read before anything is drawn.
        """
        main = Map(crs=4326)
        with pytest.raises(ValueError, match="has not been framed"):
            main.inset()
        assert main.ax.child_axes == [], (
            f"a refused inset should leave no child axes; got {main.ax.child_axes}"
        )

    def test_closing_the_locator_spares_the_parents_figure(self, framed):
        """``Scene.close`` on the locator leaves the parent's figure open (#371, resolved) — measured.

        Args:
            framed: Factory for the framed main map.

        Test scenario:
            The locator borrows the parent's figure, so `close` (which gates `pyplot.close` on
            `_owns_fig`) spares it: closing the locator does not take the map with it. A caller may now
            close a locator without destroying the figure it sits in.
        """
        main = framed(4326)
        main.inset().locator.close()
        assert main.fig.number in plt.get_fignums(), (
            "closing the locator should spare the parent figure it borrows (#371)"
        )

    def test_the_locator_leaves_the_figure_open_until_closed(self, framed):
        """Building a locator does not close anything by itself.

        Args:
            framed: Factory for the framed main map.

        Test scenario:
            The other half of the claim above: the hazard is ``close``, not the inset — a map with a
            locator on it is still there to be saved.
        """
        main = framed(4326)
        main.inset()
        assert main.fig.number in plt.get_fignums(), (
            "the figure should still be open after adding a locator"
        )


class TestWhatTheLocatorIsBeforeAndAfter:
    """``Map.locator`` — the one handle on an inset, and what it answers before there is one."""

    def test_a_map_with_no_inset_holds_no_locator(self, closed_figures):
        """Reading the handle on a map nobody called ``inset`` on answers ``None``.

        Args:
            closed_figures: Teardown fixture closing the figures.

        Test scenario:
            ``inset`` hands back the *main* map so it chains like every other builder, which makes
            ``.locator`` the only way to reach the inset — so its unset value has to be a readable
            ``None`` rather than an ``AttributeError``.
        """
        assert Map(crs=4326).locator is None, (
            "a map with no inset must report no locator"
        )

    def test_the_fourth_corner_is_placed_where_it_is_named(self, framed):
        """``"lower right"`` — the corner the parametrised case above leaves out.

        Args:
            framed: Factory for the framed main map.

        Test scenario:
            The three tested corners between them exercise ``right=1`` and ``upper=1`` but never the
            ``(1, 0)`` pair, so a transposed lookup in ``_CORNERS`` would pass them all. Checked by side,
            like its siblings: a "lower right" inset's bottom edge is below the middle and its left edge
            is right of it.
        """
        main = framed(4326)
        child = main.inset(position="lower right").locator.ax.get_position()
        parent = main.ax.get_position()
        above = (child.y0 - parent.y0) / parent.height > 0.5
        right = (child.x0 - parent.x0) / parent.width > 0.5
        assert (above, right) == (False, True), (
            f"'lower right' landed at above={above}, right={right}"
        )


class TestARefusedInsetNamesWhatWasWrong:
    """The refusals whose *message* is the only thing a caller can act on."""

    def test_the_not_framed_refusal_names_the_call_that_was_made(self, closed_figures):
        """``inset()`` on an unframed map says ``inset()``, not ``mark_extent()``.

        Args:
            closed_figures: Teardown fixture closing the figures.

        Test scenario:
            Both calls read the extent through the same value object, which used to hardcode the other
            call's name — so a caller who never typed ``mark_extent`` was told to look at it (round 1,
            L2). The message is the only thing they can act on, so it names what they wrote.
        """
        main = Map(crs=4326)
        with pytest.raises(
            ValueError, match=r"^inset\(\): the map has not been framed"
        ):
            main.inset()

    @pytest.mark.parametrize("bad", [(0.1, 0.2), 42, "0.1 0.2 0.3 0.4"])
    def test_a_position_that_is_neither_a_corner_nor_four_fractions_is_refused(
        self, bad, framed
    ):
        """Anything that does not unpack into four numbers is refused by name.

        Args:
            bad: A ``position`` that is not a corner and does not give four fractions — too short, not a
                sequence at all, and a string that looks like one but is not a corner name.
            framed: Factory for the framed main map.

        Test scenario:
            The corner names are checked against a whitelist, so the remaining job is the escape hatch:
            a two-number tuple and a bare number both reach the unpacking, and letting either through
            would raise from inside matplotlib's ``inset_axes`` with no mention of ``position``.
        """
        main = framed(4326)
        with pytest.raises(ValueError, match=r"inset\(position="):
            main.inset(position=bad)

    @pytest.mark.parametrize("bad", [True, "big", None])
    def test_a_size_that_is_not_a_number_at_all_is_refused(self, bad, framed):
        """``size`` must be a real number, and ``True`` is not one of them.

        Args:
            bad: A ``size`` that is not a usable fraction — ``True`` (a ``bool`` is an ``int``, so it
                passes a numeric check and means nothing here), a word, and ``None``.
            framed: Factory for the framed main map.

        Test scenario:
            The numeric cases (0.0, 1.5, -0.2) are covered above; these are the ones a range check alone
            would let through. ``size=True`` is the one that matters: ``isinstance(True, Real)`` is
            ``True`` and ``float(True)`` is ``1.0``, so without the explicit ``bool`` rejection it would
            be read as a request for a full-size inset.
        """
        main = framed(4326)
        with pytest.raises(ValueError, match=r"inset\(size="):
            main.inset(size=bad)


class TestAnExplicitRectangleIsAPlacementOrARefusal:
    """``inset(position=(x0, y0, w, h))`` is checked, not taken on trust (round 2, L10).

    Round 1's L1 bounded the **named corner** at :data:`LARGEST_CORNER`, and the refusal it added ends
    *"or place the inset yourself with explicit fractions"* — steering the caller onto the sibling
    spelling, which took any four numbers. So the escape hatch the refusal recommends is how a caller
    lands outside the axes, with a zero-sized locator, or with a NaN one.

    The split the checks draw: a rectangle that **overlaps** the axes but hangs over its edge is a
    deliberate placement and only warned about — that is what the escape hatch is for, and matplotlib
    allows it. A rectangle that could not be a placement at all is refused by name: a non-finite edge, a
    side of zero or less, a side too small to survive the thousandth-of-the-axes precision
    ``_InsetFrame.as_bounds`` reports, and a rectangle entirely off the axes.
    """

    #: Rectangles that describe no inset. Each is ``(position, the word the refusal must carry)``.
    IMPOSSIBLE = [
        pytest.param((0.6, 0.6, 0.0, 0.3), "zero", id="zero-width"),
        pytest.param((0.6, 0.6, 0.3, 0.0), "zero", id="zero-height"),
        pytest.param((0.6, 0.6, -0.3, 0.3), "zero", id="negative-width"),
        pytest.param((0.6, 0.6, 0.3, -0.3), "zero", id="negative-height"),
        pytest.param((0.6, 0.6, 0.0001, 0.3), "thousandth", id="side-rounds-away"),
        pytest.param((float("nan"), 0.0, 0.3, 0.3), "finite", id="nan-x0"),
        pytest.param((0.0, 0.0, float("nan"), 0.3), "finite", id="nan-width"),
        pytest.param((float("inf"), 0.0, 0.3, 0.3), "finite", id="inf-x0"),
        pytest.param((0.0, -0.5, 0.3, 0.3), "outside", id="wholly-below"),
        pytest.param((-0.4, 0.0, 0.3, 0.3), "outside", id="wholly-left"),
        pytest.param((1.0, 0.0, 0.3, 0.3), "outside", id="starts-at-right-edge"),
        pytest.param((0.0, 1.2, 0.3, 0.3), "outside", id="wholly-above"),
    ]

    @pytest.mark.parametrize("position, word", IMPOSSIBLE)
    def test_a_rectangle_that_is_no_placement_is_refused_naming_the_call(
        self, position, word, framed
    ):
        """Every one of them says ``inset(position=...)`` and what is wrong with it.

        Args:
            position: The rectangle asked for.
            word: The word the message has to carry, so the twelve cases are not all proved by one
                generic sentence.
            framed: Factory for the framed main map.

        Test scenario:
            Three of these were accepted and drawn (the zero sides, and the rectangles off the axes),
            three more were accepted as NaN geometry, and the negative side was refused by *matplotlib*
            with "Width and height specified must be non-negative" — a message that mentions neither
            ``inset`` nor ``position``, against round 1's L2.
        """
        main = framed(4326)
        with pytest.raises(ValueError, match=r"inset\(position=") as refusal:
            main.inset(position=position)
        assert word in str(refusal.value), (
            f"the refusal should say what is wrong ({word!r}); got {refusal.value}"
        )

    @pytest.mark.parametrize("position, word", IMPOSSIBLE)
    def test_a_refused_rectangle_leaves_no_axes_behind(self, position, word, framed):
        """The check runs before ``ax.inset_axes``, so a refused call draws nothing.

        Args:
            position: The rectangle asked for.
            word: Unused here; the parametrisation is shared with the refusal above.
            framed: Factory for the framed main map.

        Test scenario:
            ``inset`` answers everything a caller can get wrong first, for exactly this reason — a
            half-built locator on the figure is worse than the refusal.
        """
        main = framed(4326)
        with pytest.raises(ValueError, match=r"inset\(position="):
            main.inset(position=position)
        assert main.ax.child_axes == [], (
            f"a refused rectangle should leave no inset axes; got {main.ax.child_axes}"
        )

    def test_a_rectangle_hanging_over_the_edge_is_warned_about_not_refused(
        self, framed, caplog
    ):
        """The escape hatch stays open: an overlapping rectangle is a placement, and it is honoured.

        Args:
            framed: Factory for the framed main map.
            caplog: Captures the warning.

        Test scenario:
            ``(0.9, 0.9, 0.5, 0.5)`` puts 60% of the inset past the axes' top-right corner. Refusing it
            would close the spelling the corner refusal points callers at; saying nothing is how the
            review found it. So it draws, and says which edges it crosses.
        """
        main = framed(4326)
        with caplog.at_level("WARNING", logger="digitalearth.static.maps.inset"):
            main.inset(position=(0.9, 0.9, 0.5, 0.5), reference=())
        assert "outside" in caplog.text, (
            f"an overhanging rectangle should be reported; log was {caplog.text!r}"
        )

    def test_the_overhanging_rectangle_is_still_the_one_that_was_asked_for(
        self, framed
    ):
        """Warned about, not adjusted — the fractions reach ``ax.inset_axes`` as written.

        Args:
            framed: Factory for the framed main map.

        Test scenario:
            Clamping it into the axes would be a third behaviour, and a caller placing a locator
            deliberately off the edge would silently get a different figure.
        """
        main = framed(4326)
        main.inset(position=(0.9, 0.9, 0.5, 0.5), reference=())
        box = main.locator.ax.get_position()
        parent = main.ax.get_position()
        width = round((box.x1 - box.x0) / parent.width, 3)
        assert width == 0.5, (
            f"the inset should keep the width that was asked for; got {width}"
        )

    def test_a_contained_rectangle_says_nothing(self, framed, caplog):
        """The ordinary case must stay quiet, or the warning trains the caller to ignore it.

        Args:
            framed: Factory for the framed main map.
            caplog: Captures anything logged.

        Test scenario:
            ``(0.6, 0.6, 0.3, 0.3)`` sits wholly inside the axes, which is what the spelling is normally
            used for.
        """
        main = framed(4326)
        with caplog.at_level("WARNING", logger="digitalearth.static.maps.inset"):
            main.inset(position=(0.6, 0.6, 0.3, 0.3), reference=())
        assert caplog.text == "", (
            f"a contained rectangle has nothing to report; log was {caplog.text!r}"
        )

    def test_the_rectangle_flush_with_the_axes_is_contained(self, framed, caplog):
        """``(0, 0, 1, 1)`` is the whole axes exactly, and is neither refused nor warned about.

        Args:
            framed: Factory for the framed main map.
            caplog: Captures anything logged.

        Test scenario:
            The boundary of the containment test. A strict comparison would call the flush rectangle an
            overhang and warn about the one placement that fills the map exactly.
        """
        main = framed(4326)
        with caplog.at_level("WARNING", logger="digitalearth.static.maps.inset"):
            main.inset(position=(0.0, 0.0, 1.0, 1.0), reference=())
        assert caplog.text == "", (
            f"the flush rectangle is contained; log was {caplog.text!r}"
        )

    def test_a_size_too_small_to_survive_the_rounding_is_refused_too(self, framed):
        """The sibling spelling has the same zero-area hole, through the rounding rather than the input.

        Args:
            framed: Factory for the framed main map.

        Test scenario:
            ``size=0.0001`` is a fraction in ``(0, 1]``, so the range check passes — and
            ``_InsetFrame.as_bounds`` then rounds the side to ``0.0``, which is the invisible locator the
            explicit rectangle is refused for. Both spellings are answered, or the refusal is only about
            how the inset was spelled.
        """
        main = framed(4326)
        with pytest.raises(ValueError, match=r"inset\(size=0.0001\)"):
            main.inset(size=0.0001)


class TestTheInsetRaisesBlockCannotGoStale:
    """``inset()``'s documented refusals, as a census rather than as a count (round 2, L13).

    The docstring said *"Each of the four names `inset()`"* and then listed five conditions — the count
    went stale when ``8d4f6a50`` added the ``globe=True`` beside ``extent`` refusal, and round 2's L10
    added more. A count in front of a list is a second place to keep in step with the list, so there is
    no count any more: the claim is *"every one of them"*, and this suite is the census that makes it
    checkable.
    """

    #: Every way ``inset()`` refuses a call, as ``(kwargs, the keyword the message must name)``. A
    #: condition added to :meth:`~digitalearth.static.map.Map.inset` and not to this list leaves the
    #: docstring's "every one of them" unproven, which is the half a count could not express.
    REFUSALS = [
        pytest.param({"size": 1.5}, "size", id="size-above-one"),
        pytest.param({"size": 0.0}, "size", id="size-zero"),
        pytest.param({"size": "big"}, "size", id="size-not-a-number"),
        pytest.param({"size": 0.0001}, "size", id="size-rounds-away"),
        pytest.param({"size": 0.95}, "size", id="size-too-big-for-a-corner"),
        pytest.param({"position": "middle"}, "position", id="position-not-a-corner"),
        pytest.param({"position": (0.1, 0.2)}, "position", id="position-too-short"),
        pytest.param(
            {"position": (0.6, 0.6, 0.0, 0.3)}, "position", id="rectangle-zero-side"
        ),
        pytest.param(
            {"position": (float("nan"), 0.0, 0.3, 0.3)},
            "position",
            id="rectangle-not-finite",
        ),
        pytest.param(
            {"position": (0.0, -0.5, 0.3, 0.3)}, "position", id="rectangle-off-the-map"
        ),
        pytest.param({"reference": ("close",)}, "reference", id="reference-unknown"),
        pytest.param(
            {"globe": True, "extent": [-4.0e6, -4.0e6, 4.0e6, 4.0e6]},
            "globe",
            id="globe-beside-an-extent",
        ),
        pytest.param(
            {"extent": [60.0, 40.0, -20.0, -40.0]}, "extent", id="extent-inverted"
        ),
        pytest.param({"extent": [0.0, 0.0, 0.0, 0.0]}, "extent", id="extent-zero-area"),
        pytest.param({"extent": [1.0, 2.0, 3.0]}, "extent", id="extent-too-short"),
        pytest.param(
            {"extent": [0.0, 0.0, float("nan"), 1.0]}, "extent", id="extent-not-finite"
        ),
        pytest.param(
            {"extent": ["west", "south", "east", "north"]},
            "extent",
            id="extent-not-numbers",
        ),
    ]

    @pytest.mark.parametrize("kwargs, keyword", REFUSALS)
    def test_every_refusal_names_the_call_and_the_keyword(
        self, kwargs, keyword, framed
    ):
        """Which is what the docstring claims of all of them, counted or not.

        Args:
            kwargs: The call that is refused.
            keyword: The keyword the message has to name, so one generic sentence cannot prove the lot.
            framed: Factory for the framed main map.

        Test scenario:
            Round 1's L2 is the rule being held to — a refusal names the call that was made. The census
            is here rather than in the docstring's prose because prose cannot be run.
        """
        main = framed(4326)
        with pytest.raises(ValueError, match=rf"inset\({keyword}") as refusal:
            main.inset(**kwargs)
        assert "inset(" in str(refusal.value), (
            f"the refusal should name the call; got {refusal.value}"
        )

    def test_the_unframed_refusal_is_the_thirteenth_and_names_the_call_too(
        self, closed_figures
    ):
        """The one condition that is not a keyword's — it is the map's state.

        Args:
            closed_figures: Teardown fixture closing the figures.

        Test scenario:
            It is listed apart from the rest because there is no keyword to quote, and it is the refusal
            round 1's L2 was actually about.
        """
        main = Map(crs=4326)
        with pytest.raises(
            ValueError, match=r"^inset\(\): the map has not been framed"
        ):
            main.inset()

    def test_the_raises_block_carries_no_count_to_go_stale(self):
        """A numeral in front of the list is the thing that went stale, so there must not be one.

        Test scenario:
            "Each of the four" survived the commit that added a fifth refusal because nothing read it.
            This reads it: any "each of the <n>" phrasing fails here, whatever the number, so the fix
            cannot be re-broken by adding a refusal and updating nothing.
        """
        counted = re.search(
            r"(?:each|all|any) of the (?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|"
            r"eleven|twelve|thirteen)\b",
            Map.inset.__doc__,
            re.IGNORECASE,
        )
        assert counted is None, (
            f"the Raises block should not count its refusals; found {counted and counted.group(0)!r}"
        )

    def test_the_raises_block_still_claims_every_refusal_names_the_call(self):
        """Dropping the count must not drop the claim it was attached to.

        Test scenario:
            The count was wrong; the promise around it was right and is what round 1's L2 bought. Losing
            it would make the next sweep re-derive the rule from the code.
        """
        assert "every one of them names `inset()`" in Map.inset.__doc__, (
            "the Raises block should still promise that each refusal names the call"
        )


class TestALocatorOnAGlobe:
    """A globe's locator is a globe, so its geography is limb-clipped and its frame is drawn (M6).

    Measured on the parent commit, ``Map(crs=orthographic(10, 20), globe=True).inset()``: the locator came
    back with ``globe=False``, its land layer drawn through cleopatra's flat ``add_features`` and no
    boundary patch on the axes. The claims here read the **artist** each layer owns, because that is what
    the two paths differ in: the globe fill answers a ``PolyCollection`` and the flat one answers the axes
    it drew on.
    """

    def test_a_globe_maps_locator_is_a_globe_as_well(self, globe):
        """The frame is inherited, so the locator renders the way the map it sits in does.

        Args:
            globe: Factory for the framed globe map.

        Test scenario:
            ``inset()`` built the locator with ``crs=``, ``ax=`` and ``fig=`` and nothing else, so a globe
            map's locator was a flat map carrying a globe CRS.
        """
        locator = globe().inset().locator
        assert locator.globe is True, (
            f"a globe's locator should be a globe; got globe={locator.globe!r}"
        )

    def test_the_locators_land_is_filled_through_the_globe_path(self, globe):
        """The reference geography is re-closed at the limb rather than drawn flat.

        Args:
            globe: Factory for the framed globe map.

        Test scenario:
            The two paths are told apart by what they hand back:
            ``DecorationMixin._fill_globe_polygons`` owns a ``PolyCollection`` of limb-clipped rings,
            while the flat path lets cleopatra draw onto the axes and so owns the ``Axes``.
        """
        locator = globe().inset().locator
        drawn = type(locator.artist("land-1")).__name__
        assert drawn == "PolyCollection", (
            f"a globe locator's land should be limb-clipped rings; got {drawn}"
        )

    def test_the_locator_carries_the_projection_boundary(self, globe):
        """The limb is drawn on the locator too, which is what makes it read as a globe.

        Args:
            globe: Factory for the framed globe map.

        Test scenario:
            Nothing else ever renders the locator — ``render``/``save``/``show`` are called on the *map*,
            and the locator is a separate scene — so the frame has to go on as the locator is built.
        """
        locator = globe().inset().locator
        assert any(isinstance(patch, PathPatch) for patch in locator.ax.patches), (
            f"the frame should put a boundary on the locator; patches are {locator.ax.patches}"
        )

    def test_a_caller_can_ask_for_a_flat_locator_on_a_globe(self, globe):
        """``globe=False`` is the way out, since the default is now to inherit.

        Args:
            globe: Factory for the framed globe map.

        Test scenario:
            A flat locator in a globe CRS is a legitimate picture — the whole disc on a plain axes — and
            it is what every caller got before, so it stays reachable by name.
        """
        locator = globe().inset(globe=False).locator
        assert locator.globe is False, (
            f"globe=False should be honoured; got globe={locator.globe!r}"
        )

    def test_a_flat_map_can_still_ask_its_locator_for_a_globe(self, framed):
        """The keyword works the other way round as well, from a flat map.

        Args:
            framed: Factory for the framed main map.

        Test scenario:
            The default follows the map, and the keyword overrides it in both directions rather than only
            switching the inheritance off.
        """
        main = framed(
            projections.orthographic(10.0, 20.0), (-1.0e6, -1.0e6, 1.0e6, 1.0e6)
        )
        locator = main.inset(globe=True).locator
        assert locator.globe is True, (
            f"globe=True should be honoured on a flat map; got globe={locator.globe!r}"
        )

    def test_a_flat_maps_locator_is_still_flat(self, framed):
        """Nothing changes for the maps that are not globes.

        Args:
            framed: Factory for the framed main map.

        Test scenario:
            The inheritance is from the map, so a flat map's locator keeps the flat render every existing
            caller and baseline has.
        """
        locator = framed(4326).inset().locator
        assert locator.globe is False, (
            f"a flat map's locator should stay flat; got globe={locator.globe!r}"
        )

    def test_an_extent_makes_the_locator_flat_even_on_a_globe(self, globe):
        """A caller who frames the locator has asked for the frame that can hold a frame.

        Args:
            globe: Factory for the framed globe map.

        Test scenario:
            ``apply_projection_frame`` sets the projection's own x/y limits, so a globe locator cannot
            honour an ``extent`` at all — measured: a globe locator asked for
            ``[-4e6, -4e6, 4e6, 4e6]`` came back holding ``[-6378058, 6378064]``. Naming an extent is
            therefore the caller saying otherwise, and the locator is drawn flat on it.
        """
        locator = globe().inset(extent=[-4.0e6, -4.0e6, 4.0e6, 4.0e6]).locator
        held = [round(float(value)) for value in locator.ax.get_ylim()]
        assert held == [-4000000, 4000000], (
            f"the extent should be the locator's frame; got {held}"
        )

    def test_a_globe_locator_framed_on_an_extent_is_refused(self, globe):
        """Asking for both is a contradiction, and it is named rather than half-honoured.

        Args:
            globe: Factory for the framed globe map.

        Test scenario:
            The globe frame would throw the extent away silently, which is the shape of defect this row
            exists to end — so the two arguments together are refused.
        """
        main = globe()
        with pytest.raises(ValueError, match=r"inset\(globe=True, extent="):
            main.inset(globe=True, extent=[-4.0e6, -4.0e6, 4.0e6, 4.0e6])

    def test_that_refusal_leaves_no_axes_behind(self, globe):
        """It lands with the other front-of-method refusals, before the inset axes exists.

        Args:
            globe: Factory for the framed globe map.

        Test scenario:
            The same contract the size/position/reference refusals keep: nothing half-built is left on
            the figure.
        """
        main = globe()
        with pytest.raises(ValueError, match=r"inset\(globe=True, extent="):
            main.inset(globe=True, extent=[-4.0e6, -4.0e6, 4.0e6, 4.0e6])
        assert main.ax.child_axes == [], (
            f"a refused inset should leave no axes; got {main.ax.child_axes}"
        )


class TestAnExtentPROJCannotTransformMarksNothing:
    """The other half of "no box rather than a wrong one": a transform that *raises*."""

    def test_a_transform_that_raises_marks_nothing(self, framed, mocker):
        """A PROJ error while placing the box is answered with no box.

        Args:
            framed: Factory for the framed main map.
            mocker: Patches the reprojection the placement goes through.

        Test scenario:
            The non-finite case above is pyproj answering ``inf``; this is pyproj *refusing* — a CRS it
            cannot resolve, or a transform it rejects. Both have to end in no box, and only the
            non-finite one happens on data this suite can build, so the refusal is injected.
        """
        main = framed(4326)
        mocker.patch(
            "digitalearth.static.maps.inset.reproject_coordinates",
            side_effect=RuntimeError("PROJ refused the transform"),
        )
        locator = Map(crs=3857)
        locator.mark_extent(main, name="box")
        assert locator.layer_ids == [], (
            f"a refused transform should mark nothing; got {locator.layer_ids}"
        )

    def test_the_refusal_is_logged(self, framed, mocker, caplog):
        """And it is named in a warning, not dropped silently.

        Args:
            framed: Factory for the framed main map.
            mocker: Patches the reprojection the placement goes through.
            caplog: Captures the warning.

        Test scenario:
            Same reason as the non-finite skip: a locator with no box on it looks exactly like a
            ``mark_extent`` that was never called.
        """
        main = framed(4326)
        mocker.patch(
            "digitalearth.static.maps.inset.reproject_coordinates",
            side_effect=ValueError("unknown CRS"),
        )
        locator = Map(crs=3857)
        with caplog.at_level("WARNING"):
            locator.mark_extent(main)
        assert "no finite image" in caplog.text, (
            f"the skip should say the extent has no image; log was {caplog.text!r}"
        )

    def test_nothing_is_registered_for_a_box_that_was_not_drawn(self, framed, mocker):
        """A skipped box adds no layer either, so the figure does not describe one.

        Args:
            framed: Factory for the framed main map.
            mocker: Patches the reprojection the placement goes through.

        Test scenario:
            ``_mark`` registers the patch through ``add_layer``, which is reached only after the ring is
            placed. A layer recorded for a box nobody can see would be redrawn as nothing on every
            replay of the figure.
        """
        main = framed(4326)
        mocker.patch(
            "digitalearth.static.maps.inset.reproject_coordinates",
            side_effect=RuntimeError("PROJ refused the transform"),
        )
        locator = Map(crs=3857)
        locator.mark_extent(main)
        assert locator.layer_ids == [], (
            f"a skipped box should register no layer; got {locator.layer_ids}"
        )


class TestTheLocatorExtentIsARectangle:
    """``inset(extent=...)`` frames the locator, so an extent that frames no area is refused by name (#391).

    ``set_bounds`` honours a flipped pair as an inverted axis — a deliberate 2-D-tier contract — but the
    locator's ``extent`` is the wider rectangle the inset shows, not an axis to invert. An inverted or
    zero-area extent used to be forwarded straight to ``set_bounds``, so the refusal (or, for the zero-area
    one, matplotlib's "transformation singular" warning) read ``set_bounds`` rather than ``inset()``, and
    the zero-area one was not refused at all. It is now read in ``inset()`` before the inset axes is built.
    """

    def test_an_inverted_extent_is_refused(self, framed):
        """West east of east, or south north of north, frames no rectangle.

        Args:
            framed: Factory for the framed main map.
        """
        main = framed(4326)
        with pytest.raises(ValueError, match=r"inset\(extent="):
            main.inset(extent=[60.0, 40.0, -20.0, -40.0])

    def test_an_inverted_extent_leaves_no_axes_behind(self, framed):
        """The refusal lands with the others, before ``ax.inset_axes`` is called.

        Args:
            framed: Factory for the framed main map.
        """
        main = framed(4326)
        with pytest.raises(ValueError, match=r"inset\(extent="):
            main.inset(extent=[60.0, 40.0, -20.0, -40.0])
        assert main.ax.child_axes == [], (
            f"a refused extent should leave no inset axes; got {main.ax.child_axes}"
        )

    def test_a_zero_area_extent_is_refused(self, framed):
        """``[0, 0, 0, 0]`` is a point, not a region, so it draws no locator.

        Args:
            framed: Factory for the framed main map.

        Test scenario:
            It used to be accepted and reach matplotlib as a singular limit, which expanded it and warned
            twice about a "transformation singular" — a wrong picture rather than a refusal.
        """
        main = framed(4326)
        with pytest.raises(ValueError, match=r"inset\(extent="):
            main.inset(extent=[0.0, 0.0, 0.0, 0.0])

    def test_a_valid_extent_still_frames_the_locator(self, framed):
        """A rectangle the right way round is honoured, as the lon/lat example documents.

        Args:
            framed: Factory for the framed main map.
        """
        locator = (
            framed(3857).inset(crs=4326, extent=[-20.0, -40.0, 60.0, 40.0]).locator
        )
        held = [float(value) for value in locator.ax.get_ylim()]
        assert held == [-40.0, 40.0], f"the extent should frame the locator; got {held}"


class TestSetBoundsNamesItsCaller:
    """``set_bounds(caller=)`` lets a method that frames through it put its own name on the refusal (#391).

    ``inset(extent=...)`` frames the locator through ``set_bounds``, so a ``set_bounds`` refusal reaching a
    caller who never typed ``set_bounds`` is the round-1 L2 defect ``_ExtentBox.of(caller=)`` already
    answers for the not-framed case. The keyword defaults to ``set_bounds``, preserving the wording every
    direct caller gets.
    """

    def test_the_default_caller_is_named_set_bounds(self):
        """A direct call keeps the wording it always had.

        Test scenario:
            The default must not change the message a direct ``set_bounds`` caller reads.
        """
        main = Map(crs=4326)
        with pytest.raises(ValueError, match=r"^set_bounds needs exactly 4 values"):
            main.set_bounds([1.0, 2.0, 3.0])
        main.close()

    def test_a_given_caller_is_named_instead(self):
        """And a framing method passes its own name through.

        Test scenario:
            The name a caller threads is the one the refusal quotes, so a method framing through
            ``set_bounds`` is not reported as ``set_bounds``.
        """
        main = Map(crs=4326)
        with pytest.raises(ValueError, match=r"^locator_probe needs exactly 4 values"):
            main.set_bounds([1.0, 2.0, 3.0], caller="locator_probe")
        main.close()
