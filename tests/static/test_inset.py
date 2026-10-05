"""Tests for digitalearth.static.maps.inset — the locator map (ST-19).

Two things carry the risk and are covered closely. The **extent box is drawn in the locator's own CRS**, and
a box reprojected by its corners alone lands wrong — the EPSG:3031 case below is the proof: all four corners
of a polar-stereographic square sit at the *same* latitude, so a corner-only box has zero height. And a
borrowed axes is still closed by ``Scene.close`` (#371), so what closing the locator does to the caller's
figure is measured rather than assumed.
"""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from pyramids.base.crs import reproject_coordinates

from digitalearth.static import Map

#: A box in EPSG:3031 (Antarctic Polar Stereographic) whose four corners all reproject to one latitude.
POLAR_BOX = (-2.0e6, -2.0e6, 2.0e6, 2.0e6)

#: The lon/lat rectangle the locator's extent box is checked against in the same-CRS cases.
PLAIN_BOX = (2.0, 3.0, 8.0, 9.0)


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
        ring = np.asarray(Map(crs=4326).mark_extent(main).get_xy(), dtype=float)
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
        marked = Map(crs=4326).mark_extent(main)
        assert marked is not None, (
            "a map framed by its own data should have an extent to mark"
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
        ring = np.asarray(Map(crs=4326).mark_extent(main).get_xy(), dtype=float)
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
        ring = np.asarray(Map(crs=4326).mark_extent(main).get_xy(), dtype=float)
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
        ring = np.asarray(Map(crs=4326).mark_extent(main).get_xy(), dtype=float)
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
        marked = Map(crs=4326).mark_extent(main)
        assert marked is None, (
            f"an unplaceable extent should mark nothing; got {marked}"
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

        patch = Map(crs=4326).mark_extent(framed(4326), edgecolor="#00ff00")
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
        patch = Map(crs=4326).mark_extent(framed(4326))
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

    def test_closing_the_locator_closes_the_parents_figure(self, framed):
        """``Scene.close`` on a borrowed axes closes the whole figure (#371) — measured, not assumed.

        Args:
            framed: Factory for the framed main map.

        Test scenario:
            The locator holds the parent's figure, and ``close`` calls ``pyplot.close`` on whatever
            figure a scene holds. So closing the locator takes the map with it, which is why ``inset``
            must not be context-managed and why the docstring says so.
        """
        main = framed(4326)
        main.inset().locator.close()
        assert main.fig.number not in plt.get_fignums(), (
            "closing the locator is expected to close the parent figure too (#371)"
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
        marked = Map(crs=3857).mark_extent(main)
        assert marked is None, f"a refused transform should mark nothing; got {marked}"

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
