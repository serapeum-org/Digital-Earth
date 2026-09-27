"""The 3-D tier's decoration surface: a title, a string at a coordinate, and the coordinate frame (#203).

`Scene3D` was the only one of the four tiers with no decoration at all. A title, a label, the extent of the
data and which way is up were reachable only through `scene.plotter`, which is the abstraction the package is
built on being stepped around — and none of it was covered, documented or reachable from
`quickmap(backend="3d")`.

These tests hold the surface that closes that, and they hold it in the two halves the tier splits decoration
into:

* **a layer** — `text()` writes a `LayerSpec` of the registered `text` kind, so it has an id, is toggled with
  `set_visible`, is taken off with `remove_layer`, and survives `to_dict()`/`from_dict()` like any other layer.
* **figure furniture** — `set_title()` is the panel's own `title` (described, and redrawn from the
  description), while `axes()` and `orientation_axes()` are drawn view state the scene re-applies to whatever
  plotter it is given, exactly as its vertical exaggeration and its camera are.

Gated on the optional ``3d`` extra (pyvista), as every module in this directory is.
"""

import numpy as np
import pytest

pv = pytest.importorskip("pyvista")

from digitalearth.base.spec import FigureSpec  # noqa: E402
from digitalearth.three_d import Scene3D  # noqa: E402
from digitalearth.three_d.capabilities import CAPABILITIES  # noqa: E402
from digitalearth.three_d.decoration import (  # noqa: E402
    SUBTITLE_ACTOR,
    TITLE_SLOT,
    axis_labels,
    axis_titles,
)

#: The corner-annotation slot a subtitle is written into — `vtkCornerAnnotation`'s upper-left, which is where
#: PyVista files text asked for at `position="upper_left"`.
SUBTITLE_SLOT = 2

#: What the issue probed `dir(Scene3D)` for and got `[]` back from.
DECORATION_WORDS = ("text", "title", "axes", "bounds")


@pytest.fixture
def scene():
    """Yield an off-screen scene, closed on the way out.

    Yields:
        The scene under test.
    """
    built = Scene3D(off_screen=True)
    yield built
    built.close()


def _decoration_names(cls) -> list:
    """Return the public decoration-shaped names a class carries.

    Args:
        cls: The class to probe.

    Returns:
        The sorted public method names whose spelling contains one of :data:`DECORATION_WORDS`.
    """
    return sorted(
        name
        for name in dir(cls)
        if not name.startswith("_")
        and any(word in name.lower() for word in DECORATION_WORDS)
    )


class TestTheTitleIsFigureFurniture:
    """`set_title()` is the panel's own `title`: described, redrawn from the description, not a layer."""

    def test_the_heading_is_recorded_on_the_panel(self, scene):
        """The figure carries the heading, so a stored figure carries it too.

        Args:
            scene: The scene under test.
        """
        scene.set_title("Elevation (m)")
        assert scene.figure_spec.panels[0].title == "Elevation (m)", (
            f"the panel's title is {scene.figure_spec.panels[0].title!r}"
        )

    def test_the_heading_reaches_the_plotter(self, scene):
        """PyVista is actually asked to draw it, read back off the corner annotation it writes.

        Args:
            scene: The scene under test.

        Test scenario:
            A title recorded on the figure and never drawn would pass every description check and produce an
            untitled picture, which is the half of #203 a reader sees. The window is opened **before** the
            call: reading `scene.plotter` afterwards builds it, and building it re-applies the scene's
            furniture, so a `set_title` that only recorded would have passed on the read alone.
        """
        opened = scene.plotter
        scene.set_title("Elevation (m)")
        drawn = opened.renderer.actors["title"].GetText(TITLE_SLOT)
        assert "Elevation (m)" in drawn, f"the plotter's title band reads {drawn!r}"

    def test_a_subtitle_is_drawn_as_a_second_actor(self, scene):
        """The second line is its own actor, so it can be sized and replaced independently.

        Args:
            scene: The scene under test.
        """
        opened = scene.plotter
        scene.set_title("Elevation (m)", subtitle="SRTM, 2024")
        drawn = opened.renderer.actors[SUBTITLE_ACTOR].GetText(SUBTITLE_SLOT)
        assert drawn == "SRTM, 2024", f"the subtitle actor reads {drawn!r}"

    def test_re_titling_without_a_subtitle_takes_the_old_one_off(self, scene):
        """A second `set_title` replaces the heading and clears a subtitle it was not given.

        Args:
            scene: The scene under test.

        Test scenario:
            Two `add_text` calls stack two actors, so a subtitle dropped from the call would stay on the
            window while the figure said nothing about it. Both calls are made on an **open** window, or the
            first subtitle is never drawn and there is nothing to leave behind.
        """
        opened = scene.plotter
        scene.set_title("First", subtitle="SRTM, 2024")
        scene.set_title("Second")
        assert SUBTITLE_ACTOR not in opened.renderer.actors, sorted(
            opened.renderer.actors
        )

    def test_a_title_set_before_the_plotter_exists_is_drawn_when_it_is_built(self):
        """Describing a scene builds no render window, and the title has to survive that.

        Test scenario:
            #290 made the plotter lazy. A title that drew straight onto `self.plotter` would force the window
            open at `set_title()`, which is the cost being avoided; one that only ever drew there would be
            missing from the picture when the window was finally built.
        """
        lazy = Scene3D(off_screen=True)
        lazy.set_title("Elevation (m)")
        assert lazy._plotter is None, "set_title() built a render window"
        drawn = lazy.plotter.renderer.actors["title"].GetText(TITLE_SLOT)
        assert "Elevation (m)" in drawn, f"the plotter's title band reads {drawn!r}"
        lazy.close()

    def test_a_stored_figure_carries_the_heading(self):
        """`to_dict()`/`from_dict()` keep the heading, and `from_figure` brings it back onto the scene.

        Test scenario:
            The round trip is what makes the title *described* rather than merely drawn. A title held only in
            drawing state would be gone the moment a figure was written down.
        """
        first = Scene3D(off_screen=True)
        stored = first.set_title("Elevation (m)").figure_spec.to_dict()
        first.close()
        second = Scene3D.from_figure(FigureSpec.from_dict(stored), off_screen=True)
        assert second.figure_spec.panels[0].title == "Elevation (m)", (
            f"the redrawn panel's title is {second.figure_spec.panels[0].title!r}"
        )
        second.close()

    def test_a_figure_drawn_into_an_open_scene_brings_its_heading_with_it(self, scene):
        """`draw_figure` draws the panel title it was handed, on a scene that already has a window.

        Args:
            scene: The scene under test.

        Test scenario:
            Nothing in the layer diff re-reads the panel's title, so a figure drawn into a scene whose window
            is already open would install a heading nobody drew. Opening the window **first** is what makes
            this about `draw_figure` rather than about the lazy build re-applying the furniture.
        """
        written = Scene3D(off_screen=True)
        stored = written.set_title("Elevation (m)").figure_spec
        written.close()
        opened = scene.plotter
        scene.draw_figure(stored)
        drawn = opened.renderer.actors["title"].GetText(TITLE_SLOT)
        assert "Elevation (m)" in drawn, f"the redrawn title band reads {drawn!r}"

    def test_a_title_survives_a_plotter_swap(self, scene):
        """Handing the scene another window keeps its decoration, as it keeps its scale and camera.

        Args:
            scene: The scene under test.

        Test scenario:
            Review M7 fixed exactly this for the view scale and the camera: only the lazy build applied them,
            so swapping a plotter silently reverted the scene. Decoration is the same kind of state.
        """
        scene.set_title("Elevation (m)")
        scene.plotter = pv.Plotter(off_screen=True)
        drawn = scene.plotter.renderer.actors["title"].GetText(TITLE_SLOT)
        assert "Elevation (m)" in drawn, f"the new window's title band reads {drawn!r}"

    def test_the_title_is_not_a_layer(self, scene):
        """A heading is furniture: it is not in the tree and a layer's removal cannot take it away.

        Args:
            scene: The scene under test.
        """
        scene.set_title("Elevation (m)")
        assert scene.layer_ids == [], f"set_title() added the layers {scene.layer_ids}"

    def test_the_call_chains(self, scene):
        """`set_title` returns the scene, which is what the Core declares for the name.

        Args:
            scene: The scene under test.
        """
        assert scene.set_title("Elevation (m)") is scene, (
            "set_title() must return the scene so decoration chains"
        )

    def test_a_heading_that_is_not_a_string_is_refused(self, scene):
        """`None` drawn as `"None"` is worse than no title, so it is refused at the call.

        Args:
            scene: The scene under test.
        """
        with pytest.raises(TypeError, match="needs the heading as a string"):
            scene.set_title(None)


class TestTextIsALayer:
    """`text()` is a layer in the tree: an id, a kind, a visibility, and a description that round-trips."""

    def test_the_tier_declares_the_kind_it_writes(self):
        """A layer of a kind the tier does not declare could not be drawn back from a stored figure."""
        assert "text" in CAPABILITIES.kinds, sorted(CAPABILITIES.kinds)

    def test_a_string_at_a_coordinate_is_described_as_a_text_layer(self, scene):
        """The call leaves one `text` layer in the figure, named by its kind.

        Args:
            scene: The scene under test.
        """
        scene.text(3.0, 4.0, "peak", z=9.0)
        layer = scene.figure_spec.layers.get("text-1")
        assert layer.kind == "text", f"text() recorded a {layer.kind!r} layer"
        assert layer.symbology.props["s"] == "peak", dict(layer.symbology.props)

    def test_the_anchor_is_recorded_in_three_dimensions(self, scene):
        """A 3-D label needs a z, and the one the caller gave is the one described.

        Args:
            scene: The scene under test.
        """
        scene.text(3.0, 4.0, "peak", z=9.0)
        props = scene.figure_spec.layers.get("text-1").symbology.props
        assert (props["x"], props["y"], props["z"]) == (3.0, 4.0, 9.0), dict(props)

    def test_the_label_reaches_the_plotter(self, scene):
        """The actor the drawer produced is on the plotter and is addressable by the layer's id.

        Args:
            scene: The scene under test.
        """
        scene.text(0.0, 0.0, "here")
        anchor = scene.mesh_of("text-1").points
        assert anchor.shape == (1, 3), (
            f"a label anchors at one point, got {anchor.shape}"
        )
        assert scene.actor_of("text-1") is not None, "text() drew no actor"

    def test_the_layer_is_hidden_and_shown_by_id(self, scene):
        """`set_visible` reaches a text layer's actor, as it reaches any other layer's.

        Args:
            scene: The scene under test.
        """
        scene.text(0.0, 0.0, "here", name="mark")
        scene.set_visible("mark", False)
        hidden = scene.renderer.is_visible("mark")
        scene.set_visible("mark", True)
        assert (hidden, scene.renderer.is_visible("mark")) == (False, True), (
            f"a text layer toggled to {hidden} and then to {scene.renderer.is_visible('mark')}"
        )

    def test_the_layer_is_taken_off_by_id(self, scene):
        """`remove_layer` drops the description and the actor together.

        Args:
            scene: The scene under test.
        """
        scene.text(0.0, 0.0, "here", name="mark")
        scene.remove_layer("mark")
        assert scene.layer_ids == [], f"the scene still describes {scene.layer_ids}"
        assert "mark" not in scene.renderer.drawn, sorted(scene.renderer.drawn)

    def test_a_named_label_is_filed_under_the_name(self, scene):
        """A caller's name is the layer's id, as it is for every builder on this tier.

        Args:
            scene: The scene under test.
        """
        scene.text(0.0, 0.0, "here", name="summit")
        assert scene.layer_ids == ["summit"], scene.layer_ids

    def test_the_described_label_draws_again_from_its_dict(self):
        """A figure written with `to_dict()` draws the same label back in another scene.

        Test scenario:
            The round trip is the reason this tier describes anything, so a decoration that draws but is not
            described would pass every other check here and still be lost the moment a figure was stored.
        """
        first = Scene3D(off_screen=True)
        first.text(1.0, 2.0, "peak", z=3.0, name="summit")
        stored = first.figure_spec.to_dict()
        first.close()
        second = Scene3D.from_figure(FigureSpec.from_dict(stored), off_screen=True)
        drawn = second.mesh_of("summit").points
        assert second.layer_ids == ["summit"], second.layer_ids
        assert drawn.tolist() == [[1.0, 2.0, 3.0]], drawn.tolist()
        second.close()

    def test_a_point_in_another_crs_is_placed_in_the_scene_s(self):
        """A lon/lat anchor is reprojected into the display CRS through pyramids before it is drawn.

        Test scenario:
            The contract's Tier-2 `text` takes a `crs`, and it means the same thing on all four tiers: the
            point is measured in `crs` and drawn where the scene draws. A tier that dropped it would place a
            degree pair among UTM metres, which puts the label at the origin of a scene metres away.
        """
        placed = Scene3D(off_screen=True, crs=32618)
        placed.text(-76.0, 38.0, "chesapeake")
        drawn = placed.mesh_of("text-1").points[0]
        assert abs(drawn[0]) > 1000.0, f"the anchor was drawn at {drawn.tolist()}"
        assert abs(drawn[1]) > 1000.0, f"the anchor was drawn at {drawn.tolist()}"
        placed.close()

    def test_the_string_is_not_optional(self, scene):
        """Forgetting the string is a `TypeError` naming what is missing, not a label reading `None`.

        Args:
            scene: The scene under test.
        """
        with pytest.raises(TypeError, match="needs the string to draw"):
            scene.text(0.0, 0.0)

    def test_a_non_finite_anchor_is_refused(self, scene):
        """A `NaN` coordinate could not be written down, so it is refused at the call.

        Args:
            scene: The scene under test.
        """
        with pytest.raises(ValueError, match="must be a finite number"):
            scene.text(float("nan"), 0.0, "nowhere")


class TestTheSurfaceExists:
    """#203's own probe, as an assertion: the tier had nothing decoration-shaped on it at all."""

    def test_the_scene_names_a_title_a_text_and_a_coordinate_frame(self):
        """`dir(Scene3D)` answers with the decoration verbs rather than with nothing.

        Test scenario:
            The issue's evidence is a list comprehension over `dir(Scene3D)` that returns `[]`. That is the
            measurement this replaces, so it is written the same way — over the composed class, not over the
            mixin, because composition is what a caller holds.
        """
        assert _decoration_names(Scene3D) == [
            "axes",
            "orientation_axes",
            "set_title",
            "text",
        ], f"the 3-D tier's decoration surface is {_decoration_names(Scene3D)}"


class TestTheAxisTitlesComeFromTheData:
    """#203's acceptance criterion: meaningful axis names, never PyVista's `X`/`Y`/`Z`."""

    def test_a_projected_crs_names_its_own_axes(self):
        """A UTM zone calls its axes Easting and Northing, and the CRS is what says so."""
        assert axis_titles(32618) == ("Easting", "Northing", "Elevation"), axis_titles(
            32618
        )

    def test_a_geographic_crs_is_read_east_first_however_it_orders_its_axes(self):
        """EPSG:4326 declares latitude first; a scene draws x east, so the axes are matched by direction.

        Test scenario:
            Taking the CRS's axes in the order it declares them labels a lon/lat scene back to front — the x
            axis reading "Geodetic latitude". This is the probe that the direction, not the position, decides.
        """
        assert axis_titles(4326)[:2] == (
            "Geodetic longitude",
            "Geodetic latitude",
        ), axis_titles(4326)

    def test_a_three_dimensional_crs_names_its_vertical_axis_too(self):
        """EPSG:4979 has an `up` axis of its own, and it is read rather than defaulted."""
        assert axis_titles(4979)[2] == "Ellipsoidal height", axis_titles(4979)

    def test_a_scene_with_no_crs_is_still_not_labelled_with_array_indices(self):
        """A bare array has no CRS to read, and `X`/`Y`/`Z` is what #203 asked not to see."""
        assert axis_titles(None) == ("Easting", "Northing", "Elevation"), axis_titles(
            None
        )

    def test_an_unreadable_crs_is_not_guessed_at(self):
        """Anything pyramids cannot read as a CRS falls back rather than raising."""
        assert axis_titles("not-a-crs") == ("Easting", "Northing", "Elevation"), (
            axis_titles("not-a-crs")
        )

    def test_the_triad_labels_are_short_forms_rather_than_the_full_names(self):
        """A corner widget has room for `Lon`, not for `Geodetic longitude`."""
        assert axis_labels(4326) == ("Lon", "Lat", "Up"), axis_labels(4326)

    def test_a_projected_scenes_triad_is_labelled_for_a_survey(self):
        """Easting and Northing abbreviate to E and N, which is what a projected scene's triad reads."""
        assert axis_labels(32618) == ("E", "N", "Up"), axis_labels(32618)


class TestTheAxesBoxIsFigureFurniture:
    """`axes()` draws PyVista's labelled bounds box, titled from the scene's own display CRS."""

    def test_the_box_is_drawn_and_titled_from_the_display_crs(self):
        """A projected scene's box reads Easting / Northing / Elevation, not X / Y / Z."""
        placed = Scene3D(off_screen=True, crs=32618)
        placed.axes()
        box = placed.plotter.renderer.cube_axes_actor
        titles = (box.GetXTitle(), box.GetYTitle(), box.GetZTitle())
        assert titles == ("Easting", "Northing", "Elevation"), (
            f"the box is titled {titles}"
        )
        placed.close()

    def test_the_callers_own_titles_win(self, scene):
        """A caller who knows what the axis is gets to say so.

        Args:
            scene: The scene under test.
        """
        scene.axes(xtitle="Chainage (m)")
        assert scene.plotter.renderer.cube_axes_actor.GetXTitle() == "Chainage (m)", (
            scene.plotter.renderer.cube_axes_actor.GetXTitle()
        )

    def test_the_box_is_taken_off_again(self, scene):
        """`axes(False)` removes it, so the switch is one keyword rather than two methods.

        Args:
            scene: The scene under test.

        Test scenario:
            The window is opened **first**: reading `scene.plotter` afterwards builds it and re-applies the
            furniture, which on a scene that has dropped the box produces an empty window whether or not
            `axes(False)` ever removed anything.
        """
        opened = scene.plotter
        scene.axes()
        scene.axes(False)
        assert opened.renderer.cube_axes_actor is None, (
            "axes(False) left the bounds box on the plotter"
        )

    def test_a_removed_box_is_not_redrawn_by_a_later_dressing(self, scene):
        """Turning it off has to survive the next time the scene dresses a plotter.

        Args:
            scene: The scene under test.

        Test scenario:
            The furniture is re-applied on every plotter the scene is given, so "off" has to be an absence
            in what it re-applies, not just a call that happened once.
        """
        # Open the window, so the box is really drawn before it is removed.
        assert scene.plotter is not None, "the scene built no render window"
        scene.axes()
        scene.axes(False)
        scene.plotter = pv.Plotter(off_screen=True)
        assert scene.plotter.renderer.cube_axes_actor is None, (
            "the removed bounds box came back on the next window"
        )

    def test_the_box_survives_a_plotter_swap(self, scene):
        """Handing the scene another window keeps its box, as it keeps its scale and camera.

        Args:
            scene: The scene under test.
        """
        scene.axes(xtitle="Chainage (m)")
        scene.plotter = pv.Plotter(off_screen=True)
        box = scene.plotter.renderer.cube_axes_actor
        assert box is not None, "the new window has no bounds box"
        assert box.GetXTitle() == "Chainage (m)", box.GetXTitle()

    def test_the_box_is_not_a_layer(self, scene):
        """It is drawn round the data rather than being data, so it is not in the tree.

        Args:
            scene: The scene under test.
        """
        scene.axes()
        assert scene.layer_ids == [], f"axes() added the layers {scene.layer_ids}"

    def test_the_call_chains(self, scene):
        """`axes` returns the scene, so decoration reads as one expression.

        Args:
            scene: The scene under test.
        """
        assert scene.axes() is scene, (
            "axes() must return the scene so decoration chains"
        )


class TestTheOrientationTriadIsFigureFurniture:
    """`orientation_axes()` is the corner triad: which way is up, in the scene's own terms."""

    def test_the_triad_is_labelled_from_the_display_crs(self):
        """A geographic scene's triad reads Lon / Lat / Up rather than X / Y / Z."""
        placed = Scene3D(off_screen=True, crs=4326)
        placed.orientation_axes()
        triad = placed.plotter.renderer.axes_actor
        labels = (
            triad.GetXAxisLabelText(),
            triad.GetYAxisLabelText(),
            triad.GetZAxisLabelText(),
        )
        assert labels == ("Lon", "Lat", "Up"), f"the triad is labelled {labels}"
        placed.close()

    def test_the_callers_own_labels_win(self, scene):
        """A caller who knows what the axis is gets to say so, here as well.

        Args:
            scene: The scene under test.
        """
        scene.orientation_axes(xlabel="Downstream")
        assert scene.plotter.renderer.axes_actor.GetXAxisLabelText() == "Downstream", (
            scene.plotter.renderer.axes_actor.GetXAxisLabelText()
        )

    def test_the_triad_is_shown(self, scene):
        """The widget is enabled, which is what PyVista reports for a triad that is on the window.

        Args:
            scene: The scene under test.
        """
        scene.orientation_axes()
        assert scene.plotter.renderer.axes_enabled is True, (
            "orientation_axes() left the widget disabled"
        )

    def test_the_triad_is_hidden_again(self, scene):
        """`orientation_axes(False)` hides it, matching `axes(False)`.

        Args:
            scene: The scene under test.

        Test scenario:
            The window is opened **first**, for the reason :meth:`axes`' own removal probe states: a scene
            with no window yet would report a hidden triad whether or not anything hid it.
        """
        opened = scene.plotter
        scene.orientation_axes()
        scene.orientation_axes(False)
        assert opened.renderer.axes_enabled is False, (
            "orientation_axes(False) left the widget enabled"
        )

    def test_the_triad_survives_a_plotter_swap(self, scene):
        """The triad is scene state, so the next window gets it too.

        Args:
            scene: The scene under test.
        """
        scene.orientation_axes(xlabel="Downstream")
        scene.plotter = pv.Plotter(off_screen=True)
        assert scene.plotter.renderer.axes_actor.GetXAxisLabelText() == "Downstream", (
            scene.plotter.renderer.axes_actor.GetXAxisLabelText()
        )

    def test_a_hidden_triad_is_not_brought_back_by_a_later_dressing(self, scene):
        """Hiding it has to survive the next time the scene dresses a plotter, as `axes(False)` does.

        Args:
            scene: The scene under test.

        Test scenario:
            The other half of the same rule: "off" is an absence in what the scene re-applies, not a call
            that happened once on one window.
        """
        # Open the window, so the triad is really shown before it is hidden.
        assert scene.plotter is not None, "the scene built no render window"
        scene.orientation_axes()
        scene.orientation_axes(False)
        scene.plotter = pv.Plotter(off_screen=True)
        assert scene.plotter.renderer.axes_enabled is False, (
            "the hidden triad came back on the next window"
        )

    def test_the_triad_is_not_a_layer(self, scene):
        """It sits in a corner of the window rather than in the scene's coordinates.

        Args:
            scene: The scene under test.
        """
        scene.orientation_axes()
        assert scene.layer_ids == [], (
            f"orientation_axes() added the layers {scene.layer_ids}"
        )

    def test_the_call_chains(self, scene):
        """`orientation_axes` returns the scene, so the whole surface chains.

        Args:
            scene: The scene under test.
        """
        assert scene.orientation_axes() is scene, (
            "orientation_axes() must return the scene so decoration chains"
        )
